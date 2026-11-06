import argparse
import json
import h5py
import imageio
import numpy as np
from copy import deepcopy
import os
from pathlib import Path
import time

import torch
from scipy.spatial.transform import Rotation as R

import mimicgen
import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig


def robosuite_osc_scale_action(action):
    """
    复现 robosuite OperationalSpaceController.scale_action() 在默认 OSC_POSE 下的语义:
      xyz: clip[-1,1] -> * 0.05
      rot: clip[-1,1] -> * 0.5
    """
    a = np.array(action, dtype=np.float32).reshape(-1).copy()
    if a.size < 7:
        raise ValueError(f"Expected >= 7 dims, got {a.size}")

    a[:6] = np.clip(a[:6], -1.0, 1.0)
    a[:3] *= 0.05
    a[3:6] *= 0.5
    return a


def deoxys_preprocess_action(action, disable_rotation=False, gripper_threshold=0.0):
    """
    送给 deoxys 前的动作预处理：
    - 和 robosuite 一样先 clip 到 [-1, 1]
    - gripper 二值化
    - 可选禁用旋转
    """
    a = np.array(action, dtype=np.float32).reshape(-1).copy()
    if a.size < 7:
        raise ValueError(f"Expected >= 7 dims, got {a.size}")

    a = a[:7]
    a[:6] = np.clip(a[:6], -1.0, 1.0)


    a[6] = 1.0 if a[6] > gripper_threshold else -1.0
    return a.astype(np.float32)


def deoxys_scaled_action(action, controller_cfg):
    """
    复现 deoxys FrankaInterface.control() 内部的 scaling，仅用于打印。
    """
    a = np.array(action, dtype=np.float32).reshape(-1).copy()
    a[:3] *= float(controller_cfg.action_scale.translation)
    a[3:6] *= float(controller_cfg.action_scale.rotation)
    return a


def get_eef_pose_mat(robot_interface):
    if robot_interface is None or len(robot_interface._state_buffer) == 0:
        return None
    return np.array(robot_interface.last_eef_pose, dtype=np.float64)


def pose_delta_from_mats(T_prev, T_curr):
    if T_prev is None or T_curr is None:
        return None, None
    dpos = (T_curr[:3, 3] - T_prev[:3, 3]).astype(np.float32)
    dR = T_curr[:3, :3] @ T_prev[:3, :3].T
    drotvec = R.from_matrix(dR).as_rotvec().astype(np.float32)
    return dpos, drotvec


def rollout(
    policy,
    env,
    horizon,
    render=False,
    video_writer=None,
    video_skip=5,
    return_obs=False,
    camera_names=None,
    robot_interface=None,
    controller_type="OSC_POSE",
    controller_cfg=None,
    send_to_real=False,
    disable_rotation_real=True,
    print_action_compare=False,
):
    """
    rollout with optional mirrored action sending to real robot
    """
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()

    results = {}
    video_count = 0
    total_reward = 0.0
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    if return_obs:
        traj.update(dict(obs=[], next_obs=[]))

    success = False

    try:
        for step_i in range(horizon):
            # -----------------------------
            # 1) get action from policy
            # -----------------------------
            act = policy(ob=obs)
            act = np.array(act, dtype=np.float32).reshape(-1)

            # -----------------------------
            # 2) sim-side interpreted action
            # -----------------------------
            sim_scaled_action = robosuite_osc_scale_action(act)

            # -----------------------------
            # 3) real-side mirrored action
            # -----------------------------
            real_send_action = deoxys_preprocess_action(
                act,
                disable_rotation=disable_rotation_real,
                gripper_threshold=0.0,
            )

            real_scaled = None
            real_dpos = None
            real_drot = None

            if send_to_real and (robot_interface is not None):
                T_before = get_eef_pose_mat(robot_interface)

                robot_interface.control(
                    controller_type=controller_type,
                    action=real_send_action,
                    controller_cfg=controller_cfg,
                )

                T_after = get_eef_pose_mat(robot_interface)
                real_dpos, real_drot = pose_delta_from_mats(T_before, T_after)
                real_scaled = deoxys_scaled_action(real_send_action, controller_cfg)

            if print_action_compare:
                print(f"\n[step {step_i}]")
                print("policy_action      :", np.round(act, 4))
                print("sim_scaled_action  :", np.round(sim_scaled_action, 4))
                print("real_send_action   :", np.round(real_send_action, 4))
                if real_scaled is not None:
                    print("real_scaled_action :", np.round(real_scaled, 4))
                if real_dpos is not None:
                    print("real_measured_dpos :", np.round(real_dpos, 4))
                    print("real_measured_drot :", np.round(real_drot, 4))

            # -----------------------------
            # 4) step simulation
            # -----------------------------
            next_obs, r, done, _ = env.step(act)

            total_reward += r
            success = env.is_success()["task"]

            # visualization
            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(
                            env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name)
                        )
                    video_img = np.concatenate(video_img, axis=1)
                    video_writer.append_data(video_img)
                video_count += 1

            # collect transition
            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            traj["states"].append(state_dict["states"])
            if return_obs:
                traj["obs"].append(obs)
                traj["next_obs"].append(next_obs)

            if done or success:
                break

            obs = next_obs
            state_dict = env.get_state()

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))

    stats = dict(Return=total_reward, Horizon=(step_i + 1), Success_Rate=float(success))

    if return_obs:
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return stats, traj


def run_trained_agent(args):
    write_video = (args.video_path is not None)
    assert not (args.render and write_video)
    if args.render:
        assert len(args.camera_names) == 1

    ckpt_path = args.agent
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=ckpt_path,
        device=device,
        verbose=True,
    )

    rollout_num_episodes = args.n_rollouts
    rollout_horizon = args.horizon
    if rollout_horizon is None:
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        rollout_horizon = config.experiment.rollout.horizon

    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        env_name=args.env,
        render=args.render,
        render_offscreen=write_video,
        verbose=True,
    )

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # video writer
    video_writer = None
    video_dir = None
    tmp_video_path = None
    if write_video:
        video_dir = Path(args.video_path)
        video_dir.mkdir(parents=True, exist_ok=True)
        tmp_video_path = str(video_dir / "dr_tmp.mp4")
        video_writer = imageio.get_writer(tmp_video_path, fps=20)

    # dataset writer
    write_dataset = (args.dataset_path is not None)
    if write_dataset:
        data_writer = h5py.File(args.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0

    # real robot init
    robot_interface = None
    controller_cfg = None
    if args.send_to_real:
        controller_cfg = YamlConfig(
            os.path.join(config_root, args.real_controller_cfg)
        ).as_easydict()

        robot_interface = FrankaInterface(
            os.path.join(config_root, "charmander.yml"),
            use_visualizer=False,
        )

        print("waiting for real robot state...")
        while len(robot_interface._state_buffer) == 0:
            time.sleep(0.05)
        print("real robot state ready")
        print("real controller_type:", args.real_controller_type)
        print("real action_scale:", controller_cfg.action_scale)

    rollout_stats = []
    try:
        for i in range(rollout_num_episodes):
            stats, traj = rollout(
                policy=policy,
                env=env,
                horizon=rollout_horizon,
                render=args.render,
                video_writer=video_writer,
                video_skip=args.video_skip,
                return_obs=(write_dataset and args.dataset_obs),
                camera_names=args.camera_names,
                robot_interface=robot_interface,
                controller_type=args.real_controller_type,
                controller_cfg=controller_cfg,
                send_to_real=args.send_to_real,
                disable_rotation_real=args.disable_rotation_real,
                print_action_compare=args.print_action_compare,
            )
            rollout_stats.append(stats)

            if write_dataset:
                ep_data_grp = data_grp.create_group("demo_{}".format(i))
                ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
                ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
                ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
                ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
                if args.dataset_obs:
                    for k in traj["obs"]:
                        ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                        ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

                if "model" in traj["initial_state_dict"]:
                    ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"]
                ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0]
                total_samples += traj["actions"].shape[0]

    finally:
        if robot_interface is not None:
            try:
                zero_action = np.zeros(7, dtype=np.float32)
                for _ in range(5):
                    robot_interface.control(
                        controller_type=args.real_controller_type,
                        action=zero_action,
                        controller_cfg=controller_cfg,
                    )
                    time.sleep(0.05)
            except Exception:
                pass

            try:
                robot_interface.close()
            except Exception:
                pass

    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = {k: np.mean(rollout_stats[k]) for k in rollout_stats}
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])
    print("Average Rollout Stats")
    print(json.dumps(avg_rollout_stats, indent=4))

    if write_video:
        video_writer.close()
        sr = float(avg_rollout_stats.get("Success_Rate", 0.0))
        sr_pct = int(round(sr * 100))

        target = video_dir / f"dr_sr{sr_pct:03d}.mp4"
        if target.exists():
            k = 1
            while True:
                cand = video_dir / f"dr_sr{sr_pct:03d}_{k}.mp4"
                if not cand.exists():
                    target = cand
                    break
                k += 1

        os.replace(tmp_video_path, str(target))
        print(f"Wrote video to {target}")

    if write_dataset:
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4)
        data_writer.close()
        print("Wrote dataset trajectories to {}".format(args.dataset_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="path to saved checkpoint pth file",
    )

    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=1,
        help="number of rollouts",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=400,
        help="override maximum horizon of rollout",
    )

    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="override env name from checkpoint",
    )

    parser.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="render rollouts to this video path",
    )

    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=["agentview"],
        help="camera names for rendering",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="optional hdf5 output path",
    )

    parser.add_argument(
        "--dataset_obs",
        action='store_true',
        help="include observations in output dataset",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="set seed for rollouts",
    )

    parser.add_argument(
        "--send_to_real",
        action="store_true",
        help="also send the same policy action to the real robot",
    )

    parser.add_argument(
        "--real_controller_type",
        type=str,
        default="OSC_POSE",
        help="real robot controller type",
    )

    parser.add_argument(
        "--real_controller_cfg",
        type=str,
        default="osc-pose-controller.yml",
        help="controller yaml under deoxys config_root",
    )

    parser.add_argument(
        "--disable_rotation_real",
        action="store_true",
        help="disable rotation action when sending to real robot",
    )

    parser.add_argument(
        "--print_action_compare",
        action="store_true",
        help="print sim/real action comparison each step",
    )

    args = parser.parse_args()
    run_trained_agent(args)