"""
The main script for evaluating a policy in an environment.

Args:
    agent (str): path to saved checkpoint pth file

    horizon (int): if provided, override maximum horizon of rollout from the one 
        in the checkpoint

    env (str): if provided, override name of env from the one in the checkpoint,
        and use it for rollouts

    render (bool): if flag is provided, use on-screen rendering during rollouts

    video_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    camera_names (str or [str]): camera name(s) to use for rendering on-screen or to video

    dataset_path (str): if provided, an hdf5 file will be written at this path with the
        rollout data

    dataset_obs (bool): if flag is provided, and @dataset_path is provided, include 
        possible high-dimensional observations in output dataset hdf5 file (by default,
        observations are excluded and only simulator states are saved).

    seed (int): if provided, set seed for rollouts

Example usage:

    # Evaluate a policy with 50 rollouts of maximum horizon 400 and save the rollouts to a video.
    # Visualize the agentview and wrist cameras during the rollout.
    
    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --video_path /path/to/output.mp4 \
        --camera_names agentview robot0_eye_in_hand 

    # Write the 50 agent rollouts to a new dataset hdf5.

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5 --dataset_obs 

    # Write the 50 agent rollouts to a new dataset hdf5, but exclude the dataset observations
    # since they might be high-dimensional (they can be extracted again using the
    # dataset_states_to_obs.py script).

    python run_trained_agent.py --agent /path/to/model.pth \
        --n_rollouts 50 --horizon 400 --seed 0 \
        --dataset_path /path/to/output.hdf5
"""

import argparse
import os
import json
import h5py
import imageio
import sys
import time
import traceback
import numpy as np
from copy import deepcopy
from tqdm import tqdm

import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.log_utils import log_warning
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy
from robomimic.scripts.playback_dataset import DEFAULT_CAMERAS


def rollout(policy, env, horizon, render=False, video_writer=None, video_skip=5, return_obs=False, camera_names=None, real=False, rate_measure=None, initial_state=None):
    """
    Helper function to carry out rollouts. Supports on-screen rendering, off-screen rendering to a video, 
    and returns the rollout trajectory.
    """
    rollout_timestamp = time.time()
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    policy.start_episode()
    state_dict = dict()

    # If an initial state is provided from the dataset, reset directly to it
    if initial_state is not None and not real:
        state_dict = {"states": np.array(initial_state)}
        obs = env.reset_to(state_dict)
    else:
        obs = env.reset()
        if real:
            input("ready for next eval? hit enter to continue")
        else:
            state_dict = env.get_state()
            # hack that is necessary for robosuite tasks for deterministic action playback
            obs = env.reset_to(state_dict)

    results = {}
    video_count = 0  # video frame counter
    total_reward = 0.
    got_exception = False
    success = env.is_success()["task"]
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))
    try:
        for step_i in range(horizon):
            # HACK: some keys on real robot do not have a shape (and then they get frame stacked)
            for k in obs:
                if len(obs[k].shape) == 1:
                    obs[k] = obs[k][..., None] 

            # get action from policy
            t1 = time.time()
            act = policy(ob=obs)
            t2 = time.time()
            if real and (not env.base_env.controller_type == "JOINT_IMPEDANCE") and (policy.policy.global_config.algo_name != "diffusion_policy"):
                # joint impedance actions and diffusion policy actions are absolute in the real world
                act = np.clip(act, -1., 1.)

            if rate_measure is not None:
                rate_measure.measure()
                print("time: {}s".format(t2 - t1))
                # dummy reward and done
                r = 0.
                done = False
                next_obs = obs
            else:
                # play action
                next_obs, r, done, _ = env.step(act)

            # compute reward
            total_reward += r
            success = env.is_success()["task"]

            # visualization
            if render:
                env.render(mode="human", camera_name=camera_names[0])
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                    video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                    video_writer.append_data(video_img)
                video_count += 1

            # collect transition
            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            if not real:
                traj["states"].append(state_dict["states"])
            if return_obs:
                traj["obs"].append(ObsUtils.unprocess_obs_dict(obs))
                traj["next_obs"].append(ObsUtils.unprocess_obs_dict(next_obs))

            # break if done or if success
            if done or success:
                break

            # update for next iter
            obs = deepcopy(next_obs)
            if not real:
                state_dict = env.get_state()

    except env.rollout_exceptions as e:
        print("WARNING: got rollout exception {}".format(e))
        got_exception = True

    stats = dict(
        Return=total_reward,
        Horizon=(step_i + 1),
        Success_Rate=float(success),
        Exception_Rate=float(got_exception),
        time=(time.time() - rollout_timestamp),
    )

    # Per-nut tracking for NutAssemblyDualSquare
    rs_env = env
    while hasattr(rs_env, 'env'):
        rs_env = rs_env.env
    if hasattr(rs_env, 'which_nut_assembled'):
        stats['assembled_nut'] = rs_env.which_nut_assembled()
        stats['assembly_yaw_deg'] = rs_env.assembled_nut_yaw()

    if return_obs:
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
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

    rate_measure = None
    if args.hz is not None:
        import RobotTeleop
        from RobotTeleop.utils import Rate, RateMeasure, Timers
        rate_measure = RateMeasure(name="control_rate_measure", freq_threshold=args.hz)
    
    algo_name, ckpt_dict = FileUtils.algo_name_from_checkpoint(ckpt_path=args.agent)

    if args.dp_eval_steps is not None:
        assert algo_name == "diffusion_policy"
        log_warning("setting @num_inference_steps to {}".format(args.dp_eval_steps))

        tmp_config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        with tmp_config.values_unlocked():
            if tmp_config.algo.ddpm.enabled:
                tmp_config.algo.ddpm.num_inference_timesteps = args.dp_eval_steps
            elif tmp_config.algo.ddim.enabled:
                tmp_config.algo.ddim.num_inference_timesteps = args.dp_eval_steps
            else:
                raise Exception("should not reach here")
        ckpt_dict['config'] = tmp_config.dump()

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_dict=ckpt_dict, device=device, verbose=True)

    rollout_num_episodes = args.n_rollouts
    rollout_horizon = args.horizon
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    if rollout_horizon is None:
        rollout_horizon = config.experiment.rollout.horizon

    if (algo_name == "diffusion_policy") and EnvUtils.is_real_robot_gprs_env(env_meta=ckpt_dict["env_metadata"]):
        ckpt_dict["env_metadata"]["env_kwargs"]["absolute_actions"] = True

    # Parse and cache evaluation states from dataset if provided
    initial_states = None
    if args.eval_dataset is not None:
        if not os.path.isfile(args.eval_dataset):
            raise FileNotFoundError(f"Evaluation dataset missing: {args.eval_dataset}")
        print(f"[Deterministic Eval] Loading rollout initialization states from: {args.eval_dataset}")
        with h5py.File(args.eval_dataset, "r") as f_eval:
            demos = sorted(list(f_eval["data"].keys()), key=lambda x: int(x.split("_")[-1]))
            initial_states = [f_eval[f"data/{d}/states"][0] for d in demos]
        
        # Keep evaluations safely bounded by the dataset items available
        rollout_num_episodes = min(rollout_num_episodes, len(initial_states))
        print(f"[Deterministic Eval] Adjusted evaluation rollout iterations to: {rollout_num_episodes}")

    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict, 
        env_name=args.env, 
        render=args.render, 
        render_offscreen=(args.video_path is not None), 
        verbose=True,
    )

    if args.camera_names is None:
        env_type = EnvUtils.get_env_type(env=env)
        args.camera_names = DEFAULT_CAMERAS[env_type]
    if args.render:
        assert len(args.camera_names) == 1

    is_real_robot = EnvUtils.is_real_robot_env(env=env) or EnvUtils.is_real_robot_gprs_env(env=env)
    if is_real_robot:
        need_pause = False
        if "env_name" not in ckpt_dict["env_metadata"]["env_kwargs"]:
            log_warning("env_name not in checkpoint...proceed with caution...")
            need_pause = True
        if ckpt_dict["env_metadata"]["env_name"] != "EnvRealPandaGPRS":
            log_warning("env name in metadata appears to be class ({}) different from EnvRealPandaGPRS".format(ckpt_dict["env_metadata"]["env_name"]))
            need_pause = True
        if need_pause:
            ans = input("continue? (y/n)")
            if ans != "y":
                exit()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    write_dataset = (args.dataset_path is not None)
    if write_dataset:
        data_writer = h5py.File(args.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0

    rollout_stats = []
    for i in tqdm(range(rollout_num_episodes)):
        try:
            ep_initial_state = initial_states[i] if initial_states is not None else None
            stats, traj = rollout(
                policy=policy, 
                env=env, 
                horizon=rollout_horizon, 
                render=args.render, 
                video_writer=video_writer, 
                video_skip=args.video_skip, 
                return_obs=(write_dataset and args.dataset_obs),
                camera_names=args.camera_names,
                real=is_real_robot,
                rate_measure=rate_measure,
                initial_state=ep_initial_state,
            )
        except KeyboardInterrupt:
            if is_real_robot:
                print("ctrl-C caught, stop execution")
                ans = input("success? (y / n)")
                rollout_stats.append((1 if ans == "y" else 0))
                continue
            else:
                sys.exit(0)
        
        if is_real_robot:
            ans = input("success? (y / n)")
            rollout_stats.append((1 if ans == "y" else 0))
            continue
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

    # Pop non-numeric per-nut fields before numeric aggregation
    nut_assembly_log = []
    for s in rollout_stats:
        if 'assembled_nut' in s:
            nut_assembly_log.append({
                'assembled_nut':    s.pop('assembled_nut'),
                'assembly_yaw_deg': s.pop('assembly_yaw_deg'),
            })

    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = { k : np.mean(rollout_stats[k]) for k in rollout_stats }
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])
    avg_rollout_stats["Time_Episode"] = np.sum(rollout_stats["time"]) / 60.
    avg_rollout_stats["Num_Episode"] = len(rollout_stats["Success_Rate"])

    # Per-nut summary (NutAssemblyDualSquare only)
    if nut_assembly_log:
        from collections import Counter
        import numpy as _np
        assembled = [e['assembled_nut'] for e in nut_assembly_log]
        counts = Counter(str(a) for a in assembled)
        total = len(nut_assembly_log)
        print("\n=== Per-Nut Assembly Summary ===")
        for label, count in sorted(counts.items()):
            yaws = [e['assembly_yaw_deg'] for e in nut_assembly_log
                    if str(e['assembled_nut']) == label and e['assembly_yaw_deg'] is not None]
            yaw_str = f", mean yaw={_np.mean(yaws):.1f}° (std {_np.std(yaws):.1f}°)" if yaws else ""
            display = "No assembly" if label == "None" else label
            print(f"  {display}: {count}/{total} rollouts ({100*count/total:.0f}%){yaw_str}")
        avg_rollout_stats["Nut_Counts"] = dict(counts)
        avg_rollout_stats["Assembly_Yaw_Mean_Deg"] = {
            label: float(_np.mean([e['assembly_yaw_deg'] for e in nut_assembly_log
                                    if str(e['assembled_nut']) == label and e['assembly_yaw_deg'] is not None]))
            for label in counts if label != "None"
        }

    print("Average Rollout Stats")
    stats_json = json.dumps(avg_rollout_stats, indent=4)
    print(stats_json)
    if args.json_path is not None:
        json_f = open(args.json_path, "w")
        json_f.write(stats_json)
        json_f.close()

    if write_video:
        video_writer.close()

    if write_dataset:
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) 
        data_writer.close()
        print("Wrote dataset trajectories to {}".format(args.dataset_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, required=True, help="path to saved checkpoint pth file")
    parser.add_argument("--n_rollouts", type=int, default=27, help="number of rollouts")
    parser.add_argument("--horizon", type=int, default=None, help="override maximum horizon of rollout")
    parser.add_argument("--env", type=str, default=None, help="override name of env from checkpoint")
    parser.add_argument("--render", action='store_true', help="on-screen rendering")
    parser.add_argument("--video_path", type=str, default=None, help="render rollouts to this video path")
    parser.add_argument("--video_skip", type=int, default=5, help="render video every n steps")
    parser.add_argument("--camera_names", type=str, nargs='+', default=None, help="cameras to use for rendering")
    parser.add_argument("--dataset_path", type=str, default=None, help="path to save rollout hdf5 dataset")
    parser.add_argument("--dataset_obs", action='store_true', help="include high-dimensional observations in output dataset")
    parser.add_argument("--seed", type=int, default=None, help="set seed for rollouts")
    parser.add_argument("--json_path", type=str, default=None, help="dump rollout stats json here")
    parser.add_argument("--error_path", type=str, default=None, help="dump error traceback text file here")
    parser.add_argument("--eval_dataset", type=str, default=None, help="validation HDF5 dataset for deterministic rollouts")
    parser.add_argument("--hz", type=int, default=None, help="measure target action computation frequency threshold")
    parser.add_argument("--dp_eval_steps", type=int, default=None, help="set inference timesteps explicitly for DP evaluation")

    args = parser.parse_args()
    res_str = None
    try:
        run_trained_agent(args)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
        if args.error_path is not None:
            f = open(args.error_path, "w")
            f.write(res_str)
            f.close()
        raise e