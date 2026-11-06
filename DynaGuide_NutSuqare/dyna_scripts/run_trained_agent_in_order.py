"""
The main script for evaluating a policy in an environment with deterministic initial states.
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

# Ensure local robosuite/robomimic packages are found before any installed versions
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _pkg in ("robosuite", "robomimic"):
    _pkg_path = os.path.join(_REPO_ROOT, _pkg)
    if _pkg_path not in sys.path:
        sys.path.insert(0, _pkg_path)
del _REPO_ROOT, _pkg, _pkg_path
from collections import Counter
from copy import deepcopy
from tqdm import tqdm

import torch

import cv2

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


CLUSTER_ORDER = ["Right (45-135°)", "Back (135-225°)", "Left (225-315°)", "Front (315-45°)"]


def get_orientation_cluster(yaw_deg):
    """Map a yaw angle (degrees) to one of 4 insertion-direction clusters (robot-relative)."""
    y = yaw_deg % 360
    if 45 <= y < 135:
        return "Right (45-135°)"
    elif 135 <= y < 225:
        return "Back (135-225°)"
    elif 225 <= y < 315:
        return "Left (225-315°)"
    else:
        return "Front (315-45°)"


class _Tee:
    """Write to multiple streams simultaneously (console + log file)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()

def _unwrap_env(env):
    """Peel robomimic wrapper layers to reach the underlying robosuite environment."""
    e = env
    while hasattr(e, 'env'):
        e = e.env
    return e


def _output_stem(args):
    """Return path stem (no extension) to derive log/args filenames from, or None."""
    for p in [args.json_path, args.video_path, args.dataset_path]:
        if p is not None:
            return os.path.splitext(p)[0]
    return None


def rollout(
    policy,
    env,
    horizon,
    render=False,
    video_writer=None,
    video_skip=5,
    return_obs=False,
    camera_names=None,
    real=False,
    rate_measure=None,
    initial_state=None,
):
    """
    Helper function to carry out rollouts.
    """
    rollout_timestamp = time.time()
    assert isinstance(env, EnvBase) or isinstance(env, EnvWrapper)
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    policy.start_episode()
    state_dict = dict()

    if initial_state is not None and not real:
        state_dict = {"states": np.array(initial_state)}
        obs = env.reset_to(state_dict)
    else:
        obs = env.reset()
        if real:
            input("ready for next eval? hit enter to continue")
        else:
            state_dict = env.get_state()
            obs = env.reset_to(state_dict)

    video_count = 0
    total_reward = 0.0
    got_exception = False
    success = env.is_success()["task"]
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)

    if return_obs:
        traj.update(dict(obs=[], next_obs=[]))

    try:
        for step_i in range(horizon):
            for k in obs:
                if len(obs[k].shape) == 1:
                    obs[k] = obs[k][..., None]

            t1 = time.time()
            act = policy(ob=obs)
            t2 = time.time()

            if real and (not env.base_env.controller_type == "JOINT_IMPEDANCE") and (
                policy.policy.global_config.algo_name != "diffusion_policy"
            ):
                act = np.clip(act, -1.0, 1.0)

            if rate_measure is not None:
                rate_measure.measure()
                print("time: {}s".format(t2 - t1))
                r = 0.0
                done = False
                next_obs = obs
            else:
                next_obs, r, done, _ = env.step(act)

            total_reward += r
            success = env.is_success()["task"]

            if render:
                env.render(mode="human", camera_name=camera_names[0])

            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(
                            env.render(
                                mode="rgb_array",
                                height=512,
                                width=512,
                                camera_name=cam_name,
                            )
                        )
                    video_img = np.concatenate(video_img, axis=1).copy()
                    text = "SUCCESS" if success else "FAIL"
                    color = (0, 255, 0) if success else (255, 0, 0)
                    cv2.putText(video_img, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
                    if success:
                        rs_env = _unwrap_env(env)
                        overlay_y = 80
                        if hasattr(rs_env, 'which_nut_assembled'):
                            nut = rs_env.which_nut_assembled()
                            if nut is not None:
                                cv2.putText(video_img, f"Nut: {nut}", (10, overlay_y),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
                                overlay_y += 30
                        if hasattr(rs_env, 'assembled_nut_yaw'):
                            yaw = rs_env.assembled_nut_yaw()
                            if yaw is not None:
                                cluster = get_orientation_cluster(yaw)
                                cv2.putText(video_img, f"Yaw: {yaw:.1f}deg", (10, overlay_y),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
                                cv2.putText(video_img, f"Dir: {cluster}", (10, overlay_y + 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
                    video_writer.append_data(video_img)
                video_count += 1

            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)

            if not real:
                if "states" in state_dict:
                    traj["states"].append(state_dict["states"])
                else:
                    traj["states"].append([])

            if return_obs:
                traj["obs"].append(ObsUtils.unprocess_obs_dict(obs))
                traj["next_obs"].append(ObsUtils.unprocess_obs_dict(next_obs))

            if done or success:
                break

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

    # Capture insertion yaw / nut identity (non-numeric, popped before aggregation)
    rs_env = _unwrap_env(env)
    if hasattr(rs_env, 'assembled_nut_yaw'):
        stats['assembly_yaw_deg'] = rs_env.assembled_nut_yaw()
    if hasattr(rs_env, 'which_nut_assembled'):
        stats['assembled_nut'] = rs_env.which_nut_assembled()

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
            try:
                traj[k] = np.array(traj[k])
            except ValueError:
                traj[k] = np.array(traj[k], dtype=object)

    return stats, traj


def run_trained_agent(args):
    write_video = args.video_path is not None
    assert not (args.render and write_video)

    # ── Derive output stem and start logging ──────────────────────────────────
    stem = _output_stem(args)
    _log_file = None
    _orig_stdout = sys.stdout
    if stem is not None:
        out_dir = os.path.dirname(stem)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        log_path = stem + "_log.txt"
        args_path = stem + "_args.json"
        with open(args_path, "w") as f:
            json.dump({
                "command": " ".join(sys.argv),
                "args": vars(args),
            }, f, indent=4)
        _log_file = open(log_path, "w")
        sys.stdout = _Tee(sys.__stdout__, _log_file)
        print(f"[Log] Console output → {log_path}")
        print(f"[Log] Arguments      → {args_path}")

    try:
        _run_trained_agent_inner(args, write_video)
    finally:
        sys.stdout = _orig_stdout
        if _log_file is not None:
            _log_file.close()


def _run_trained_agent_inner(args, write_video):
    rate_measure = None
    if args.hz is not None:
        import RobotTeleop
        from RobotTeleop.utils import RateMeasure
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
        ckpt_dict["config"] = tmp_config.dump()

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_dict=ckpt_dict, device=device, verbose=True
    )

    rollout_num_episodes = args.n_rollouts
    rollout_horizon = args.horizon
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    if rollout_horizon is None:
        rollout_horizon = config.experiment.rollout.horizon

    if (algo_name == "diffusion_policy") and EnvUtils.is_real_robot_gprs_env(
        env_meta=ckpt_dict["env_metadata"]
    ):
        ckpt_dict["env_metadata"]["env_kwargs"]["absolute_actions"] = True

    initial_states = None

    if args.eval_dataset is not None:
        if not os.path.isfile(args.eval_dataset):
            raise FileNotFoundError(f"Evaluation dataset missing: {args.eval_dataset}")
        print(f"[Deterministic Eval] Loading rollout initialization states from: {args.eval_dataset}")
        with h5py.File(args.eval_dataset, "r") as f_eval:
            demos = sorted(list(f_eval["data"].keys()), key=lambda x: int(x.split("_")[-1]))
            initial_states = [f_eval[f"data/{d}/states"][0] for d in demos]
        rollout_num_episodes = min(rollout_num_episodes, len(initial_states))
        print(f"[Deterministic Eval] Adjusted evaluation rollout iterations to: {rollout_num_episodes}")

    if args.initial_states_json is not None:
        if not os.path.isfile(args.initial_states_json):
            raise FileNotFoundError(f"Initial states JSON missing: {args.initial_states_json}")
        print(f"[Deterministic Eval] Loading rollout initialization states from JSON: {args.initial_states_json}")
        with open(args.initial_states_json, "r") as f_json:
            payload = json.load(f_json)
        if "robot_states" in payload:
            initial_states = payload["robot_states"]
        elif "states" in payload:
            initial_states = payload["states"]
        else:
            raise KeyError("Expected key 'robot_states' or 'states' in initial states JSON")
        rollout_num_episodes = min(rollout_num_episodes, len(initial_states))
        print(f"[Deterministic Eval] Adjusted evaluation rollout iterations to: {rollout_num_episodes}")

    # Tight-fit experiment: switch env and inject hole dimensions into metadata kwargs.
    if args.hole_x is not None or args.hole_y is not None:
        if args.env is None:
            args.env = "NutAssemblyTightSquare"
        hx = args.hole_x if args.hole_x is not None else 0.018
        hy = args.hole_y if args.hole_y is not None else 0.018
        ckpt_dict["env_metadata"]["env_kwargs"]["hole_x"] = hx
        ckpt_dict["env_metadata"]["env_kwargs"]["hole_y"] = hy

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
            log_warning(
                "env name in metadata appears to be class ({}) different from EnvRealPandaGPRS".format(
                    ckpt_dict["env_metadata"]["env_name"]
                )
            )
            need_pause = True
        if need_pause:
            ans = input("continue? (y/n)")
            if ans != "y":
                exit()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    if write_video:
        video_dir = os.path.dirname(args.video_path)
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)
        video_writer = imageio.get_writer(args.video_path, fps=20)
    else:
        video_writer = None

    write_dataset = args.dataset_path is not None
    if write_dataset:
        dataset_dir = os.path.dirname(args.dataset_path)
        if dataset_dir:
            os.makedirs(dataset_dir, exist_ok=True)
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
                print("ctrl-C catched, stop execution")
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

    # Pop non-numeric yaw/nut fields before numeric aggregation
    yaw_log = []
    nut_assembly_log = []
    for s in rollout_stats:
        entry = {}
        if 'assembly_yaw_deg' in s:
            entry['assembly_yaw_deg'] = s.pop('assembly_yaw_deg')
            yaw_log.append(entry['assembly_yaw_deg'])
        if 'assembled_nut' in s:
            entry['assembled_nut'] = s.pop('assembled_nut')
        if entry:
            nut_assembly_log.append(entry)

    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = {k: np.mean(rollout_stats[k]) for k in rollout_stats}
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])
    avg_rollout_stats["Time_Episode"] = np.sum(rollout_stats["time"]) / 60.0
    avg_rollout_stats["Num_Episode"] = len(rollout_stats["Success_Rate"])

    # Insertion angle and orientation cluster summary
    valid_yaws = [y for y in yaw_log if y is not None]
    if valid_yaws:
        clusters = [get_orientation_cluster(y) for y in valid_yaws]
        cluster_counts = Counter(clusters)
        total_success = len(valid_yaws)
        print("\n=== Insertion Angle Summary ===")
        print(f"  All yaw angles (deg): {[round(y, 1) for y in valid_yaws]}")
        print(f"  Mean: {np.mean(valid_yaws):.1f}°  Std: {np.std(valid_yaws):.1f}°  "
              f"Min: {np.min(valid_yaws):.1f}°  Max: {np.max(valid_yaws):.1f}°")
        print("\n  Orientation Clusters:")
        for cluster in CLUSTER_ORDER:
            count = cluster_counts.get(cluster, 0)
            bar = "#" * count
            print(f"    {cluster:<22}: {count:3d}/{total_success} ({100*count/total_success:.0f}%)  {bar}")
        avg_rollout_stats["All_Insertion_Yaws_Deg"] = valid_yaws
        avg_rollout_stats["Insertion_Yaw_Mean_Deg"] = float(np.mean(valid_yaws))
        avg_rollout_stats["Insertion_Yaw_Std_Deg"] = float(np.std(valid_yaws))
        avg_rollout_stats["Orientation_Cluster_Counts"] = dict(cluster_counts)

    # Per-nut summary (NutAssemblyDualSquare only)
    dual_log = [e for e in nut_assembly_log if 'assembled_nut' in e]
    if dual_log:
        assembled = [e['assembled_nut'] for e in dual_log]
        counts = Counter(str(a) for a in assembled)
        total = len(dual_log)
        print("\n=== Per-Nut Assembly Summary ===")
        for label, count in sorted(counts.items()):
            yaws = [e['assembly_yaw_deg'] for e in dual_log
                    if str(e['assembled_nut']) == label and e.get('assembly_yaw_deg') is not None]
            display = "No assembly" if label == "None" else label
            yaw_str = f"  mean yaw={np.mean(yaws):.1f}° (std {np.std(yaws):.1f}°)" if yaws else ""
            print(f"  {display}: {count}/{total} rollouts ({100*count/total:.0f}%){yaw_str}")
            if yaws:
                nut_clusters = Counter(get_orientation_cluster(y) for y in yaws)
                for cl in CLUSTER_ORDER:
                    c = nut_clusters.get(cl, 0)
                    bar = "#" * c
                    print(f"      {cl:<22}: {c:3d}/{len(yaws)} ({100*c/len(yaws):.0f}%)  {bar}")
        avg_rollout_stats["Nut_Counts"] = dict(counts)
        avg_rollout_stats["Assembly_Yaw_Mean_Deg"] = {
            label: float(np.mean([e['assembly_yaw_deg'] for e in dual_log
                                   if str(e['assembled_nut']) == label and e.get('assembly_yaw_deg') is not None]))
            for label in counts if label != "None"
            if any(e.get('assembly_yaw_deg') is not None for e in dual_log if str(e['assembled_nut']) == label)
        }

    print("\nAverage Rollout Stats")
    stats_json = json.dumps(avg_rollout_stats, indent=4)
    print(stats_json)

    if args.json_path is not None:
        json_dir = os.path.dirname(args.json_path)
        if json_dir:
            os.makedirs(json_dir, exist_ok=True)
        with open(args.json_path, "w") as json_f:
            json_f.write(stats_json)

    if write_video:
        video_writer.close()

    if write_dataset:
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4)
        data_writer.close()
        print("Wrote dataset trajectories to {}".format(args.dataset_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--agent", type=str, default=None, help="path to saved checkpoint pth file")
    parser.add_argument("--n_rollouts", type=int, default=27, help="number of rollouts")
    parser.add_argument("--horizon", type=int, default=None, help="override maximum horizon of rollout")
    parser.add_argument("--env", type=str, default=None, help="override env name from checkpoint")
    parser.add_argument("--render", action="store_true", help="on-screen rendering")
    parser.add_argument("--video_path", type=str, default=None, help="render rollouts to this video file path")
    parser.add_argument("--video_skip", type=int, default=5, help="render frames to video every n steps")
    parser.add_argument("--camera_names", type=str, nargs="+", default=None, help="camera names to use for rendering")
    parser.add_argument("--dataset_path", type=str, default=None, help="path to save rollout hdf5 dataset")
    parser.add_argument("--dataset_obs", action="store_true", help="include observations in output hdf5")
    parser.add_argument("--seed", type=int, default=None, help="set seed for rollouts")
    parser.add_argument("--json_path", type=str, default=None, help="dump rollout stats json here")
    parser.add_argument("--error_path", type=str, default=None, help="dump error traceback text file here")
    parser.add_argument("--eval_dataset", type=str, default=None, help="validation HDF5 dataset for deterministic rollouts")
    parser.add_argument("--initial_states_json", type=str, default=None, help="json file containing ordered rollout initialization states")
    parser.add_argument("--hz", type=int, default=None, help="measure target action computation frequency threshold")
    parser.add_argument("--dp_eval_steps", type=int, default=None, help="set inference timesteps explicitly for DP evaluation")
    parser.add_argument(
        "--hole_x", type=float, default=None,
        help="inner hole half-extent in x (metres) for NutAssemblyTightSquare; "
             "omit to use the standard nut geometry"
    )
    parser.add_argument(
        "--hole_y", type=float, default=None,
        help="inner hole half-extent in y (metres) for NutAssemblyTightSquare; "
             "omit to use the standard nut geometry"
    )

    args = parser.parse_args()
    res_str = None
    try:
        run_trained_agent(args)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
        if args.error_path is not None:
            error_dir = os.path.dirname(args.error_path)
            if error_dir:
                os.makedirs(error_dir, exist_ok=True)
            with open(args.error_path, "w") as f:
                f.write(res_str)
        raise e
