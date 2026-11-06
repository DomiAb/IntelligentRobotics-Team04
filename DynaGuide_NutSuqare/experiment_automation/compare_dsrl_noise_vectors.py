"""
Safely parallelized batch evaluation for DSRL checkpoints.
- Keeps DP policy and Robosuite environment persistently loaded per worker process.
- Distributes (checkpoint, run_id) evaluation tasks across a worker pool.
- Aggregates results into JSON and CSV summary files.
"""

import os
import glob
import json
import csv
import argparse
import numpy as np
import torch
import torch.multiprocessing as mp

# Set rendering backends before any submodules load
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

try:
    import mimicgen
except ImportError:
    pass

import sys
from pathlib import Path

dsrl_scripts_path = Path(__file__).resolve().parent.parent / "dsrl_scripts"
sys.path.append(str(dsrl_scripts_path))

import robomimic.utils.file_utils as FileUtils
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from dsrl.networks import SquashedGaussianActor
from dsrl.rollout import extract_state_feature, denoise_actions
from dsrl.utils import set_seed

# Global worker handles (initialized once per process)
G_ENV = None
G_ROLLOUT_POLICY = None
G_DP_ALGO = None
G_TA = None
G_TP = None
G_DA = None
G_STATE_DIM = None
G_ACT_DIM = None
G_DEVICE = None


def init_worker(dp_ckpt, env_name, ddim_steps, device_id):
    """
    Initializes the environment and frozen Diffusion Policy once per worker process.
    """
    global G_ENV, G_ROLLOUT_POLICY, G_DP_ALGO, G_TA, G_TP, G_DA, G_STATE_DIM, G_ACT_DIM, G_DEVICE

    # Assign worker to specified GPU
    if torch.cuda.is_available() and device_id is not None:
        G_DEVICE = torch.device(f"cuda:{device_id}")
        torch.cuda.set_device(G_DEVICE)
    else:
        G_DEVICE = torch.device("cpu")

    # Load frozen diffusion policy
    G_ROLLOUT_POLICY, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=dp_ckpt,
        device=G_DEVICE,
        verbose=False,
    )
    G_DP_ALGO = G_ROLLOUT_POLICY.policy

    # Configure DDIM
    with G_DP_ALGO.algo_config.values_unlocked():
        G_DP_ALGO.algo_config.ddpm.enabled = False
        G_DP_ALGO.algo_config.ddim.enabled = True
        G_DP_ALGO.algo_config.ddim.num_inference_timesteps = ddim_steps

    G_DP_ALGO.noise_scheduler = DDIMScheduler(
        num_train_timesteps=G_DP_ALGO.algo_config.ddim.num_train_timesteps,
        beta_schedule=G_DP_ALGO.algo_config.ddim.beta_schedule,
        clip_sample=G_DP_ALGO.algo_config.ddim.clip_sample,
        set_alpha_to_one=G_DP_ALGO.algo_config.ddim.set_alpha_to_one,
        steps_offset=G_DP_ALGO.algo_config.ddim.steps_offset,
        prediction_type=G_DP_ALGO.algo_config.ddim.prediction_type,
    )

    # Create worker-local environment
    G_ENV, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        env_name=env_name,
        render=False,
        render_offscreen=False,
        verbose=False,
    )

    G_TA = int(G_DP_ALGO.algo_config.horizon.action_horizon)
    G_TP = int(G_DP_ALGO.algo_config.horizon.prediction_horizon)
    G_DA = int(G_DP_ALGO.ac_dim)
    G_ACT_DIM = G_TP * G_DA

    obs0 = G_ENV.reset()
    G_ROLLOUT_POLICY.start_episode()
    s0 = extract_state_feature(G_DP_ALGO, G_ROLLOUT_POLICY, obs0)
    G_STATE_DIM = int(s0.shape[-1])


@torch.no_grad()
def eval_task_worker(task):
    """
    Evaluates one (checkpoint, run_id) instance in parallel.
    """
    global G_ENV, G_ROLLOUT_POLICY, G_DP_ALGO, G_TA, G_TP, G_DA, G_STATE_DIM, G_ACT_DIM, G_DEVICE

    ckpt_path, run_idx, seed, n_episodes, max_steps, hidden_dim = task
    ckpt_name = os.path.basename(ckpt_path)

    set_seed(seed)

    # Load actor weights
    ckpt_data = torch.load(ckpt_path, map_location=G_DEVICE)
    action_mag = float(ckpt_data.get("args", {}).get("action_magnitude", 0.5)) if isinstance(ckpt_data, dict) else 0.5
    hidden = (hidden_dim, hidden_dim, hidden_dim)

    actor = SquashedGaussianActor(
        state_dim=G_STATE_DIM,
        action_dim=G_ACT_DIM,
        hidden=hidden,
        action_magnitude=action_mag,
    ).to(G_DEVICE)

    actor_state = ckpt_data["actor"] if isinstance(ckpt_data, dict) and "actor" in ckpt_data else ckpt_data
    actor.load_state_dict(actor_state)
    actor.eval()

    successes = 0
    returns = []
    episode_steps = []

    for _ in range(n_episodes):
        obs = G_ENV.reset()
        G_ROLLOUT_POLICY.start_episode()

        ep_ret = 0.0
        steps = 0
        done = False
        success = False

        while steps < max_steps and not done and not success:
            s_feat = extract_state_feature(G_DP_ALGO, G_ROLLOUT_POLICY, obs)
            a_flat = actor.deterministic(s_feat)
            noise_w = a_flat.view(1, G_TP, G_DA)

            actions = denoise_actions(G_DP_ALGO, noise_w, s_feat)
            if hasattr(G_DP_ALGO, "normalizers") and "actions" in G_DP_ALGO.normalizers:
                actions = G_DP_ALGO.normalizers["actions"].unnormalize(actions)
            elif hasattr(G_DP_ALGO, "unnormalize_actions"):
                actions = G_DP_ALGO.unnormalize_actions(actions)

            actions_np = actions.detach().cpu().numpy()

            for i in range(G_TA):
                act = actions_np[i]
                obs, r, done, _ = G_ENV.step(act)
                ep_ret += float(r)
                steps += 1

                is_succ = G_ENV.is_success()
                success = bool(is_succ["task"] if isinstance(is_succ, dict) else is_succ)

                if done or success or steps >= max_steps:
                    break

        successes += int(success)
        returns.append(ep_ret)
        episode_steps.append(steps)

    sr = successes / float(n_episodes)
    mean_ret = float(np.mean(returns))
    mean_steps = float(np.mean(episode_steps))

    return {
        "ckpt_name": ckpt_name,
        "ckpt_path": ckpt_path,
        "run_idx": run_idx,
        "success_rate": sr,
        "mean_return": mean_ret,
        "mean_steps": mean_steps,
    }


def main(args):
    mp.set_start_method("spawn", force=True)

    # 1. Discover all checkpoints
    ckpt_pattern = os.path.join(args.ckpt_dir, "*.pth")
    ckpt_paths = sorted(glob.glob(ckpt_pattern))
    if not ckpt_paths:
        raise FileNotFoundError(f"No .pth files found in {args.ckpt_dir}")

    print(f"[INIT] Found {len(ckpt_paths)} checkpoints in {args.ckpt_dir}")
    print(f"[INIT] Spawning {args.num_workers} parallel workers...")

    # 2. Build task list
    tasks = []
    for ckpt_path in ckpt_paths:
        for r in range(args.n_runs_per_policy):
            task_seed = args.base_seed + r * 1000 + hash(ckpt_path) % 10000
            tasks.append((
                ckpt_path,
                r + 1,
                task_seed,
                args.n_episodes,
                args.max_steps,
                args.hidden,
            ))

    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device_id = 0 if num_devices > 0 else None

    # 3. Execute tasks in parallel pool
    results = []
    total_tasks = len(tasks)
    completed = 0

    with mp.Pool(
        processes=args.num_workers,
        initializer=init_worker,
        initargs=(args.dp_ckpt, args.env, args.ddim_steps, device_id),
    ) as pool:
        for res in pool.imap_unordered(eval_task_worker, tasks):
            results.append(res)
            completed += 1
            print(f"[{completed:03d}/{total_tasks:03d}] Finished {res['ckpt_name']} (Run {res['run_idx']}): SR={res['success_rate']:.2f}")

    # 4. Group and aggregate metrics
    aggregated = {}
    for r in results:
        name = r["ckpt_name"]
        if name not in aggregated:
            aggregated[name] = {
                "checkpoint_path": r["ckpt_path"],
                "runs": [],
                "srs": [],
                "returns": [],
                "steps": [],
            }
        aggregated[name]["runs"].append({
            "run": r["run_idx"],
            "success_rate": r["success_rate"],
            "mean_return": r["mean_return"],
            "mean_steps": r["mean_steps"],
        })
        aggregated[name]["srs"].append(r["success_rate"])
        aggregated[name]["returns"].append(r["mean_return"])
        aggregated[name]["steps"].append(r["mean_steps"])

    # 5. Format summary table & outputs
    print("\n" + "=" * 95)
    print(f"{'Checkpoint':<45} | {'Runs SR':<30} | {'Mean SR':<8} | {'Std':<6}")
    print("=" * 95)

    all_results = {}
    csv_rows = []

    for ckpt_name in sorted(aggregated.keys()):
        data = aggregated[ckpt_name]
        srs = data["srs"]
        mean_sr = float(np.mean(srs))
        std_sr = float(np.std(srs))

        sr_str = "[" + ", ".join([f"{v:.2f}" for v in srs]) + "]"
        print(f"{ckpt_name:<45} | {sr_str:<30} | {mean_sr:<8.3f} | {std_sr:<6.3f}")

        all_results[ckpt_name] = {
            "checkpoint_path": data["checkpoint_path"],
            "runs": sorted(data["runs"], key=lambda x: x["run"]),
            "metrics": {
                "mean_success_rate": mean_sr,
                "std_success_rate": std_sr,
                "min_success_rate": float(np.min(srs)),
                "max_success_rate": float(np.max(srs)),
                "overall_mean_return": float(np.mean(data["returns"])),
                "overall_mean_steps": float(np.mean(data["steps"])),
            },
        }

        row = {
            "checkpoint": ckpt_name,
            "mean_sr": round(mean_sr, 4),
            "std_sr": round(std_sr, 4),
            "overall_mean_return": round(float(np.mean(data["returns"])), 4),
        }
        for idx, val in enumerate(srs):
            row[f"run_{idx+1}_sr"] = val
        csv_rows.append(row)

    print("=" * 95)

    # 6. Save JSON & CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
    json_path = args.out_file if args.out_file.endswith(".json") else f"{args.out_file}.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"[SAVED JSON] {json_path}")

    csv_path = json_path.replace(".json", ".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[SAVED CSV]  {csv_path}")

    os._exit(0)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Parallel batch evaluation for DSRL.")
    p.add_argument("--dp_ckpt", type=str, default="dataset/ep_300_square_d0_dsrl.pth",
                   help="Path to pre-trained Diffusion Policy checkpoint")
    p.add_argument("--ckpt_dir", type=str,
                   default="/home/pauld/Desktop/SMLR/ss26_g2_smlr/dataset/DSRL_from_WS",
                   help="Directory containing noise actor .pth checkpoints")
    p.add_argument("--env", type=str, default=None, help="Override environment name")
    p.add_argument("--n_episodes", type=int, default=100, help="Episodes per run")
    p.add_argument("--n_runs_per_policy", type=int, default=5, help="Repetitions per checkpoint")
    p.add_argument("--max_steps", type=int, default=200, help="Max steps per episode")
    p.add_argument("--ddim_steps", type=int, default=10, help="DDIM inference steps")
    p.add_argument("--hidden", type=int, default=2048, help="Actor MLP hidden dimension")
    p.add_argument("--base_seed", type=int, default=42, help="Base seed")
    p.add_argument("--num_workers", type=int, default=4, help="Number of parallel worker processes")
    p.add_argument("--out_file", type=str, default="./dsrl_eval_summary.json",
                   help="Path to output summary file")

    args = p.parse_args()
    main(args)