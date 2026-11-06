#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import subprocess
from datetime import datetime
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

DP_CKPT = "dataset/ep_300_square_d0_dsrl.pth"
ACTOR_CKPT = "dataset/DSRL_retraiend_on_Desktop/noise_actor_ep000054_sr0.815.pth"
VIDEO_BASE = "./dsrl_tight_nut"
N_EPISODES = 100
MAX_STEPS = 200
N_RUNS = 5

#Don't forget to place the correct env down below!


def run_single_eval(run_id):
    # Generate timestamp for the individual run directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"run_{run_id}_{timestamp}"
    video_dir = os.path.join(VIDEO_BASE, folder_name)
    os.makedirs(video_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "dsrl_scripts/dsrl_test.py",
        "--dp_ckpt", DP_CKPT,
        "--noise_actor_ckpt", ACTOR_CKPT,
        "--n_episodes", str(N_EPISODES),
        "--max_steps", str(MAX_STEPS),
        "--video_path", video_dir,
        "--env", "NutAssemblyRedSquare"
    ]

    print(f"[RUN {run_id}] Started -> Target Folder: {folder_name}")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=dict(os.environ, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl"),
    )

    # Extract success rate and average return from output
    sr_match = re.search(r"Success rate:\s*([0-9\.]+)", result.stdout)
    ret_match = re.search(r"Avg return:\s*([0-9\.\-]+)", result.stdout)

    sr = float(sr_match.group(1)) if sr_match else None
    avg_ret = float(ret_match.group(1)) if ret_match else None

    if sr is not None:
        print(f"[RUN {run_id}] Finished ({folder_name}) -> SR: {sr * 100:.1f}% | Avg Return: {avg_ret:.2f}")
    else:
        print(f"\n{'=' * 25} [RUN {run_id}] ERROR LOG ({folder_name}) {'=' * 25}")
        print(result.stdout)
        print("=" * 70 + "\n")

    return run_id, folder_name, sr, avg_ret, result.stdout


def main():
    os.makedirs(VIDEO_BASE, exist_ok=True)
    print(f"[INIT] Launching {N_RUNS} parallel runs on GPU...")

    results = []
    with ProcessPoolExecutor(max_workers=N_RUNS) as executor:
        futures = [executor.submit(run_single_eval, i + 1) for i in range(N_RUNS)]
        for f in as_completed(futures):
            results.append(f.result())

    results.sort(key=lambda x: x[0])
    valid_srs = [r[2] for r in results if r[2] is not None]
    valid_rets = [r[3] for r in results if r[3] is not None]

    print("\n" + "=" * 70)
    print(f"{'Run':<6} | {'Subfolder':<30} | {'Success Rate':<14} | {'Avg Return':<12}")
    print("=" * 70)
    for run_id, folder_name, sr, avg_ret, _ in results:
        sr_str = f"{sr * 100:.1f}%" if sr is not None else "ERROR"
        ret_str = f"{avg_ret:.2f}" if avg_ret is not None else "ERROR"
        print(f"Run {run_id:<2} | {folder_name:<30} | {sr_str:<14} | {ret_str:<12}")

    if valid_srs:
        mean_sr = float(np.mean(valid_srs))
        std_sr = float(np.std(valid_srs))
        mean_ret = float(np.mean(valid_rets))
        print("-" * 70)
        print(f"Overall Mean SR:     {mean_sr * 100:.2f}% ± {std_sr * 100:.2f}%")
        print(f"Overall Mean Return: {mean_ret:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()