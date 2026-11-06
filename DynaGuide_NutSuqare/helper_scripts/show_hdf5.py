"""Inspect a Robomimic-style HDF5 dataset and print key statistics.

The script prints dataset size, demo count, action/state shapes, and observation keys.

Usage:
    python helper_scripts/show_hdf5.py path/to/dataset.hdf5
"""

import h5py
import numpy as np
import argparse


def get_trajectory_lengths(data, demos):
    return [data[d]["actions"].shape[0] for d in demos]


def explore(file_path):
    with h5py.File(file_path, "r") as f:
        data = f["data"]
        demos = list(data.keys())

        print("\n==============================")
        print("ROBOMIMIC HDF5 INSPECTOR")
        print("==============================\n")

        # =========================
        # 1. Dataset Overview
        # =========================
        print("Num demos:", len(demos))

        lengths = get_trajectory_lengths(data, demos)

        print("Min traj length:", min(lengths))
        print("Max traj length:", max(lengths))
        print("Avg traj length:", int(np.mean(lengths)))

        # =========================
        # 2. Keys (like columns)
        # =========================
        demo0 = data[demos[0]]

        print("\nTop-level keys:")
        print(list(demo0.keys()))

        print("\nObs keys:")
        print(list(demo0["obs"].keys()))

        # =========================
        # 3. Sample viewer
        # =========================
        demo = data[demos[0]]

        print("\n================ SAMPLE ================\n")
        print("Demo:", demos[0])
        print("Length:", demo["actions"].shape[0])

        # Actions
        actions = demo["actions"][:]
        print("\nActions shape:", actions.shape)
        print("Actions head:\n", actions[:5])

        # Rewards
        if "rewards" in demo:
            rewards = demo["rewards"][:]
            print("\nRewards head:", rewards[:10])
        else:
            print("\nRewards: Not found in dataset")

        # States
        states = demo["states"][:]
        print("\nStates shape:", states.shape)

        # Obs summary (clean)
        print("\nOBS SUMMARY:")
        obs = demo["obs"]

        for k in obs.keys():
            arr = obs[k]
            print(f"- {k:30} shape={arr.shape}, dtype={arr.dtype}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()

    explore(args.path)
