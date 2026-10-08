#!/usr/bin/env python3
"""
Inject a unified 9D proprioception vector into an existing HDF5 dataset.

This script opens an HDF5 file in read/write mode and, for each demonstration
under the `data/` group, concatenates the following low-dimensional observation
streams along the last dimension:

- `robot0_eef_pos`      -> 3D end-effector position
- `robot0_eef_quat`     -> 4D end-effector orientation quaternion
- `robot0_gripper_qpos` -> 2D gripper joint positions

The resulting tensor has shape `(T, 9)` and is written to `obs/proprio`.
If `obs/proprio` already exists, it is overwritten.

Requirements:
- The HDF5 file must contain a top-level `data/` group.
- Each demo must contain all required keys inside `data/<demo>/obs/`.

Usage:
    python add_proprio_to_dataset.py --input dataset/square_splits/failed_policy_rollouts.hdf5
    python add_proprio_to_dataset.py --input dataset/square_splits/square_valid.hdf5
"""

import argparse
import os
import h5py
import numpy as np
from tqdm import tqdm

LOWDIM_KEYS = [
    "robot0_eef_pos",       # (T, 3)
    "robot0_eef_quat",      # (T, 4)
    "robot0_gripper_qpos",  # (T, 2)
]

def build_proprio_vector(obs_grp):
    """Concatenate LOWDIM_KEYS from an obs group into the unified (T, 9) proprio vector.

    This is the single source of truth for the DynaGuide proprio layout; other scripts
    (e.g. create_dynamics_dataset.py) should import this instead of re-implementing it.
    """
    missing_keys = [k for k in LOWDIM_KEYS if k not in obs_grp]
    if missing_keys:
        raise KeyError(f"Missing obs key(s) needed for proprio: {missing_keys}")
    parts = [obs_grp[k][...].astype(np.float32) for k in LOWDIM_KEYS]
    return np.concatenate(parts, axis=-1).astype(np.float32)  # Shape: (T, 9)

def add_proprio_vector(hdf5_path):
    if not os.path.isfile(hdf5_path):
        print(f"Error: File not found at {hdf5_path}")
        return

    print(f"Processing dataset: {hdf5_path}")
    
    # Open file in read/write append mode
    with h5py.File(hdf5_path, "r+") as f:
        if "data" not in f:
            print("Error: Invalid HDF5 format, missing 'data' group.")
            return
            
        demo_keys = list(f["data"].keys())
        
        for demo_key in tqdm(demo_keys, desc="Cleaning actions & injecting proprio"):
            demo_grp = f["data"][demo_key]
            
            # 1. clamp actions to [-1.0, 1.0] - eliminates floating-point precision overshoots
            if "actions" in demo_grp:
                actions = demo_grp["actions"][...]
                clipped_actions = np.clip(actions, -1.0, 1.0)
                del demo_grp["actions"]
                demo_grp.create_dataset("actions", data=clipped_actions)
            
            # 2. Inject 9D proprio vector
            obs_grp = demo_grp["obs"]
            
            # 3. strip frame-stackinf histroy if present (e.g. when hdf5 was created from rollouts)
            for k in list(obs_grp.keys()):
                data = obs_grp[k][...]
                
                # Squeeze image history stack: (T, 2, H, W, C) -> (T, H, W, C)
                if "image" in k and data.ndim == 5:
                    unstacked = data[:, -1, ...]
                    del obs_grp[k]
                    obs_grp.create_dataset(k, data=unstacked, compression="gzip", compression_opts=4, shuffle=True)
                
                # Squeeze tracking history stack: (T, 2, Dim) -> (T, Dim)
                elif k in LOWDIM_KEYS and data.ndim == 3:
                    unstacked = data[:, -1, ...]
                    del obs_grp[k]
                    obs_grp.create_dataset(k, data=unstacked, compression="gzip", compression_opts=4, shuffle=True)


            # Verify all necessary tracking keys exist
            missing_keys = [k for k in LOWDIM_KEYS if k not in obs_grp]
            if missing_keys:
                print(f"\nWarning: Skipping {demo_key}, missing keys: {missing_keys}")
                continue
                
            # Extract and concatenate to form the 9D vector
            proprio_vector = build_proprio_vector(obs_grp)
            
            # Delete existing dataset if it needs to be overwritten
            if "proprio" in obs_grp:
                del obs_grp["proprio"]
                
            # Write the unified feature layout
            obs_grp.create_dataset(
                "proprio", 
                data=proprio_vector, 
                compression="gzip", 
                compression_opts=4, 
                shuffle=True
            )

    print("Proprio vector injection & dataset cleanup complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject 9D proprio vector into an existing HDF5 file.")
    parser.add_argument("--input", required=True, help="Path to the HDF5 target file")
    args = parser.parse_args()
    add_proprio_vector(args.input)