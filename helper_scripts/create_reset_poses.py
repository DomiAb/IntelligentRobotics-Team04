#!/usr/bin/env python3
"""
Extract initial environment states from a robomimic-style HDF5 file 
to create a deterministic reset poses configuration for DynaGuide evaluation.

Description:
    When running benchmarks across different policy iterations, evaluating on 
    randomly randomized simulator layouts introduces significant statistical noise. 
    This script reads a robomimic-style HDF5 dataset split, extracts the complete 
    low-dimensional simulator configuration state vector at the exact initial 
    timestep (t=0) for each recorded trajectory, and serializes them into a 
    standardized JSON sequence file. 
    
    Passing this output file to DynaGuide's evaluation loop guarantees that 
    the baseline policy and guided rollouts are tested against identical object 
    poses and mechanical configurations.

Usage:
    python extract_reset_poses.py \
        --input dataset/square_splits/square_valid.hdf5 \
        --output dataset/square_splits/square_initial_states.json
"""

import argparse
import os
import h5py
import json

def main():
    parser = argparse.ArgumentParser(
        description="Extract initial simulator states (t=0) from HDF5 datasets into a reset poses JSON file."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input robomimic-style HDF5 dataset file (e.g. square_valid.hdf5)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Target destination path for the generated reset poses JSON file"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Source HDF5 file not found: {args.input}")

    # Ensure parent output directory exists
    parent_dir = os.path.dirname(os.path.abspath(args.output))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    initial_states = []

    with h5py.File(args.input, "r") as f:
        if "data" not in f:
            raise KeyError("Provided HDF5 file structure is invalid: missing top-level 'data' group.")
            
        demo_names = list(f["data"].keys())
        
        # Sort demo keys numerically (demo_0, demo_1, ...) instead of lexically
        try:
            sorted_demos = sorted(demo_names, key=lambda x: int(x.split("_")[-1]))
        except Exception:
            sorted_demos = sorted(demo_names)

        for demo_name in sorted_demos:
            demo_grp = f["data"][demo_name]
            if "states" not in demo_grp:
                print(f"Warning: Skipping {demo_name}, missing low-dimensional 'states' matrix.")
                continue
                
            # Extract the first row state vector (t=0) and map to basic float array list
            first_state = demo_grp["states"][0].tolist()
            initial_states.append(first_state)

    # adapt list to expected dictionary schema
    with open(args.output, "w") as jf:
        json.dump({"robot_states": initial_states}, jf, indent=4)

    print(f"Successfully extracted {len(initial_states)} initial states to: {args.output}")

if __name__ == "__main__":
    main()