"""Convert rollout-format HDF5 data into a base dataset structure.

This script flattens stacked observation frames from the rollout format and copies core
trajectory keys. It also duplicates actions into an actions_abs field for compatibility.

Mandatory flags:
    --input PATH    source rollout HDF5 file
    --output PATH   destination HDF5 file with converted structure

Example:
    python helper_scripts/convert_rollout_to_dataset.py --input rollout.hdf5 --output base.hdf5
"""

import argparse
import h5py
import numpy as np

def convert_dataset(input_path, output_path):
    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_path}")

    with h5py.File(input_path, "r") as f_in, h5py.File(output_path, "w") as f_out:
        
        # 1. Copy root attributes
        for attr_name in f_in.attrs:
            f_out.attrs[attr_name] = f_in.attrs[attr_name]
            
        # 2. Copy env_meta group
        if "env_meta" in f_in:
            f_out.copy(f_in["env_meta"], "env_meta")
        
        # 3. Create the data group and copy its attributes
        data_group_out = f_out.create_group("data")
        if "data" in f_in:
            for attr_name in f_in["data"].attrs:
                data_group_out.attrs[attr_name] = f_in["data"].attrs[attr_name]
        
        # Loop through all episodes
        for ep_key in f_in["data"].keys():
            ep_in = f_in["data"][ep_key]
            ep_out = f_out.create_group(f"data/{ep_key}")
            
            for attr_name in ep_in.attrs:
                ep_out.attrs[attr_name] = ep_in.attrs[attr_name]
            
            # Copy standard keys
            ep_out.create_dataset("dones", data=ep_in["dones"][:])
            ep_out.create_dataset("rewards", data=ep_in["rewards"][:])
            ep_out.create_dataset("states", data=ep_in["states"][:])
            
            # Copy actions and duplicate for actions_abs
            actions = ep_in["actions"][:]
            ep_out.create_dataset("actions", data=actions)
            ep_out.create_dataset("actions_abs", data=actions)
            
            # 4. Flatten obs and next_obs
            for obs_group_name in ["obs", "next_obs"]:
                if obs_group_name in ep_in:
                    obs_out = ep_out.create_group(obs_group_name)
                    obs_in = ep_in[obs_group_name]
                    
                    for k in obs_in.keys():
                        # Exclude 'timesteps' or 'proprio' if they are not in the base dataset info
                        if k in ["timesteps", "proprio"]:
                            continue
                            
                        # Extract the latest frame from the stacked dimension (index 1)
                        # Changes shape from (N, 2, ...) to (N, ...)
                        data_reduced = obs_in[k][:, 1, ...] 
                        obs_out.create_dataset(k, data=data_reduced)

    print("Conversion completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert rollout hdf5 dataset structure to match base dataset structure.")
    parser.add_argument("--input", type=str, required=True, help="Path to the original rollout hdf5 file.")
    parser.add_argument("--output", type=str, required=True, help="Path where the converted hdf5 file will be saved.")
    
    args = parser.parse_args()
    convert_dataset(args.input, args.output)
