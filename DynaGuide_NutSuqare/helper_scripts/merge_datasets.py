"""Merge a base Robomimic HDF5 dataset with one or more rollout datasets.

This script preserves observation keys from the base dataset, appends rollout episodes,
and extends the train mask to include newly added rollouts.

Mandatory flags:
    --base PATH      path to the base dataset
    --rollout PATH [PATH ...]   one or more rollout dataset paths
    --output PATH    destination path for the merged HDF5 dataset

Optional flags:
    --shuffle        shuffle rollout episodes before appending

Example:
    python helper_scripts/merge_datasets.py --base base.hdf5 --rollout rollout1.hdf5 rollout2.hdf5 --output merged.hdf5
"""

import argparse
import h5py
import random

def merge_datasets(base_path, rollout_paths, output_path, shuffle=False):
    print(f"Base dataset: {base_path}")
    print(f"Rollout datasets: {rollout_paths}")
    print(f"Output dataset: {output_path}")
    print(f"Shuffle rollouts: {shuffle}")

    with h5py.File(base_path, "r") as f_base, \
         h5py.File(output_path, "w") as f_out:
        
        # 1. Copy root attributes and env_meta from base
        for attr_name in f_base.attrs:
            f_out.attrs[attr_name] = f_base.attrs[attr_name]
            
        if "env_meta" in f_base:
            f_out.copy(f_base["env_meta"], "env_meta")
            
        # Create root data group and copy attributes
        data_out = f_out.create_group("data")
        if "data" in f_base:
            for attr_name in f_base["data"].attrs:
                data_out.attrs[attr_name] = f_base["data"].attrs[attr_name]

        # 2. Get list of allowed observation keys from the base dataset
        allowed_obs_keys = list(f_base["data/demo_0/obs"].keys())
        print(f"Allowed observation keys detected: {allowed_obs_keys}")

        # 3. Copy all episodes from the base dataset
        print("Copying base dataset episodes...")
        base_episodes = list(f_base["data"].keys())
        base_episodes.sort(key=lambda x: int(x.split("_")[1]))
        
        for ep_key in base_episodes:
            f_out.copy(f_base[f"data/{ep_key}"], f"data/{ep_key}")
            
        total_base_eps = len(base_episodes)
        print(f"Copied {total_base_eps} baseline episodes.")

        # 4. Gather and sort all episodes across all rollout files
        all_rollout_episodes = [] # List of tuples: (hdf5_file_handle, original_episode_key)
        
        opened_files = []
        try:
            for r_path in rollout_paths:
                f_roll = h5py.File(r_path, "r")
                opened_files.append(f_roll)
                
                rollout_episodes = list(f_roll["data"].keys())
                rollout_episodes.sort(key=lambda x: int(x.split("_")[1]))
                
                for ep_key in rollout_episodes:
                    all_rollout_episodes.append((f_roll, ep_key))
            
            if shuffle:
                print("Shuffling all rollout episodes before appending...")
                random.seed(42)
                random.shuffle(all_rollout_episodes)
            
            new_rollout_keys = []
            for i, (f_roll, ep_key) in enumerate(all_rollout_episodes):
                new_ep_key = f"demo_{total_base_eps + i}"
                ep_in = f_roll[f"data/{ep_key}"]
                ep_out = f_out.create_group(f"data/{new_ep_key}")
                
                # Copy episode attributes
                for attr_name in ep_in.attrs:
                    ep_out.attrs[attr_name] = ep_in.attrs[attr_name]
                
                # Copy root keys for this episode
                ep_out.create_dataset("actions", data=ep_in["actions"][:])
                ep_out.create_dataset("actions_abs", data=ep_in["actions_abs"][:])
                ep_out.create_dataset("dones", data=ep_in["dones"][:])
                ep_out.create_dataset("rewards", data=ep_in["rewards"][:])
                ep_out.create_dataset("states", data=ep_in["states"][:])
                
                # Copy obs and next_obs with key filtering
                for obs_group in ["obs", "next_obs"]:
                    if obs_group in ep_in:
                        obs_out = ep_out.create_group(obs_group)
                        for k in ep_in[obs_group].keys():
                            if k in allowed_obs_keys:
                                obs_out.create_dataset(k, data=ep_in[f"{obs_group}/{k}"][:])
                
                new_rollout_keys.append(new_ep_key)
                
            print(f"Appended {len(all_rollout_episodes)} total rollout episodes from {len(rollout_paths)} files.")

            # 5. Handle Filter Keys (mask group)
            if "mask" in f_base:
                mask_out = f_out.create_group("mask")
                for mask_key in f_base["mask"].keys():
                    existing_keys = [k.decode('utf-8') if isinstance(k, bytes) else k for k in f_base[f"mask/{mask_key}"][:]]
                    
                    if mask_key == "train":
                        existing_keys.extend(new_rollout_keys)
                        print(f"Added new rollouts to '{mask_key}' split. Total split size: {len(existing_keys)}")
                    
                    ascii_keys = [k.encode("ascii", "ignore") for k in existing_keys]
                    mask_out.create_dataset(mask_key, data=ascii_keys)
                    
        finally:
            for f in opened_files:
                f.close()

    print("Merge completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge a base Robomimic HDF5 dataset with multiple rollout datasets.")
    parser.add_argument("--base", type=str, required=True, help="Path to the original base dataset.")
    parser.add_argument("--rollout", type=str, nargs="+", required=True, help="Paths to one or more converted rollout hdf5 files.")
    parser.add_argument("--output", type=str, required=True, help="Path where the merged dataset will be saved.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle rollout episodes before appending.")
    
    args = parser.parse_args()
    merge_datasets(args.base, args.rollout, args.output, args.shuffle)
