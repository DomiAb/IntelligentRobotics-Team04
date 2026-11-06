"""
Use this script to extract the first n samples from a dataset and save them as a new .HDF5.
Usage:
python extract_hdf5.py --input PATH --output PATH --n N
"""

import h5py
import argparse

def create_subset(input_path, output_path, num_demos=20):
    with h5py.File(input_path, "r") as f_in:
        # Robomimic keeps all trajectories under the "data" group
        demo_keys = sorted(list(f_in["data"].keys()), key=lambda x: int(x.split("_")[1]))
        subset_keys = demo_keys[:num_demos]
        
        print(f"Copying first {len(subset_keys)} demos from {input_path} to {output_path}...")
        
        with h5py.File(output_path, "w") as f_out:
            # Copy global root attributes (env_meta, etc.)
            for attr_name in f_in.attrs:
                f_out.attrs[attr_name] = f_in.attrs[attr_name]
                
            data_out = f_out.create_group("data")
            total_samples = 0
            
            for key in subset_keys:
                # Copy the entire episode group structure directly
                f_in.copy(f"data/{key}", data_out, name=key)
                total_samples += f_in[f"data/{key}"].attrs["num_samples"]
            
            # Update global dataset tracking attribute flags
            data_out.attrs["total"] = total_samples
            if "env_args" in f_in["data"].attrs:
                data_out.attrs["env_args"] = f_in["data"].attrs["env_args"]
                
    print("Subset creation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to original HDF5 file")
    parser.add_argument("--output", type=str, required=True, help="Path to save subset HDF5 file")
    parser.add_argument("--n", type=int, default=20, help="Number of episodes to copy")
    args = parser.parse_args()
    
    create_subset(args.input, args.output, args.n)