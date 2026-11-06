"""Filter unwanted demo indices out of a guidance HDF5 dataset.

The script renumbers remaining demos sequentially and rewrites the dataset. If no
--output path is supplied, it overwrites the input file atomically.

Mandatory flags:
    --input PATH    source guidance HDF5 file
    --exclude N [N ...]    demo indices or ranges to remove, e.g. 3 7 50-100

Optional flags:
    --output PATH   destination HDF5 file; if omitted, input is replaced in place

Example:
    python helper_scripts/filter_guidance_dataset.py --input guidance.hdf5 --exclude 5 10-15 --output guidance_filtered.hdf5
"""

import argparse
import os
import h5py
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Remove specific unaligned demos from a guidance HDF5 file.")
    parser.add_argument("--input", required=True, help="Path to the generated guidance hdf5 file")
    parser.add_argument(
        "--output", 
        default=None, 
        help="Path where the filtered hdf5 file will be saved. If omitted, the input file will be overwritten in place."
    )
    parser.add_argument(
        "--exclude", 
        type=str, 
        nargs="+", 
        required=True, 
        help="Space-separated list of demo indices or ranges to remove (e.g., --exclude 3 7 50-100 14)"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f"Input file not found: {args.input}")

    # Parse individual indices and range strings (e.g., "50-100") into a distinct set
    exclude_set = set()
    for item in args.exclude:
        if "-" in item:
            try:
                start, end = map(int, item.split("-"))
                exclude_set.update(range(start, end + 1))
            except ValueError:
                raise SystemExit(f"Invalid range format: '{item}'. Expected format like '50-100'.")
        else:
            try:
                exclude_set.add(int(item))
            except ValueError:
                raise SystemExit(f"Invalid exclusion value: '{item}'. Expected an integer or range.")

    # Use a temporary file path if overwriting in place to prevent file corruption
    if args.output is None:
        output_path = args.input + ".tmp"
        overwrite_mode = True
    else:
        output_path = args.output
        overwrite_mode = False

    with h5py.File(args.input, "r") as fin, h5py.File(output_path, "w") as fout:
        if "data" not in fin:
            if os.path.exists(output_path):
                os.remove(output_path)
            raise SystemExit("Input file is missing top-level group 'data'")

        # Parse and sort existing demos numerically
        old_demo_names = list(fin["data"].keys())
        try:
            old_demo_names = sorted(old_demo_names, key=lambda x: int(x.split("_")[-1]))
        except Exception:
            old_demo_names = sorted(old_demo_names)

        data_out = fout.create_group("data")
        # Copy global attributes
        for key, val in fin["data"].attrs.items():
            data_out.attrs[key] = val

        new_idx = 0
        for demo_name in old_demo_names:
            old_idx = int(demo_name.split("_")[-1])
            
            # Skip the sample if it matches your exclusion criteria
            if old_idx in exclude_set:
                print(f"Skipping {demo_name} (Index {old_idx})")
                continue

            # Copy the group and rename it sequentially to maintain clean indexing
            fin.copy(f"data/{demo_name}", data_out, name=f"demo_{new_idx}")
            new_idx += 1

        # Update metadata counts to reflect the remaining dataset size
        data_out.attrs["total"] = int(new_idx)

        # Re-generate the mask tracking group for standard compatibility
        mask = fout.create_group("mask")
        mask.create_dataset(
            "all",
            data=np.array([f"demo_{i}".encode() for i in range(new_idx)], dtype="S"),
        )

    # Swap the temporary file into place if overwriting
    if overwrite_mode:
        os.replace(output_path, args.input)
        final_destination = args.input
    else:
        final_destination = output_path

    print(f"\nSuccessfully filtered dataset! Saved {new_idx} demos to {final_destination}")

if __name__ == "__main__":
    main()
