"""Concatenate Robomimic-style HDF5 datasets while validating camera observation shapes.

This script can either write a single merged dataset (--output) or preserve train/val splits
by appending the additional dataset to train/validation portions based on the input counts.

Mandatory flags:
    --train_dataset PATH    path to the existing training HDF5 dataset
    --val_dataset PATH      path to the existing validation HDF5 dataset
    --additional_dataset PATH    path to the additional HDF5 dataset to append

Optional flags:
    --output_train PATH     path to write the augmented training dataset
    --output_val PATH       path to write the augmented validation dataset
    --output PATH           path to write a single concatenated output dataset
    --shuffle               shuffle appended demos before writing
    --seed INT              random seed for deterministic shuffling
    --cameras KEY [KEY ...] camera keys to validate (default: agentview_image)

Example:
    python helper_scripts/concat_datasets.py         --train_dataset dataset/train.hdf5         --val_dataset dataset/val.hdf5         --additional_dataset dataset/extra.hdf5         --output_train dataset/train_merged.hdf5         --output_val dataset/val_merged.hdf5
"""

from __future__ import annotations

import argparse
import os
import random
from typing import List, Tuple, Dict, Any

import h5py
import numpy as np


# -----------------------------
# Utils
# -----------------------------

def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def sorted_demo_names(data_group: h5py.Group) -> List[str]:
    demos = list(data_group.keys())
    try:
        return sorted(demos, key=lambda x: int(x.split("_")[-1]))
    except Exception:
        return sorted(demos)


def copy_attrs(src, dst) -> None:
    dst_attrs = dst.attrs if hasattr(dst, "attrs") else dst
    for k, v in src.items():
        dst_attrs[k] = v


# -----------------------------
# COPY LOGIC
# -----------------------------

def copy_group_recursive(
    src_group: h5py.Group,
    dst_group: h5py.Group,
) -> None:
    copy_attrs(src_group.attrs, dst_group.attrs)

    for name, item in src_group.items():
        if isinstance(item, h5py.Dataset):
            # Read and write data exactly as it is, preserving all original dimensions
            ds = dst_group.create_dataset(name, data=item[...])
            copy_attrs(item.attrs, ds)
        elif isinstance(item, h5py.Group):
            subgrp = dst_group.create_group(name)
            copy_group_recursive(item, subgrp)


# -----------------------------
# DATA COLLECTION
# -----------------------------

def collect_demos(h5_file: h5py.File):
    if "data" not in h5_file:
        raise ValueError(f"Missing 'data' group in {h5_file.filename}")
    return [(h5_file, name) for name in sorted_demo_names(h5_file["data"])]


# -----------------------------
# VALIDATION
# -----------------------------

def inspect_demo_shapes(file, demo_name: str, cameras: List[str]) -> Dict[str, Any]:
    demo = file["data"][demo_name]
    obs = demo["obs"]

    shapes = {}
    for cam in cameras:
        if cam not in obs:
            raise ValueError(f"Missing camera {cam} in {demo_name}")
        shapes[cam] = np.array(obs[cam][0]).shape
    return shapes


def check_compatibility(base_shapes, new_shapes, demo_name: str):
    for cam in base_shapes:
        if cam not in new_shapes:
            continue
        if len(base_shapes[cam]) != len(new_shapes[cam]):
            print(
                f"[WARN] Dim mismatch in {demo_name} camera {cam}: "
                f"base={base_shapes[cam]} vs new={new_shapes[cam]}"
            )


# -----------------------------
# WRITE DATASET
# -----------------------------

def write_dataset(
    output_path: str,
    demos,
    reference_file,
):
    ensure_parent_dir(output_path)

    with h5py.File(output_path, "w") as fout:
        data_out = fout.create_group("data")
        copy_attrs(reference_file["data"].attrs, data_out)

        for i, (src_file, demo_name) in enumerate(demos):
            src_demo = src_file["data"][demo_name]
            dst_demo = data_out.create_group(f"demo_{i}")
            copy_group_recursive(src_demo, dst_demo)

        data_out.attrs["total"] = len(demos)

        mask_grp = fout.create_group("mask")
        mask_grp.create_dataset(
            "all",
            data=np.array(
                [f"demo_{i}".encode() for i in range(len(demos))],
                dtype="S",
            ),
        )


# -----------------------------
# MAIN
# -----------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dataset", required=True)
    parser.add_argument("--val_dataset", required=True)
    parser.add_argument("--additional_dataset", required=True)

    parser.add_argument("--output_train", required=False, default=None)
    parser.add_argument("--output_val", required=False, default=None)
    parser.add_argument("--output", required=False, default=None, help="Output for plain concatenation")

    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["agentview_image"],
        help="camera keys to validate",
    )

    args = parser.parse_args()

    if args.output:
        if args.output_train or args.output_val:
            parser.error("Cannot combine --output with --output_train or --output_val.")
    else:
        if not args.output_train or not args.output_val:
            parser.error("Must provide either --output OR both --output_train and --output_val.")

    rng = random.Random(args.seed)

    with (
        h5py.File(args.train_dataset, "r") as train_file,
        h5py.File(args.val_dataset, "r") as val_file,
        h5py.File(args.additional_dataset, "r") as add_file,
    ):
        train_demos = collect_demos(train_file)
        val_demos = collect_demos(val_file)
        add_demos = collect_demos(add_file)

        base_shapes = inspect_demo_shapes(train_file, train_demos[0][1], args.cameras)
        print("Base shapes:", base_shapes)

        clean_add = []
        for file, demo_name in add_demos:
            shapes = inspect_demo_shapes(file, demo_name, args.cameras)
            check_compatibility(base_shapes, shapes, demo_name)
            clean_add.append((file, demo_name))

        if args.output:
            print("Mode: Plain Sequential Concatenation")
            final_all = train_demos + val_demos + clean_add
            
            if args.shuffle:
                rng.shuffle(final_all)
                
            write_dataset(args.output, final_all, train_file)
            print(f"Saved concatenated dataset ({len(final_all)} total demos) -> {args.output}")
        
        else:
            print("Mode: Ratio-Balanced Train/Val Split")
            n_train = len(train_demos)
            n_val = len(val_demos)
            ratio = n_train / (n_train + n_val)

            rng.shuffle(clean_add)

            n_add_train = round(len(clean_add) * ratio)
            add_train = clean_add[:n_add_train]
            add_val = clean_add[n_add_train:]

            final_train = train_demos + add_train
            final_val = val_demos + add_val

            if args.shuffle:
                rng.shuffle(final_train)
                rng.shuffle(final_val)

            write_dataset(args.output_train, final_train, train_file)
            write_dataset(args.output_val, final_val, val_file)
            print(f"Train: {len(final_train)} | Val: {len(final_val)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
