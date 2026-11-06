"""
Build a guidance dataset from train/val/additional HDF5 sources.

This script copies specified camera observations, optional proprio inputs, and generates
new dataset files while preserving robomimic-style HDF5 structure.

Mandatory flags:
    --train_dataset PATH    source training HDF5 dataset
    --val_dataset PATH      source validation HDF5 dataset
    --additional_dataset PATH    source additional HDF5 dataset

Optional flags:
    --output_train PATH     destination for augmented training HDF5
    --output_val PATH       destination for augmented validation HDF5
    --output PATH           destination for a single concatenated dataset
    --shuffle               shuffle appended additional demos
    --seed INT              random seed for deterministic splits
    --cameras KEY [KEY ...] camera keys to preserve and validate (default: agentview_image)

Example:
    python helper_scripts/create_guidance_dataset.py         --train_dataset dataset/train.hdf5         --val_dataset dataset/val.hdf5         --additional_dataset dataset/extra.hdf5         --output_train dataset/train_guidance.hdf5         --output_val dataset/val_guidance.hdf5
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
    for k, v in src.items():
        dst[k] = v


# -----------------------------
# IMAGE NORMALIZATION
# -----------------------------

def normalize_image(img: np.ndarray, camera: str = "") -> np.ndarray:
    """
    Convert any image format to HWC.
    Supported:
      - HWC -> OK
      - CHW -> HWC
      - HW  -> HWC (grayscale -> 1 channel)
    """
    img = np.array(img)

    if img.ndim == 2:
        img = img[..., None]
    elif img.ndim == 3:
        if img.shape[0] in [1, 3] and img.shape[-1] not in [1, 3]:
            img = np.transpose(img, (1, 2, 0))
    else:
        raise ValueError(f"[{camera}] Invalid image shape: {img.shape}")

    if img.ndim != 3:
        raise ValueError(f"[{camera}] Failed normalization: {img.shape}")

    return img


# -----------------------------
# COPY LOGIC
# -----------------------------

def copy_group_recursive(
    src_group: h5py.Group,
    dst_group: h5py.Group,
    cameras: List[str],
) -> None:
    copy_attrs(src_group.attrs, dst_group.attrs)

    for name, item in src_group.items():
        if isinstance(item, h5py.Dataset):
            data = item[...]
            if name in cameras:
                data = normalize_image(data, camera=name)

            ds = dst_group.create_dataset(name, data=data)
            copy_attrs(item.attrs, ds)
        elif isinstance(item, h5py.Group):
            subgrp = dst_group.create_group(name)
            copy_group_recursive(item, subgrp, cameras)


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
    cameras: List[str],
):
    ensure_parent_dir(output_path)

    with h5py.File(output_path, "w") as fout:
        data_out = fout.create_group("data")
        copy_attrs(reference_file["data"].attrs, data_out)

        for i, (src_file, demo_name) in enumerate(demos):
            src_demo = src_file["data"][demo_name]
            dst_demo = data_out.create_group(f"demo_{i}")
            copy_group_recursive(src_demo, dst_demo, cameras)

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

    # Changed outputs to optional, added the single unified output flag
    parser.add_argument("--output_train", required=False, default=None)
    parser.add_argument("--output_val", required=False, default=None)
    parser.add_argument("--output", required=False, default=None, help="Output for plain concatenation")

    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["agentview_image"],
        help="camera keys to validate + normalize",
    )

    args = parser.parse_args()

    # Enforce clear execution path bounds early
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

        # Branching logic based on the user output flags
        if args.output:
            print("Mode: Plain Sequential Concatenation")
            final_all = train_demos + val_demos + clean_add
            
            if args.shuffle:
                rng.shuffle(final_all)
                
            write_dataset(args.output, final_all, train_file, args.cameras)
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

            write_dataset(args.output_train, final_train, train_file, args.cameras)
            write_dataset(args.output_val, final_val, val_file, args.cameras)
            print(f"Train: {len(final_train)} | Val: {len(final_val)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
