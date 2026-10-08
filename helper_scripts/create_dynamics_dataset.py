#!/usr/bin/env python3
"""
Build train/valid dynamics HDF5 files from a robomimic-style dataset. Splits dataset according to the mask train and val.
Make sure that input dataset contains train & val masks! (Provided by split_train_val.py)

Required flags:
    --input PATH      source HDF5 file
    --out_dir PATH    output directory for the split files

Optional flags:
    --env_name NAME   env_name written into data.attrs['env_args']
    --camera KEY      keep only a specific camera obs key; repeatable
                                        if omitted, all *_image keys are kept
    --no_proprio      skip writing obs/proprio
    --train_mask KEY  mask name used for the train split
    --valid_mask KEY  mask name used for the valid split
"""

import argparse
import json
import os
import h5py
import numpy as np


LOWDIM_KEYS_FOR_PROPRIO = [
    "robot0_eef_pos",       # (T, 3)
    "robot0_eef_quat",      # (T, 4)
    "robot0_gripper_qpos",  # (T, 2)
    # "object",               # (T, 14)
]


def _is_camera_key(name):
    return name.endswith("_image")

def _decode_demo_list(mask_dataset):
    return [x.decode("utf-8") for x in np.array(mask_dataset)]

def _maybe_copy_dataset(in_grp, out_grp, name, compression=True):
    if name not in in_grp:
        return
    data = in_grp[name][...]
    if compression and data.nbytes > 1_000_000:
        out_grp.create_dataset(name, data=data, compression="gzip", compression_opts=4, shuffle=True)
    else:
        out_grp.create_dataset(name, data=data)

def _copy_attrs(src, dst):
    for k, v in src.attrs.items():
        dst.attrs[k] = v

def _set_env_name(data_grp, env_name):
    if "env_args" not in data_grp.attrs:
        return

    raw = data_grp.attrs["env_args"]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")

    env_args = json.loads(raw)
    env_args["env_name"] = env_name
    data_grp.attrs["env_args"] = json.dumps(env_args)

def _build_proprio(obs_grp):
    parts = []
    for k in LOWDIM_KEYS_FOR_PROPRIO:
        if k not in obs_grp:
            raise KeyError(f"Missing obs key '{k}' needed for proprio")
        parts.append(obs_grp[k][...].astype(np.float32))
    proprio = np.concatenate(parts, axis=-1).astype(np.float32)  # (T, 9)
    return proprio

def _write_split(in_f, demo_keys, out_path, camera_keys=None, add_proprio=True, env_name=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with h5py.File(out_path, "w") as out_f:
        out_data = out_f.create_group("data")
        _copy_attrs(in_f["data"], out_data)  # keeps env_args etc. if present

        total = 0
        proprio_sum = None
        proprio_sumsq = None
        proprio_count = 0

        for new_idx, demo_key in enumerate(demo_keys):
            in_demo = in_f["data"][demo_key]
            out_demo = out_data.create_group(f"demo_{new_idx}")
            _copy_attrs(in_demo, out_demo)

            # copy core datasets if present
            for name in ["actions", "dones", "rewards", "states", "actions_abs"]:
                _maybe_copy_dataset(in_demo, out_demo, name)

            # obs group (all camera views by default + lowdim for proprio + proprio)
            out_obs = out_demo.create_group("obs")
            in_obs = in_demo["obs"]

            if camera_keys is None:
                selected_camera_keys = [key for key in in_obs.keys() if _is_camera_key(key)]
            else:
                selected_camera_keys = list(camera_keys)

            if not selected_camera_keys:
                raise KeyError(f"No camera observations found in demo '{demo_key}'")

            for camera_key in selected_camera_keys:
                if camera_key not in in_obs:
                    raise KeyError(f"Missing camera obs '{camera_key}' in demo '{demo_key}'")
                _maybe_copy_dataset(in_obs, out_obs, camera_key)

            if add_proprio:
                if "proprio" in in_obs:
                    # Skip construction: copy the existing dataset directly
                    _maybe_copy_dataset(in_obs, out_obs, "proprio")
                    proprio = in_obs["proprio"][...]
                else:
                    # Fallback: copy lowdim sources and construct it manually
                    for k in LOWDIM_KEYS_FOR_PROPRIO:
                        _maybe_copy_dataset(in_obs, out_obs, k)
                    proprio = _build_proprio(in_obs)
                    out_obs.create_dataset("proprio", data=proprio, compression="gzip", compression_opts=4, shuffle=True)

                # Accumulate stats for global mean/std metadata
                if proprio_sum is None:
                    proprio_sum = proprio.sum(axis=0)
                    proprio_sumsq = (proprio ** 2).sum(axis=0)
                else:
                    proprio_sum += proprio.sum(axis=0)
                    proprio_sumsq += (proprio ** 2).sum(axis=0)
                proprio_count += proprio.shape[0]

            # keep num_samples consistent
            if "actions" in in_demo:
                out_demo.attrs["num_samples"] = int(in_demo["actions"].shape[0])
                total += int(in_demo["actions"].shape[0])

        out_data.attrs["total"] = int(total)

        if env_name is not None:
            _set_env_name(out_data, env_name)

        if add_proprio and proprio_count > 0:
            mean = proprio_sum / proprio_count
            var = proprio_sumsq / proprio_count - mean**2
            std = np.sqrt(np.maximum(var, 1e-12))
            out_data.attrs["proprio_mean"] = mean.astype(np.float32)
            out_data.attrs["proprio_std"] = std.astype(np.float32)

        # create mask group so robomimic tools still work
        mask = out_f.create_group("mask")
        mask.create_dataset("all", data=np.array([f"demo_{i}".encode("utf-8") for i in range(len(demo_keys))]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to square_d0_visual.hdf5")
    ap.add_argument("--out_dir", required=True, help="output directory for dynaguide split")
    ap.add_argument(
        "--env_name",
        default="NutAssemblySquare",
        help="env_name to write into data.attrs['env_args'] in the output files.",
    )
    ap.add_argument(
        "--camera",
        action="append",
        default=None,
        help="camera obs key to keep (repeatable). If omitted, keep all camera image keys.",
    )
    ap.add_argument("--no_proprio", action="store_true", help="do not add obs/proprio")
    ap.add_argument("--train_mask", default="train", help="mask key name for train split (default: train)")
    ap.add_argument("--valid_mask", default="valid", help="mask key name for valid split (default: valid)")
    args = ap.parse_args()

    train_out = os.path.join(args.out_dir, "square_train.hdf5")
    valid_out = os.path.join(args.out_dir, "square_valid.hdf5")

    with h5py.File(args.input, "r") as in_f:
        if "mask" not in in_f:
            raise RuntimeError("Input file has no 'mask' group (no filter keys found).")

        train_keys = _decode_demo_list(in_f[f"mask/{args.train_mask}"])
        valid_keys = _decode_demo_list(in_f[f"mask/{args.valid_mask}"])

        _write_split(
            in_f, train_keys, train_out,
            camera_keys=args.camera,
            add_proprio=(not args.no_proprio),
            env_name=args.env_name,
        )
        _write_split(
            in_f, valid_keys, valid_out,
            camera_keys=args.camera,
            add_proprio=(not args.no_proprio),
            env_name=args.env_name,
        )

    print("Wrote:")
    print(" ", train_out)
    print(" ", valid_out)

if __name__ == "__main__":
    main()