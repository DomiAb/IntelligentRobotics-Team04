"""
Automated sweep runner to evaluate DynaGuide performance across different
guidance directions and dataset sizes.
"""
import os
import sys
import json
import argparse
import subprocess
import h5py
import numpy as np

# -----------------------------
# DIRECTORY SETTINGS
# -----------------------------
ORIGINAL_DATASETS_DIR = "dataset/guidance"
OUTPUT_BASE_DIR = "evaluation_guidance_sizes"

# -----------------------------
# DEFAULT RUNNING PARAMETERS
# -----------------------------
RUN_PARAMS = {
    "horizon": 200,
    "n_rollouts": 50,
    "dp_model": "dataset/diffusion_model_epoch_500.pth",
    "env": "NutAssemblySquare",
    "video_skip": 2,
    "dynamics_model": "dataset/dynamics_model_epoch_999.pth",
    "camera_names": "agentview_image",
    "seed": 0,
    "dataset_obs": True
}

# -----------------------------
# SWEEP VALUES
# -----------------------------
SWEEP_SIZES = [5, 10, 15, 20, 30, 40, 50]

# -----------------------------
# HDF5 UTILITIES
# -----------------------------
def copy_attrs(src, dst):
    dst_attrs = dst.attrs if hasattr(dst, "attrs") else dst
    for k, v in src.items():
        dst_attrs[k] = v

def copy_group_recursive(src_group, dst_group):
    copy_attrs(src_group.attrs, dst_group.attrs)
    for name, item in src_group.items():
        if isinstance(item, h5py.Dataset):
            ds = dst_group.create_dataset(name, data=item[...])
            copy_attrs(item.attrs, ds)
        elif isinstance(item, h5py.Group):
            subgrp = dst_group.create_group(name)
            copy_group_recursive(item, subgrp)

def create_subset_dataset(input_path, output_path, N):
    """Slice the first N demos from input_path and write to output_path."""
    print(f"Creating subset: {output_path} (N={N} from {os.path.basename(input_path)})")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with h5py.File(input_path, "r") as fin:
        demos = list(fin["data"].keys())
        try:
            demos = sorted(demos, key=lambda x: int(x.split("_")[-1]))
        except Exception:
            demos = sorted(demos)
        
        selected = demos[:min(N, len(demos))]
        if len(selected) < N:
            print(f"[Warning] Requested N={N} but only {len(selected)} demos available in {input_path}")
            
        with h5py.File(output_path, "w") as fout:
            data_out = fout.create_group("data")
            copy_attrs(fin["data"].attrs, data_out)
            for i, demo_name in enumerate(selected):
                src_demo = fin["data"][demo_name]
                dst_demo = data_out.create_group(f"demo_{i}")
                copy_group_recursive(src_demo, dst_demo)
            
            data_out.attrs["total"] = len(selected)
            mask_grp = fout.create_group("mask")
            mask_grp.create_dataset(
                "all",
                data=np.array(
                    [f"demo_{i}".encode() for i in range(len(selected))],
                    dtype="S",
                ),
            )

def concatenate_datasets(input_paths, output_path):
    """Concatenate all demos from multiple files into one."""
    print(f"Concatenating to {output_path} from {[os.path.basename(p) for p in input_paths]}")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    all_demos_to_write = []
    opened_files = []
    try:
        for p in input_paths:
            f = h5py.File(p, "r")
            opened_files.append(f)
            demos = list(f["data"].keys())
            try:
                demos = sorted(demos, key=lambda x: int(x.split("_")[-1]))
            except Exception:
                demos = sorted(demos)
            for d in demos:
                all_demos_to_write.append((f, d))
                
        with h5py.File(output_path, "w") as fout:
            data_out = fout.create_group("data")
            if opened_files:
                copy_attrs(opened_files[0]["data"].attrs, data_out)
            
            for i, (src_file, demo_name) in enumerate(all_demos_to_write):
                src_demo = src_file["data"][demo_name]
                dst_demo = data_out.create_group(f"demo_{i}")
                copy_group_recursive(src_demo, dst_demo)
                
            data_out.attrs["total"] = len(all_demos_to_write)
            mask_grp = fout.create_group("mask")
            mask_grp.create_dataset(
                "all",
                data=np.array(
                    [f"demo_{i}".encode() for i in range(len(all_demos_to_write))],
                    dtype="S",
                ),
            )
    finally:
        for f in opened_files:
            f.close()

# -----------------------------
# COMMAND RUNNING UTILITY
# -----------------------------
def run_dynaguide_experiment(config_path, output_folder, run_params):
    cmd = [
        "conda", "run", "-n", "dynaguide_paul", "python", "dyna_scripts/run_dynaguide.py",
        "--output_folder", output_folder,
        "--horizon", str(run_params["horizon"]),
        "--n_rollouts", str(run_params["n_rollouts"]),
        "--dp_model", run_params["dp_model"],
        "--env", run_params["env"],
        "--video_skip", str(run_params["video_skip"]),
        "--exp_setup_config", config_path,
        "--dynamics_model", run_params["dynamics_model"],
        "--camera_names", run_params["camera_names"],
        "--seed", str(run_params["seed"])
    ]
    if run_params.get("render_denoising"):
        cmd.append("--render_denoising")
    if run_params.get("dataset_obs"):
        cmd.append("--dataset_obs")
    if run_params.get("save_images"):
        cmd.append("--save_images")
        
    print(f"\n>>> Running command:\n    {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"[Error] Command failed with return code {result.returncode}:\n{result.stderr}")
    return result.returncode == 0

# -----------------------------
# MAIN SWEEP LOOP
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="DynaGuide Direction and Size Evaluation Sweep")
    parser.add_argument("--dry-run", action="store_true", help="Run a single N=5, 1 rollout execution to verify end-to-end functionality")
    args = parser.parse_args()
    
    # Load base config
    base_config_path = "dynaguide_env_config.json"
    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"Base configuration file {base_config_path} not found in workspace root.")
        
    with open(base_config_path, "r") as f:
        base_config = json.load(f)
        
    # Override parameters for dry run
    params = RUN_PARAMS.copy()
    sweep_sizes = SWEEP_SIZES
    if args.dry_run:
        print("[Dry-Run Mode] Overriding sweep parameters to N=5, n_rollouts=1")
        params["n_rollouts"] = 1
        sweep_sizes = [5]
        
    results_summary = []
    
    for N in sweep_sizes:
        print(f"\n==========================================")
        print(f"  Processing dataset size: N = {N}")
        print(f"==========================================")
        
        # Create N-specific target subfolder under OUTPUT_BASE_DIR
        n_folder = os.path.join(OUTPUT_BASE_DIR, f"n_{N}")
        os.makedirs(n_folder, exist_ok=True)
        
        # Paths for directional subsets inside n_folder
        back_sub_path = os.path.join(n_folder, f"guidance_back_{N}.hdf5")
        left_sub_path = os.path.join(n_folder, f"guidance_left_{N}.hdf5")
        right_sub_path = os.path.join(n_folder, f"guidance_right_{N}.hdf5")
        all_sub_path = os.path.join(n_folder, f"guidance_all_{3*N}.hdf5")
        
        # Create subsets from ORIGINAL_DATASETS_DIR
        create_subset_dataset(os.path.join(ORIGINAL_DATASETS_DIR, "guidance_back.hdf5"), back_sub_path, N)
        create_subset_dataset(os.path.join(ORIGINAL_DATASETS_DIR, "guidance_left.hdf5"), left_sub_path, N)
        create_subset_dataset(os.path.join(ORIGINAL_DATASETS_DIR, "guidance_right.hdf5"), right_sub_path, N)
        
        # Concatenate subsets
        concatenate_datasets([back_sub_path, left_sub_path, right_sub_path], all_sub_path)
        
        # Conditions to run: (mode_name, pos_examples_path, total_samples)
        conditions = [
            ("back", back_sub_path, N),
            ("all", all_sub_path, 3*N)
        ]
        
        for name, pos_path, total_s in conditions:
            print(f"\n--- Running Condition: {name.capitalize()} (Total Samples: {total_s}) ---")
            
            # Define output folder inside n_folder
            output_folder = os.path.join(n_folder, f"eval_robomimic_{name}_{total_s}/")
            os.makedirs(output_folder, exist_ok=True)
            
            # Write condition specific JSON config inside the output folder
            config_copy = base_config.copy()
            config_copy["pos_examples"] = pos_path
            
            cond_config_path = os.path.join(output_folder, f"config_{name}_{total_s}.json")
            with open(cond_config_path, "w") as f:
                json.dump(config_copy, f, indent=4)
                
            # Run simulation experiment using the saved config
            success = run_dynaguide_experiment(cond_config_path, output_folder, params)
            
            # Record summary info
            results_summary.append({
                "N": N,
                "total_samples": total_s,
                "direction": name,
                "success": success,
                "output_folder": output_folder
            })
            
    print("\n==========================================")
    print("  Evaluation Sweep Completed Summary")
    print("==========================================")
    for res in results_summary:
        status = "SUCCESS" if res["success"] else "FAILED"
        print(f"Direction: {res['direction']:<5} | Size: {res['total_samples']:<3} | Status: {status} | Output: {res['output_folder']}")

if __name__ == "__main__":
    main()
