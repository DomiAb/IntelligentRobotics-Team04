"""
Automated sweep runner to evaluate DynaGuide performance across different
guidance directions and dataset sizes.
"""
import os
import sys
import json
import argparse
import subprocess

# -----------------------------
# TARGET SAMPLE SPECIFICATION
# -----------------------------
TARGET_SAMPLE_FOLDER = "n_20_sample_6_dp_300"

# -----------------------------
# DIRECTORY SETTINGS
# -----------------------------
ORIGINAL_DATASETS_DIR = "evaluation_guidance_directions/"
OUTPUT_BASE_DIR = "evaluation_guidance_directions"
base_config_path = "configs/dynaGuide/dynaguide_env_config.json"

# Fixed dataset size based on image context (N=20)
N = 20

# -----------------------------
# DEFAULT RUNNING PARAMETERS
# -----------------------------
RUN_PARAMS = {
    "horizon": 200,
    "n_rollouts": 50,
    "dp_model": "dataset/diffusion_model_epoch_300.pth",
    "env": "NutAssemblySquare",
    "video_skip": 2,
    "dynamics_model": "dataset/dynamics_model_epoch_999.pth",
    "camera_names": "agentview_image",
    "seed": 0,
    "dataset_obs": True,
    "save_images": True
}

# -----------------------------
# COMMAND RUNNING UTILITY
# -----------------------------
def run_dynaguide_experiment(config_path, output_folder, run_params):

#>>>>>>>>>>>> DO NOT FORGET TO DEFINE YOUR USED CONDA ENVIRONMENT BELOW THIS COMMENT. YOU HAVE TO REPLACE "CONDA_ENV" WITH THE NAME OF YOUR ENV <<<<<<<<<<<<

    cmd = [
        "conda", "run", "-n", "CONDA_ENV", "python", "dyna_scripts/run_dynaguide.py",
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


#>>>>>>>>>>>> YOU DO NOT HAVE TO CHANGE BELOW THIS LINE <<<<<<<<<<<<


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

def main():
    parser = argparse.ArgumentParser(description="DynaGuide Direction and Evaluation Run")
    parser.add_argument("--dry-run", action="store_true", help="Run a single rollout execution to verify end-to-end functionality")
    args = parser.parse_args()
    
 
    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"Base configuration file {base_config_path} not found in workspace root.")
        
    with open(base_config_path, "r") as f:
        base_config = json.load(f)
        
    # Override parameters for dry run
    params = RUN_PARAMS.copy()
    if args.dry_run:
        print("[Dry-Run Mode] Overriding parameters to n_rollouts=1")
        params["n_rollouts"] = 1
        
    results_summary = []
    
    print(f"\n==========================================")
    print(f"  Processing folder: {TARGET_SAMPLE_FOLDER}")
    print(f"==========================================")
    
    # Target folder path under base output directory
    target_folder_path = os.path.join(OUTPUT_BASE_DIR, TARGET_SAMPLE_FOLDER)
    os.makedirs(target_folder_path, exist_ok=True)
    
    # Absolute paths to existing dataset files
    conditions = [
        ("back", os.path.join(ORIGINAL_DATASETS_DIR, f"guidance_back_{N}.hdf5")),
        ("left", os.path.join(ORIGINAL_DATASETS_DIR, f"guidance_left_{N}.hdf5")),
        ("right", os.path.join(ORIGINAL_DATASETS_DIR, f"guidance_right_{N}.hdf5")),
        ("all", os.path.join(ORIGINAL_DATASETS_DIR, f"guidance_all_{N}.hdf5"))
    ]
    
    for name, pos_path in conditions:
        print(f"\n--- Running Condition: {name.capitalize()} ---")
        
        if not os.path.exists(pos_path):
            print(f"[Error] Dataset not found at: {pos_path}. Skipping this condition.")
            continue
            
        # Define condition-specific subfolder directly inside TARGET_SAMPLE_FOLDER
        output_folder = os.path.join(target_folder_path, name)
        os.makedirs(output_folder, exist_ok=True)
        
        # Write condition specific JSON config inside the specific subfolder
        config_copy = base_config.copy()
        config_copy["pos_examples"] = pos_path
        
        cond_config_path = os.path.join(output_folder, f"config_{name}.json")
        with open(cond_config_path, "w") as f:
            json.dump(config_copy, f, indent=4)
            
        # Run simulation experiment using the saved config
        success = run_dynaguide_experiment(cond_config_path, output_folder, params)
        
        # Record summary info
        results_summary.append({
            "direction": name,
            "success": success,
            "output_folder": output_folder
        })
            
    print("\n==========================================")
    print("  Evaluation Run Completed Summary")
    print("==========================================")
    for res in results_summary:
        status = "SUCCESS" if res["success"] else "FAILED"
        print(f"Direction: {res['direction']:<5} | Status: {status} | Output: {res['output_folder']}")

if __name__ == "__main__":
    main()