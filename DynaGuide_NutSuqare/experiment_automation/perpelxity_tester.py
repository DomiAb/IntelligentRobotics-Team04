import os
import subprocess

# ==========================================
#               CONFIGURATION
# ==========================================
# Define your sweep ranges
START_PERPLEXITY = 10
END_PERPLEXITY = 200
STEP_SIZE = 5

# Hardcoded absolute or relative project paths
DATASET_PATH = "evaluation_2406/eval_robomimic_tsne2/rollout_database.hdf5"
GUIDANCE_PATH = "trained_models/result100.pth"  # Adjust to your model filename
GOOD_STATES_PATH = "dataset/square_splits/square_pos_examples_right.hdf5" # Adjust to your targets file
OUTPUT_DIR = "evaluation_2506/tsne_sweeps"

# Paths to python executable and plotting script
PYTHON_EXE = "python" 
PLOT_SCRIPT = "helper_scripts/plot_latent_space.py"
# ==========================================

def main():
    # Guarantee output folder structure exists safely
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate the parameter sweep sequence
    perplexity_values = list(range(START_PERPLEXITY, END_PERPLEXITY + 1, STEP_SIZE))
    total_runs = len(perplexity_values)
    
    print(f">>> Beginning t-SNE perplexity sweep across {total_runs} steps...")
    print(f">>> Target Directory: {OUTPUT_DIR}\n" + "-"*50)
    
    for idx, perp in enumerate(perplexity_values, 1):
        # Construct unique specific filename suffix string 
        output_filename = f"latent_space_manifold_p{perp}.png"
        full_output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Build the exact execution argument string vector 
        cmd = [
            PYTHON_EXE, PLOT_SCRIPT,
            "--dataset_path", DATASET_PATH,
            "--dynamics_model", GUIDANCE_PATH,
            "--good_states", GOOD_STATES_PATH,
            "--output_path", full_output_path,
            "--method", "tsne",
            "--perplexity", str(perp)
        ]
        
        print(f"[{idx}/{total_runs}] Processing perplexity = {perp} -> {output_filename}...")
        
        try:
            # Trigger subprocess execution loop 
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Run failed at perplexity {perp}!")
            print(f"Terminal Error Code Output:\n{e.stderr}")
            continue
            
    print("-"*50 + f"\n>>> Sweep complete! All renders compiled inside: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()