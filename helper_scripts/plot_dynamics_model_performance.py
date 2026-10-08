"""
python helper_scripts/plot_latent_space.py \
    --dataset_path path/to/output/dynaguide_env_data.hdf5 \
    --dynamics_model path/to/models/dynaguide_checkpoint.pth \
    --good_states path/to/datasets/pos_examples.hdf5 \
    --output_path path/to/output/latent_space_manifold.png \
    --action_dim 7 \
    --chunk_length 16 \
    --execution_horizon 8 \
    --method pca or tsne \
    --perplexity 30


f your plot looks like a single dense cloud, lower the perplexity (e.g., 15). 
If the points look completely fragmented into tiny independent islands, raise it (e.g., 50).
"""

import os
import sys

# Resolve the absolute path of the project root directory (one level up from helper_scripts/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "helper_scripts" else SCRIPT_DIR

# Ensure the root project folder is searched first for local modules like 'core'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Handle local repository overrides for robosuite and mimicgen cleanly
for local_repo in ["robosuite", "mimicgen"]:
    repo_path = os.path.join(PROJECT_ROOT, local_repo)
    if os.path.exists(repo_path) and not os.path.exists(os.path.join(repo_path, "__init__.py")):
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

import argparse
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

# Local core modules from your repository
from core.dynamics_models import FinalStatePredictionDino
from core.embedder_datasets import MultiviewDataset
from core.dynaguide import calculate_classifier_guidance

def main():
    parser = argparse.ArgumentParser(description="Offline Latent Space Analysis for DynaGuide")
    
    # Core File Paths
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the evaluated rollouts HDF5 file database")
    parser.add_argument("--dynamics_model", type=str, required=True, help="Path to the trained DynaGuide dynamics checkpoint (.pth)")
    parser.add_argument("--good_states", type=str, required=True, help="Path to the positive target examples HDF5 file")
    parser.add_argument("--output_path", type=str, default="latent_space_analysis.png", help="Destination path for the generated PNG plot")
    
    # Reduction Parameters
    parser.add_argument("--method", type=str, choices=["pca", "tsne"], default="pca", help="Dimensionality reduction method to apply (pca or tsne)")
    parser.add_argument("--perplexity", type=int, default=30, help="Perplexity parameter specifically for tuning t-SNE localized neighborhood spacing")
    
    # Task Hyperparameters
    parser.add_argument("--camera", type=str, default="agentview_image", help="Camera modality name stored in the datasets")
    parser.add_argument("--action_dim", type=int, default=7, help="Dimension size of the action vector space")
    parser.add_argument("--chunk_length", type=int, default=16, help="Action chunk prediction horizon sequence length")
    parser.add_argument("--execution_horizon", type=int, default=8, help="Step interval used between live policy re-planning calls")
    parser.add_argument("--proprio", type=str, default="proprio", help="Type of proprioception to use ('proprio' or None)")
    parser.add_argument("--proprio_dim", type=int, default=9, help="Dimension of the proprioception vector (e.g., 9 for EEF position/quat/gripper)")
    parser.add_argument("--scale", type=float, default=1.5, help="Guidance scale factor parameter required by the signature")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(">>> Initializing Dynamics Model...")
    model = FinalStatePredictionDino(
        args.action_dim, 
        args.chunk_length, 
        cameras=[args.camera], 
        reconstruction=True,
        proprio=args.proprio,
        proprio_dim=args.proprio_dim
    )
    model.load_state_dict(torch.load(args.dynamics_model, map_location=device))
    model.to(device).eval()

    print(">>> Encoding Guidance Target Manifest Clusters...")
    print(">>> Initializing Target Dataset with Padding...")
    good_dataset = MultiviewDataset(
        args.good_states, 
        action_chunk_length=args.chunk_length, 
        cameras=[args.camera],
        padding=True,
        pad_mode="repeat",
        proprio=args.proprio
    )
    print(">>> Encoding Guidance Target Manifest Clusters...")
    _, good_embeddings, _ = calculate_classifier_guidance(
        model, 
        good_dataset=good_dataset, 
        main_camera=args.camera, 
        scale=args.scale
    )
    good_np = good_embeddings.detach().cpu().numpy()

    print(f">>> Crawling Rollout Database: {args.dataset_path}")
    rollout_currents = []
    rollout_preds = []
    
    with h5py.File(args.dataset_path, "r") as f:
        for demo_key in f["data"].keys():
            obs_grp = f[f"data/{demo_key}/obs"]
            obs_images = obs_grp[args.camera][:]
            actions = f[f"data/{demo_key}/actions"][:]
            
            has_proprio_key = "proprio" in obs_grp
            has_split_keys = all(k in obs_grp for k in ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"])
            
            if has_proprio_key:
                proprio_data = obs_grp["proprio"][:]
            elif has_split_keys:
                proprio_data = np.concatenate([
                    obs_grp["robot0_eef_pos"][:],
                    obs_grp["robot0_eef_quat"][:],
                    obs_grp["robot0_gripper_qpos"][:]
                ], axis=-1)
            else:
                proprio_data = None

            demo_curr = []
            demo_pred = []
            
            # Change loop boundary to guarantee execution even for short rollouts
            loop_end = max(args.execution_horizon, len(actions) - args.chunk_length + 1)
            for i in range(0, loop_end, args.execution_horizon):
                # Clamp index to prevent out-of-bounds reading
                i = min(i, max(0, len(actions) - 1))
                raw_img = obs_images[i]
                if raw_img.ndim == 4:
                    raw_img = raw_img[-1]
                
                img_tensor = torch.tensor(raw_img).permute(2, 0, 1).unsqueeze(0).to(device).float()
                state_feed = {args.camera: img_tensor}
                
                if proprio_data is not None:
                    raw_proprio = proprio_data[i]
                    if raw_proprio.ndim == 2:
                        raw_proprio = raw_proprio[-1]
                    state_feed["proprio"] = torch.tensor(raw_proprio).unsqueeze(0).to(device).float()
                
                # Extract action sequence and pad with trailing repeat if sequence ends early
                chunk_actions = actions[i : i + args.chunk_length]
                if len(chunk_actions) < args.chunk_length:
                    pad_size = args.chunk_length - len(chunk_actions)
                    chunk_actions = np.concatenate([chunk_actions, np.repeat(chunk_actions[-1:], pad_size, axis=0)], axis=0)
                
                action_feed = torch.tensor(chunk_actions).unsqueeze(0).to(device).float()
                
                with torch.no_grad():
                    # Extract the model's unified lookahead representation
                    pred_embed, _ = model(state_feed, action_feed)
                    flattened_embed = pred_embed.flatten(start_dim=1).cpu().numpy()
                    
                    # Store both current tracking and lookahead in the same space
                    demo_curr.append(flattened_embed)
                    demo_pred.append(flattened_embed)

            if demo_curr:
                rollout_currents.append(np.concatenate(demo_curr, axis=0))
                rollout_preds.append(np.concatenate(demo_pred, axis=0))

    if not rollout_currents:
        print("Error: No valid rollout steps were extracted.")
        return

    # Flatten lists for global dimensionality reduction fitting
    flat_currents = np.concatenate(rollout_currents, axis=0)
    flat_preds = np.concatenate(rollout_preds, axis=0)

    print(f">>> Computing {args.method.upper()} Manifold Reductions...")
    all_embeddings = np.concatenate([good_np, flat_currents, flat_preds], axis=0)
    
    # ADD THIS: Scale features to unit variance to balance the latent space distributions
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    all_embeddings_scaled = scaler.fit_transform(all_embeddings)
    
    if args.method == "pca":
        reducer = PCA(n_components=2)
    elif args.method == "tsne":
        reducer = TSNE(
            n_components=2, 
            perplexity=args.perplexity, 
            max_iter=1000, 
            random_state=42, 
            init='pca'
        )

    # Change fit_transform to use the scaled embeddings matrix
    coords = reducer.fit_transform(all_embeddings_scaled)

    # Slice the projected coordinates back into their specific domains
    idx_good_end = len(good_np)
    idx_curr_end = idx_good_end + len(flat_currents)
    
    coords_good = coords[:idx_good_end]
    coords_curr = coords[idx_good_end:idx_curr_end]
    coords_pred = coords[idx_curr_end:]

    print(">>> Rendering Individual Spatial Visualizations per Rollout...")
    cmap = plt.get_cmap('jet')
    curr_ptr = 0
    
    # Separate the base output path string from its file extension
    base_out, ext = os.path.splitext(args.output_path)
    
    # Calculate the fixed global limits of the shared coordinate canvas space
    x_min, x_max = coords[:, 0].min() - 15, coords[:, 0].max() + 15
    y_min, y_max = coords[:, 1].min() - 15, coords[:, 1].max() + 15

    # Iterate through every single extracted rollout sequence
    for idx, demo_np in enumerate(rollout_currents):
        length = len(demo_np)
        
        # Extract slices for this specific rollout iteration
        c_coords = coords_curr[curr_ptr : curr_ptr + length]
        p_coords = coords_pred[curr_ptr : curr_ptr + length]
        curr_ptr += length
        
        if length < 2:
            continue
            
        # Initialize a brand new independent canvas for this rollout
        plt.figure(figsize=(12, 12))
        
        # 1. Plot static Expert Guidance Targets as background landmarks
        plt.scatter(coords_good[:, 0], coords_good[:, 1], 
                    c='green', alpha=0.9, s=110, marker='X', label='Guidance Targets (Goals)', zorder=2)
        
        time_steps = np.linspace(0.0, 1.0, num=length)
        colors = cmap(time_steps)
        
        # Plot the faint continuous physical line path of the robot arm
        plt.plot(c_coords[:, 0], c_coords[:, 1], color='gray', alpha=0.3, linewidth=1.5, zorder=1)
        
        # Step horizon offset calculation for lookahead comparison (16 / 8 = 2)
        step_offset = args.chunk_length // args.execution_horizon
        
        for step in range(length):
            # Actual state s_t (Solid Dot)
            plt.scatter(c_coords[step, 0], c_coords[step, 1], color=colors[step], alpha=0.9, s=40, marker='o', zorder=4)
            
            # Predicted state \hat{s}_{t+H} (Hollow Dot)
            # Offset the visual circle slightly to show lookahead direction clearly if perfectly overlapping
            plt.scatter(p_coords[step, 0], p_coords[step, 1], edgecolors=colors[step], facecolors='none', alpha=0.8, s=40, linewidths=1.2, zorder=3)
            
            # Connect the lookahead prediction to its true future outcome state
            if step + step_offset < length:
                outcome_coord = c_coords[step + step_offset]
                plt.plot([p_coords[step, 0], outcome_coord[0]], [p_coords[step, 1], outcome_coord[1]], 
                         color=colors[step], alpha=0.5, linewidth=1, linestyle=':', zorder=2)

        # Lock the axes limits globally so the frame context remains perfectly stationary
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        # 2. Add a persistent progression colorbar
        sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=100))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.046, pad=0.04)
        cbar.set_label("Relative Rollout Progression (% of Total Steps)", fontsize=11, labelpad=10)
        cbar.ax.yaxis.set_major_locator(plt.FixedLocator([0, 20, 40, 60, 80, 100]))
        cbar.ax.set_yticklabels(['0% (Start)', '20%', '40%', '60%', '80%', '100% (End)'])

        # 3. Clean manual legend layout
        legend_elements = [
            Line2D([0], [0], marker='X', color='w', markerfacecolor='green', markersize=9, label='Expert Targets'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=7, label=r'Actual State ($s_t$)'),
            Line2D([0], [0], marker='o', color='w', markeredgecolor='gray', markerfacecolor='none', markersize=7, markeredgewidth=1.2, label=r'Predicted Lookahead ($\hat{s}_{t+H}$)'),
            Line2D([0], [0], color='gray', linestyle=':', linewidth=1, label='True Error Vector')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        # Headings, structural labels, and layout grid setup
        plt.suptitle("Dynamics Model Lookahead Alignment Analysis", fontsize=14, fontweight='bold')
        if args.method == "tsne":
            plt.title(f"Method: t-SNE (Perplexity: {args.perplexity}) | Rollout: {idx+1}", fontsize=11, pad=10)
        else:
            plt.title(f"Method: {args.method.upper()} | Rollout: {idx+1}", fontsize=11, pad=10)
            
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True, linestyle=':', alpha=0.6)

        # Build a dynamically indexed destination string signature
        individual_output_path = f"{base_out}_rollout_{idx+1}{ext}"
        
        output_dir = os.path.dirname(individual_output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Export the distinct standalone image asset 
        plt.savefig(individual_output_path, dpi=225)
        plt.close()
        print(f"--> Exported isolated rollout map successfully: {individual_output_path}")

    print("-" * 60 + "\n>>> Execution complete! All individual rollout plots compiled.")

if __name__ == "__main__":
    main()