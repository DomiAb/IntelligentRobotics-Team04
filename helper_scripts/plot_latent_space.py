"""
python helper_scripts/plot_latent_space.py \
    --dynamics_model path/to/models/dynaguide_checkpoint.pth \
    --good_states path/to/datasets/pos_examples.hdf5 \
    --guided_rollout_path path/to/guided_rollouts.hdf5 \
    --unguided_rollout_path path/to/unguided_rollouts.hdf5 \
    --output_path path/to/output/guidance_ablation_manifold.png \
    --action_dim 7 \
    --chunk_length 16 \
    --execution_horizon 8 \
    --method pca or tsne \
    --perplexity 30

If the plot looks like a single dense cloud, lower the perplexity (e.g., 15).
If the points look fragmented into tiny islands, raise it (e.g., 50).
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Local core modules from your repository
from core.dynamics_models import FinalStatePredictionDino
from core.embedder_datasets import MultiviewDataset
from core.dynaguide import calculate_classifier_guidance


def build_synthetic_image_from_state(state_vector):
    state_vector = np.asarray(state_vector, dtype=np.float32).reshape(-1)
    state_vector = np.nan_to_num(state_vector, nan=0.0, posinf=0.0, neginf=0.0)

    if state_vector.size == 0:
        return np.zeros((8, 8, 3), dtype=np.uint8)

    height, width, channels = 8, 8, 3
    size = height * width * channels
    values = np.pad(state_vector[:size], (0, max(0, size - state_vector.size)))
    values = values[:size].reshape(height, width, channels)

    values = np.abs(values)
    if values.max() > 0:
        values = (values / values.max()) * 255.0
    else:
        values = np.zeros_like(values)

    return values.astype(np.uint8)


def load_rollout_embeddings(path, model, args, device):
    rollout_by_demo = []

    print(f">>> Crawling Rollout Database: {path}")
    with h5py.File(path, "r") as handle:
        for demo_key in handle["data"].keys():
            demo_grp = handle[f"data/{demo_key}"]
            obs_grp = demo_grp["obs"] if "obs" in demo_grp else None
            actions = demo_grp["actions"][:]

            use_state_fallback = obs_grp is None
            if use_state_fallback:
                if "states" not in demo_grp:
                    print(f"Skipping demo {demo_key}: no observations or states found.")
                    continue
                state_data = demo_grp["states"][:]
                if state_data.ndim == 1:
                    state_data = state_data[None, :]
                print(f"Warning: {demo_key} does not contain an obs group; using synthetic image input derived from state vectors.")
            else:
                obs_images = obs_grp[args.camera][:]

                has_proprio_key = "proprio" in obs_grp
                has_split_keys = all(k in obs_grp for k in ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"])

                if has_proprio_key:
                    proprio_data = obs_grp["proprio"][:]
                elif has_split_keys:
                    proprio_data = np.concatenate([
                        obs_grp["robot0_eef_pos"][:],
                        obs_grp["robot0_eef_quat"][:],
                        obs_grp["robot0_gripper_qpos"][:],
                    ], axis=-1)
                else:
                    proprio_data = None

            demo_embeddings = []
            for i in range(0, len(actions) - args.chunk_length, args.execution_horizon):
                if use_state_fallback:
                    raw_state = state_data[i]
                    raw_img = build_synthetic_image_from_state(raw_state)
                    img_tensor = torch.tensor(raw_img).permute(2, 0, 1).unsqueeze(0).to(device).float()
                    state_feed = {args.camera: img_tensor}

                    if args.proprio is not None:
                        raw_proprio = np.asarray(raw_state[: args.proprio_dim], dtype=np.float32)
                        if raw_proprio.size < args.proprio_dim:
                            raw_proprio = np.pad(raw_proprio, (0, args.proprio_dim - raw_proprio.size), mode="constant")
                        state_feed["proprio"] = torch.tensor(raw_proprio).unsqueeze(0).to(device).float()
                else:
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

                action_feed = torch.tensor(actions[i:i + args.chunk_length]).unsqueeze(0).to(device).float()

                with torch.no_grad():
                    pred_embedding, _ = model(state_feed, action_feed)
                    demo_embeddings.append(pred_embedding.flatten(start_dim=1).cpu().numpy())

            if demo_embeddings:
                rollout_by_demo.append(np.concatenate(demo_embeddings, axis=0))

    if not rollout_by_demo:
        raise ValueError(f"No valid rollout steps were extracted from {path}.")

    return rollout_by_demo


def plot_rollout_group(ax, rollout_by_demo, coords, start_idx, cmap, line_color, label):
    current_idx = start_idx
    for demo_np in rollout_by_demo:
        length = len(demo_np)
        if length < 2:
            demo_coords = coords[current_idx : current_idx + length]
            current_idx += length
            if demo_coords.size:
                ax.scatter(
                    demo_coords[:, 0],
                    demo_coords[:, 1],
                    c=cmap([0.0])[0],
                    alpha=0.8,
                    s=28,
                    marker="o",
                    zorder=3,
                )
            continue

        demo_coords = coords[current_idx : current_idx + length]
        current_idx += length

        ax.plot(
            demo_coords[:, 0],
            demo_coords[:, 1],
            color=line_color,
            alpha=0.18,
            linewidth=1.2,
            zorder=1,
        )

        time_steps = np.linspace(0.0, 1.0, num=demo_coords.shape[0])
        colors = cmap(time_steps)
        ax.scatter(
            demo_coords[:, 0],
            demo_coords[:, 1],
            c=colors,
            alpha=0.75,
            s=18,
            marker="o",
            zorder=3,
        )
        ax.scatter(
            demo_coords[-1, 0],
            demo_coords[-1, 1],
            facecolors="none",
            edgecolors=cmap(1.0),
            alpha=0.9,
            s=42,
            linewidths=1.2,
            zorder=4,
        )

    return current_idx


def main():
    parser = argparse.ArgumentParser(description="Offline Latent Space Analysis for DynaGuide")

    # Core File Paths
    parser.add_argument("--dataset_path", type=str, default=None, help="Deprecated alias for --guided_rollout_path")
    parser.add_argument("--guided_rollout_path", type=str, default=None, help="Path to the DynaGuide-guided rollout database HDF5 file")
    parser.add_argument("--unguided_rollout_path", type=str, default=None, help="Path to the unguided baseline rollout database HDF5 file")
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

    guided_rollout_path = args.guided_rollout_path or args.dataset_path
    if not guided_rollout_path:
        parser.error("Please provide --guided_rollout_path (or the legacy --dataset_path).")

    print(">>> Initializing Dynamics Model...")
    model = FinalStatePredictionDino(
        args.action_dim,
        args.chunk_length,
        cameras=[args.camera],
        reconstruction=True,
        proprio=args.proprio,
        proprio_dim=args.proprio_dim,
    )
    model.load_state_dict(torch.load(args.dynamics_model, map_location=device))
    model.to(device).eval()

    print(">>> Encoding Guidance Target Manifest Clusters...")
    good_dataset = MultiviewDataset(
        args.good_states,
        action_chunk_length=args.chunk_length,
        cameras=[args.camera],
        padding=True,
        pad_mode="repeat",
        proprio=args.proprio,
    )
    _, good_embeddings, _ = calculate_classifier_guidance(
        model,
        good_dataset=good_dataset,
        main_camera=args.camera,
        scale=args.scale,
    )
    good_np = good_embeddings.detach().cpu().numpy()

    print(f">>> Loading Guided Rollouts from {guided_rollout_path}")
    guided_rollouts = load_rollout_embeddings(guided_rollout_path, model, args, device)
    guided_flat = np.concatenate(guided_rollouts, axis=0)

    embedding_groups = [("Expert Guidance Targets", good_np, None)]
    if args.unguided_rollout_path:
        print(f">>> Loading Unguided Rollouts from {args.unguided_rollout_path}")
        unguided_rollouts = load_rollout_embeddings(args.unguided_rollout_path, model, args, device)
        unguided_flat = np.concatenate(unguided_rollouts, axis=0)
        embedding_groups.append(("DynaGuide Guided Rollouts", guided_flat, guided_rollouts))
        embedding_groups.append(("Unguided Base Policy Rollouts", unguided_flat, unguided_rollouts))
    else:
        embedding_groups.append(("DynaGuide Guided Rollouts", guided_flat, guided_rollouts))

    embedding_arrays = []
    rollout_groups = []
    for _, emb_array, rollout_group in embedding_groups:
        embedding_arrays.append(emb_array)
        rollout_groups.append(rollout_group)

    print(f">>> Computing {args.method.upper()} Manifold Reductions...")
    all_embeddings = np.concatenate(embedding_arrays, axis=0)

    if args.method == "pca":
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(
            n_components=2,
            perplexity=args.perplexity,
            max_iter=1000,
            random_state=42,
            init="pca",
        )

    coords = reducer.fit_transform(all_embeddings)

    print(">>> Rendering Guidance Ablation Manifold...")
    fig, ax = plt.subplots(figsize=(10, 10))

    target_coords = coords[: len(good_np)]
    ax.scatter(
        target_coords[:, 0],
        target_coords[:, 1],
        c="green",
        alpha=0.95,
        s=130,
        marker="X",
        label="Expert Guidance Targets",
        zorder=3,
    )

    guided_cmap = LinearSegmentedColormap.from_list("guided", ["#2f6bff", "#7b2cbf"])
    unguided_cmap = LinearSegmentedColormap.from_list("unguided", ["#ff8c00", "#fdd835"])

    start_idx = len(good_np)
    guided_start_idx = start_idx
    guided_rollout_group = rollout_groups[1] if len(rollout_groups) > 1 else rollout_groups[0]
    if guided_rollout_group is not None:
        plot_rollout_group(
            ax,
            guided_rollout_group,
            coords,
            guided_start_idx,
            guided_cmap,
            line_color="#4c6ef5",
            label="DynaGuide Guided Rollouts",
        )
        start_idx += sum(len(demo) for demo in guided_rollout_group)

    if args.unguided_rollout_path:
        unguided_rollout_group = rollout_groups[2] if len(rollout_groups) > 2 else None
        if unguided_rollout_group is not None:
            plot_rollout_group(
                ax,
                unguided_rollout_group,
                coords,
                start_idx,
                unguided_cmap,
                line_color="#ff8c00",
                label="Unguided Base Policy Rollouts",
            )

    ax.set_title(
        f"Guidance Ablation Manifold ({args.method.upper()})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, linestyle=":", alpha=0.6)

    legend_elements = [
        Line2D([0], [0], marker="X", color="w", markerfacecolor="green", markersize=10, label="Expert Guidance Targets"),
        Line2D([0], [0], color="#4c6ef5", linewidth=2, label="DynaGuide Guided Rollouts"),
        Line2D([0], [0], color="#ff8c00", linewidth=2, label="Unguided Base Policy Rollouts"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(args.output_path, dpi=150)
    plt.close(fig)
    print(f">>> Execution finished! Plot successfully saved to: {args.output_path}")


if __name__ == "__main__":
    main()