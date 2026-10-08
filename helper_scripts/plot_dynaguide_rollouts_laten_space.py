"""
Plot per-rollout dynamics in the DynaGuide latent space.

Produces one PNG per rollout showing:
- Expert guidance targets (green X)
- The actual trajectory of the rollout in latent space (colored by time)
- The dynamics model's predicted lookahead embeddings (hollow circles)
- Optional unguided baseline trajectory for comparison

Usage example:
  python helper_scripts/plot_rollout_dynamics.py \
      --dynamics_model path/to/checkpoint.pth \
      --good_states path/to/good_examples.hdf5 \
      --guided_rollout_path path/to/guided_rollouts.hdf5 \
      --unguided_rollout_path path/to/unguided_rollouts.hdf5 \
      --output_dir outputs/latent_rollouts \
      --camera agentview_image \
      --chunk_length 16 \
      --method pca

This script adapts existing utilities in the repo and aims to be robust
to the common robomimic HDF5 layout under `/data/demo_*` with keys
`obs/<camera>` and `actions`.
"""
import os
import argparse
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# local imports
from core.dynamics_models import FinalStatePredictionDino
from core.embedder_datasets import MultiviewDataset
from core.dynaguide import calculate_classifier_guidance


def prepare_state_for_model(img_uint8, proprio_vec=None, camera_key="agentview_image", device="cuda"):
    # img_uint8: H,W,C (uint8, 0-255)
    img = torch.tensor(img_uint8).permute(2, 0, 1).unsqueeze(0).float().to(device)
    state = {camera_key: img}
    if proprio_vec is not None:
        state["proprio"] = torch.tensor(proprio_vec).unsqueeze(0).float().to(device)
    return state


def load_rollout_embeddings(path, model, args, device):
    """
    Crawl the HDF5 rollout database and compute per-timestep current and predicted embeddings.

    Returns: list of dicts for each demo: {"curr": (T, D), "pred": (T, D)} (only times with full action chunk)
    """
    rollout_list = []
    print(f">>> Crawling Rollout Database: {path}")
    with h5py.File(path, "r") as f:
        for demo_key in f["data"].keys():
            demo = f["data"][demo_key]
            # read images
            try:
                imgs = np.array(demo["obs"][args.camera])
            except Exception:
                print(f"Skipping {demo_key}: camera {args.camera} not found")
                continue

            # read actions
            try:
                actions = np.array(demo["actions"])
            except Exception:
                print(f"Skipping {demo_key}: actions not found")
                continue

            # optional proprio: try common keys and reconstruct if necessary
            proprio = None
            try:
                obs_keys = list(demo['obs'].keys())
                if args.proprio is not None and args.proprio in demo['obs']:
                    proprio = np.array(demo['obs'][args.proprio])
                else:
                    pkey = next((k for k in obs_keys if 'proprio' in k.lower()), None)
                    if pkey is not None:
                        proprio = np.array(demo['obs'][pkey])
                    else:
                        parts = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']
                        if all(p in demo['obs'] for p in parts):
                            parts_arrs = [np.array(demo['obs'][p]) for p in parts]
                            parts_arrs = [a[:, -1] if a.ndim == 3 else a for a in parts_arrs]
                            proprio = np.concatenate(parts_arrs, axis=-1)
            except Exception:
                proprio = None

            # collapse frame stacking if present
            if imgs.ndim == 5:
                imgs = imgs[:, -1]
            if proprio is not None and proprio.ndim == 3:
                proprio = proprio[:, -1]

            T = imgs.shape[0]
            seq_len = args.chunk_length
            if T < seq_len:
                # nothing to predict
                continue

            curr_list = []
            pred_list = []

            for t in range(0, T - seq_len + 1):
                img_t = imgs[t]
                action_chunk = actions[t : t + seq_len]

                pvec = proprio[t] if proprio is not None else None

                state = prepare_state_for_model(img_t, pvec, camera_key=args.camera, device=device)
                actions_tensor = torch.tensor(action_chunk).unsqueeze(0).float().to(device)  # [1, seq_len, act_dim]

                with torch.no_grad():
                    curr_emb = model.state_embedding(state).flatten(start_dim=1).cpu().numpy()
                    pred_emb = model.state_action_embedding(state, actions_tensor).flatten(start_dim=1).cpu().numpy()

                curr_list.append(curr_emb[0])
                pred_list.append(pred_emb[0])

            if len(curr_list) > 0:
                rollout_list.append({"curr": np.stack(curr_list, axis=0), "pred": np.stack(pred_list, axis=0)})

    if not rollout_list:
        raise ValueError(f"No valid rollouts extracted from {path}")

    return rollout_list


def compute_and_project(good_np, guided_rollouts, unguided_rollouts, args):
    # concatenate all embeddings for global projection
    guided_curr = np.concatenate([r["curr"] for r in guided_rollouts], axis=0) if guided_rollouts else np.zeros((0, good_np.shape[1]))
    guided_pred = np.concatenate([r["pred"] for r in guided_rollouts], axis=0) if guided_rollouts else np.zeros((0, good_np.shape[1]))

    sources = [good_np, guided_curr, guided_pred]
    if unguided_rollouts is not None:
        ung_curr = np.concatenate([r["curr"] for r in unguided_rollouts], axis=0)
        ung_pred = np.concatenate([r["pred"] for r in unguided_rollouts], axis=0)
        sources.extend([ung_curr, ung_pred])

    all_embeddings = np.concatenate(sources, axis=0)

    if args.method == "pca":
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2, perplexity=args.perplexity, init="pca", random_state=42)

    coords = reducer.fit_transform(all_embeddings)

    # compute indices
    idx = 0
    idx_good = idx; idx += len(good_np)
    idx_guided_curr = idx; idx += guided_curr.shape[0]
    idx_guided_pred = idx; idx += guided_pred.shape[0]
    idx_ung_curr = None; idx_ung_pred = None
    if unguided_rollouts is not None:
        idx_ung_curr = idx; idx += ung_curr.shape[0]
        idx_ung_pred = idx; idx += ung_pred.shape[0]

    slices = {
        "good": coords[idx_good:idx_guided_curr],
        "guided_curr": coords[idx_guided_curr:idx_guided_pred],
        "guided_pred": coords[idx_guided_pred: (idx_ung_curr if idx_ung_curr is not None else idx)],
    }
    if unguided_rollouts is not None:
        slices["ung_curr"] = coords[idx_ung_curr:idx_ung_pred]
        slices["ung_pred"] = coords[idx_ung_pred:]

    return coords, slices


def plot_per_rollout(good_coords, guided_rollouts, guided_coords_slices, unguided_rollouts, ung_coords_slices, args):
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Split guided coords back into per-demo pieces
    gc = guided_coords_slices
    guided_curr_all = gc["guided_curr"]
    guided_pred_all = gc["guided_pred"]

    ptr = 0
    ung_ptr = 0  # Track unguided slices globally outside the loop

    for i, demo in enumerate(guided_rollouts):
        L = demo["curr"].shape[0]
        c_coords = guided_curr_all[ptr : ptr + L]
        p_coords = guided_pred_all[ptr : ptr + L]
        ptr += L

        fig, ax = plt.subplots(figsize=(10, 10))

        # guidance targets
        ax.scatter(good_coords[:, 0], good_coords[:, 1], c="green", marker='X', s=110, label='Guidance Targets')

        # plot actual path
        cmap = plt.get_cmap('viridis')
        time_steps = np.linspace(0.0, 1.0, num=L)
        colors = cmap(time_steps)
        ax.plot(c_coords[:, 0], c_coords[:, 1], color='gray', alpha=0.25, linewidth=1.2)
        ax.scatter(c_coords[:, 0], c_coords[:, 1], c=colors, s=28)

        # plot predictions
        ax.scatter(p_coords[:, 0], p_coords[:, 1], facecolors='none', edgecolors='red', s=30, label='Predicted Lookahead')

        # mark start and end
        ax.scatter(c_coords[0, 0], c_coords[0, 1], c='blue', s=80, marker='o', label='Start')
        ax.scatter(c_coords[-1, 0], c_coords[-1, 1], c='black', s=80, marker='s', label='End')

        # plot unguided overlay if provided
        if unguided_rollouts is not None and i < len(unguided_rollouts):
            ung_L = unguided_rollouts[i]['curr'].shape[0]
            uc_coords = ung_coords_slices['ung_curr'][ung_ptr : ung_ptr + ung_L]
            
            ax.plot(uc_coords[:, 0], uc_coords[:, 1], color='orange', linestyle='--', alpha=0.6, linewidth=1.5)
            ax.scatter(uc_coords[:, 0], uc_coords[:, 1], c='orange', s=15, alpha=0.5, label='Unguided Path')
            
            ung_ptr += ung_L

        ax.set_title(f"Rollout {i+1}")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(True, linestyle=':', alpha=0.6)

        legend_elements = [
            Line2D([0], [0], marker='X', color='w', markerfacecolor='green', markersize=9, label='Expert Targets'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=7, label='Start'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=7, label='End'),
            Line2D([0], [0], marker='o', color='red', markerfacecolor='none', markersize=7, label='Predicted Lookahead')
        ]
        
        if unguided_rollouts is not None:
             legend_elements.append(Line2D([0], [0], color='orange', linestyle='--', linewidth=1.5, marker='o', markersize=5, label='Unguided Path'))

        ax.legend(handles=legend_elements, loc='upper right')

        out_path = os.path.join(out_dir, f"rollout_{i+1}.png")
        plt.savefig(out_path, dpi=180)
        plt.close(fig)
        print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Per-rollout latent dynamics plotting for DynaGuide")
    parser.add_argument("--dynamics_model", type=str, required=False, help="(optional) path to checkpoint for recomputing embeddings")
    parser.add_argument("--good_states", type=str, required=False, help="(optional) path to good states HDF5 for computing target embeddings")
    parser.add_argument("--good_embeddings_path", type=str, default=None, help="(optional) precomputed good embeddings (npy or hdf5) to skip recomputing")
    parser.add_argument("--guided_rollout_path", type=str, required=True)
    parser.add_argument("--unguided_rollout_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/rollout_latent")
    parser.add_argument("--method", choices=["pca", "tsne"], default="pca")
    parser.add_argument("--perplexity", type=int, default=30)
    parser.add_argument("--camera", type=str, default="agentview_image")
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--chunk_length", type=int, default=16)
    parser.add_argument("--proprio", type=str, default="proprio")
    parser.add_argument("--proprio_dim", type=int, default=9)
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--use_saved_embeddings", action='store_true', help="Prefer reading embeddings from the rollout HDF5 if present")
    parser.add_argument("--subsample", type=int, default=0, help="If >1, subsample embeddings every N steps before projection/plotting")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.camera = args.camera
    args.chunk_length = args.chunk_length
    args.proprio = args.proprio

    model = None
    good_np = None

    # If requested, try to load precomputed target embeddings directly from the guided rollout HDF5
    if args.use_saved_embeddings and args.guided_rollout_path is not None:
        try:
            with h5py.File(args.guided_rollout_path, 'r') as fh:
                if 'good_embeddings' in fh:
                    good_np = np.array(fh['good_embeddings'])
                    print(f"Loaded good_embeddings from {args.guided_rollout_path}")
                elif 'bad_embeddings' in fh and good_np is None:
                    # fallback case: user only saved bad embeddings
                    print(f"Found bad_embeddings in {args.guided_rollout_path} but no good_embeddings present")
        except Exception as e:
            print(f"Could not read global embeddings from {args.guided_rollout_path}: {e}")

    # Try loading precomputed good embeddings from explicit path if provided
    if good_np is None and args.good_embeddings_path is not None:
        p = args.good_embeddings_path
        try:
            if p.endswith('.npy'):
                good_np = np.load(p)
            else:
                with h5py.File(p, 'r') as fh:
                    if 'embeddings' in fh:
                        good_np = np.array(fh['embeddings'])
                    elif 'data' in fh and 'embeddings' in fh['data']:
                        good_np = np.array(fh['data']['embeddings'])
                    else:
                        # fallback to first dataset
                        key = list(fh.keys())[0]
                        good_np = np.array(fh[key])
        except Exception as e:
            print(f"Failed to load good embeddings from {p}: {e}")

    # If we don't have good embeddings, compute them using the model and good_states HDF5
    if good_np is None:
        if args.dynamics_model is None or args.good_states is None:
            raise RuntimeError("Either provide --good_embeddings_path, or use --use_saved_embeddings with a guided rollout HDF5 that contains root 'good_embeddings', or provide both --dynamics_model and --good_states to compute them.")
        print(">>> Loading model for computing good embeddings...")
        model = FinalStatePredictionDino(args.action_dim, args.chunk_length, cameras=[args.camera], reconstruction=True, proprio=args.proprio, proprio_dim=args.proprio_dim)
        model.load_state_dict(torch.load(args.dynamics_model, map_location=device))
        model.to(device).eval()
        print(">>> Preparing guidance targets embeddings...")
        good_dataset = MultiviewDataset(args.good_states, action_chunk_length=args.chunk_length, cameras=[args.camera], padding=True, pad_mode="repeat", proprio=args.proprio)
        _, good_embeddings, _ = calculate_classifier_guidance(model, good_dataset=good_dataset, main_camera=args.camera, scale=args.scale)
        good_np = good_embeddings.detach().cpu().numpy()

    def load_saved_rollout_embeddings(path):
        saved_rollouts = []
        with h5py.File(path, 'r') as fh:
            for demo_key in fh['data'].keys():
                grp = fh['data'][demo_key]
                if 'embeddings' in grp:
                    emb = grp['embeddings']
                    if 'current' in emb and 'predicted' in emb:
                        saved_rollouts.append({'curr': np.array(emb['current']), 'pred': np.array(emb['predicted'])})
        return saved_rollouts

    # Load or compute guided rollouts embeddings
    if args.use_saved_embeddings:
        guided_rollouts = load_saved_rollout_embeddings(args.guided_rollout_path)
        if guided_rollouts:
            print(f"Loaded {len(guided_rollouts)} saved rollout embeddings from {args.guided_rollout_path}")
        else:
            print("No saved per-demo embeddings found in guided_rollout_path.")
            if args.dynamics_model is None:
                raise RuntimeError(
                    "The guided rollout HDF5 does not contain per-demo embeddings, so --dynamics_model is required to recompute them from saved observations. "
                    "Rerun run_dynaguide.py with the updated code and --dataset_obs, or provide --dynamics_model here."
                )
            if model is None:
                model = FinalStatePredictionDino(args.action_dim, args.chunk_length, cameras=[args.camera], reconstruction=True, proprio=args.proprio, proprio_dim=args.proprio_dim)
                model.load_state_dict(torch.load(args.dynamics_model, map_location=device))
                model.to(device).eval()
            guided_rollouts = load_rollout_embeddings(args.guided_rollout_path, model, args, device)
            try:
                with h5py.File(args.guided_rollout_path, 'r+') as fh:
                    for demo_key, rollout in zip(fh['data'].keys(), guided_rollouts):
                        grp = fh['data'][demo_key]
                        if 'embeddings' not in grp:
                            emb_grp = grp.create_group('embeddings')
                            emb_grp.create_dataset('current', data=rollout['curr'].astype(np.float32), compression='gzip')
                            emb_grp.create_dataset('predicted', data=rollout['pred'].astype(np.float32), compression='gzip')
                print(f"Saved recomputed per-demo embeddings back into {args.guided_rollout_path}")
            except Exception as e:
                print(f"Warning: could not persist recomputed embeddings to HDF5: {e}")
    else:
        if model is None:
            model = FinalStatePredictionDino(args.action_dim, args.chunk_length, cameras=[args.camera], reconstruction=True, proprio=args.proprio, proprio_dim=args.proprio_dim)
            model.load_state_dict(torch.load(args.dynamics_model, map_location=device))
            model.to(device).eval()
        guided_rollouts = load_rollout_embeddings(args.guided_rollout_path, model, args, device)

    unguided_rollouts = None
    if args.unguided_rollout_path is not None:
        if args.use_saved_embeddings:
            unguided_rollouts = []
            with h5py.File(args.unguided_rollout_path, 'r') as fh:
                for demo_key in fh['data'].keys():
                    grp = fh['data'][demo_key]
                    if 'embeddings' in grp:
                        emb = grp['embeddings']
                        if 'current' in emb and 'predicted' in emb:
                            unguided_rollouts.append({'curr': np.array(emb['current']), 'pred': np.array(emb['predicted'])})
            if not unguided_rollouts:
                if model is None:
                    model = FinalStatePredictionDino(args.action_dim, args.chunk_length, cameras=[args.camera], reconstruction=True, proprio=args.proprio, proprio_dim=args.proprio_dim)
                    model.load_state_dict(torch.load(args.dynamics_model, map_location=device))
                    model.to(device).eval()
                unguided_rollouts = load_rollout_embeddings(args.unguided_rollout_path, model, args, device)
        else:
            if model is None:
                model = FinalStatePredictionDino(args.action_dim, args.chunk_length, cameras=[args.camera], reconstruction=True, proprio=args.proprio, proprio_dim=args.proprio_dim)
                model.load_state_dict(torch.load(args.dynamics_model, map_location=device))
                model.to(device).eval()
            unguided_rollouts = load_rollout_embeddings(args.unguided_rollout_path, model, args, device)

    # Optional subsampling to reduce points (applied before projection)
    if args.subsample and args.subsample > 1:
        def subsample_rollouts(rollouts, step):
            return [{'curr': r['curr'][::step], 'pred': r['pred'][::step]} for r in rollouts]

        guided_rollouts = subsample_rollouts(guided_rollouts, args.subsample)
        if unguided_rollouts is not None:
            unguided_rollouts = subsample_rollouts(unguided_rollouts, args.subsample)

    print(">>> Computing projection and plotting per-rollout images...")
    coords, slices = compute_and_project(good_np, guided_rollouts, unguided_rollouts, args)

    # build slices mapping for plotting
    guided_slices = {"guided_curr": slices['guided_curr'], "guided_pred": slices['guided_pred']}
    ung_slices = None
    if unguided_rollouts is not None:
        ung_slices = {"ung_curr": slices['ung_curr'], "ung_pred": slices['ung_pred']}

    plot_per_rollout(slices['good'], guided_rollouts, guided_slices, unguided_rollouts, ung_slices, args)


if __name__ == "__main__":
    main()
