"""Evaluate DynaGuide embeddings with PCA/TSNE visualization.

This script loads a checkpoint and dataset to compute and visualize latent embeddings
for training or validation analysis.

Mandatory flags:
    --exp_dir PATH        experiment directory with logs
    --dataset PATH        labeled dataset HDF5 path
    --checkpoint_dir PATH    directory containing model checkpoints
    --action_chunk_length INT    action chunk length used by the model
    --action_dim INT      action dimension
    --proprio_key KEY     key for proprio inputs in the dataset
    --proprio_dim INT     proprio vector dimension
    --camera KEY          active camera key to use for embeddings
"""

import matplotlib.pyplot as plt 
import argparse 
import torch 
import tqdm 
import numpy as np 

from core.dynamics_models import FinalStatePredictionDino

from core.embedder_datasets import MultiviewDataset
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import random 
import imageio
import torchvision 
import os 
import json

import PIL.Image as PILImage
from PIL import ImageDraw, ImageFont

torch.set_printoptions(sci_mode=False)


def generate_TSNE_points(embeddings, dims):
    """
    Project high-dimensional embeddings into a lower-dimensional space using t-SNE.

    Args:
        embeddings (numpy.ndarray): Input feature arrays to transform.
        dims (int): Targeted output dimension space (typically 2 or 3).

    Returns:
        numpy.ndarray: Low-dimensional projection coordinates.
    """
    tsne = TSNE(dims, verbose=1)
    tsne_proj = tsne.fit_transform(embeddings)
    return tsne_proj


def generate_PCA_points(embeddings, dims):
    """
    Performs Principal Component Analysis (PCA) to extract dominant feature components.

    Args:
        embeddings (numpy.ndarray): Source high-dimensional state representations.
        dims (int): Number of principal components to keep.

    Returns:
        tuple: (Transformed low-dimensional projection arrays, fitted PCA object instance).
    """
    pca = PCA(n_components=dims)
    pca.fit(embeddings)
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    pca_proj = pca.transform(embeddings)
    return pca_proj, pca 


def prepare(data, device = DEVICE):
    """
    Recursively moves deep learning tensors or dictionaries of tensors onto the specified hardware device.

    Args:
        data (dict or torch.Tensor): Input sample data batch.
        device (str or torch.device): Targeted hardware, defaults to the runtime-selected device.

    Returns:
        dict or torch.Tensor: Casted float32 structure allocated on the target device.
    """
    if type(data) == dict:
        return {k : v.to(device).to(torch.float32) for k, v in data.items()}
    return data.to(device).to(torch.float32)


def prepare_np(data, device = DEVICE):
    """
    Converts raw numpy arrays or matching structured dictionaries into torch floating-point tensors on device.

    Args:
        data (dict or numpy.ndarray): Numerical data arrays extracted directly from dataset files.
        device (str or torch.device): Destination compute device hardware.

    Returns:
        dict or torch.Tensor: Tensor wrappers allocated and prepared for inference.
    """
    if type(data) == dict:
        return {k : torch.tensor(v).to(device).to(torch.float32) for k, v in data.items()}
    return torch.tensor(data).to(device).to(torch.float32)


def _safe_name(text):
    """
    Sanitizes arbitrary strings into disk-safe filenames by stripping non-standard operational characters.

    Args:
        text (str or None): Targeted operational name payload.

    Returns:
        str: Cleansed output string utilizing snake_case separator conventions.
    """
    text = "default" if text is None else str(text)
    for char in ["/", " ", ":", "\\"]:
        text = text.replace(char, "_")
    return text


def _sample_demo_indices(dataset, max_demos=8):
    """
    Randomly selects a subset of demonstration index IDs available inside the dataset.

    Args:
        dataset (Dataset): Target evaluation dataset container.
        max_demos (int): Upper bound limit of validation episodes to extract.

    Returns:
        list: Sorted collection containing chosen unique episode integer pointers.
    """
    total = len(dataset.lengths_list)
    if total <= max_demos:
        return list(range(total))
    return sorted(random.sample(range(total), max_demos))


def _resample_curve(curve, num_points=40):
    """
    Normalizes time-series sequence lengths onto a fixed dimension array for comparative tracking visualization.

    Args:
        curve (list or numpy.ndarray): Source variable-length array sequence.
        num_points (int): Density of targeted target interpolation points.

    Returns:
        numpy.ndarray: Resampled standardized 1D data progression tracking curve.
    """
    if len(curve) == 1:
        return np.full(num_points, curve[0], dtype=np.float32)
    source_x = np.linspace(0.0, 1.0, num=len(curve))
    target_x = np.linspace(0.0, 1.0, num=num_points)
    return np.interp(target_x, source_x, curve)


def _save_figure(output_dir, filename):
    """
    Helper to compile a standardized target operational filepath for logging graph figures.

    Args:
        output_dir (str): Base output storage workspace directory path.
        filename (str): Name string of target asset file structure.

    Returns:
        str: Fully absolute concatenated system target destination path string.
    """
    return os.path.join(output_dir, filename)


def plot_loss_history(exp_dir, output_dir):
    """
    Parses logged json arguments or loss files to render a historical train vs validation loss curve across epochs.

    Args:
        exp_dir (str): The root training directory where history artifacts are generated.
        output_dir (str): Location where the resulting plot image file should be stored.
    """
    print("Plotting historical loss progression curves...")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.set_title("Training & Validation Loss History")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss Magnitude")
    ax.grid(True, alpha=0.25)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "historical_loss_progression.png"), dpi=300)
    plt.close()


def generate_labeled_filmstrip(model, dataset, output_dir, step, camera_key, static_sample=None):
    """
    Generates an explicitly captioned filmstrip graphic mapping predictions side-by-side with ground-truth frames.
    """
    if static_sample is not None:
        sample = static_sample
    else:
        sample = dataset.__getitem__(random.randint(0, len(dataset) - 1))
        
    state, action, last_state = prepare_np(sample[0]), prepare_np(sample[1]), prepare_np(sample[2])
    
    state = {k : torch.unsqueeze(v, dim = 0) for k, v in state.items()}
    last_state = {k : torch.unsqueeze(v, dim = 0) for k, v in last_state.items()}
    action = torch.unsqueeze(action, dim = 0)

    with torch.no_grad():
        _, reco = model(state, action)
        reco = torch.clip(reco[0], 0, 1).cpu().numpy().transpose(1, 2, 0)
        
    img_curr = (state[camera_key][0].cpu().numpy().transpose(1, 2, 0) / 255.0)
    img_true = (last_state[camera_key][0].cpu().numpy().transpose(1, 2, 0) / 255.0)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img_curr)
    axs[0].set_title("Current State (t)")
    axs[0].axis('off')

    axs[1].imshow(reco)
    axs[1].set_title(f"Predicted Future (t+{dataset.action_chunk_length})")
    axs[1].axis('off')

    axs[2].imshow(img_true)
    axs[2].set_title(f"Ground Truth Future (t+{dataset.action_chunk_length})")
    axs[2].axis('off')

    plt.tight_layout()
    target_path = os.path.join(output_dir, f"labeled_filmstrip_epoch_{step}.png")
    plt.savefig(target_path, dpi=200)
    plt.close()

    return target_path


def generate_evolution_grid(filmstrip_paths, epoch_numbers, output_dir):
    """
    Combines individual filmstrip images into a unified vertical grid layout.
    """
    print("\nStitching all filmstrips into a single vertical evolution matrix...")
    
    if not filmstrip_paths:
        print("No filmstrips found to stitch!")
        return

    sample_img = PILImage.open(filmstrip_paths[0])
    w, h = sample_img.size
    
    label_margin = 180
    grid_w = w + label_margin
    grid_h = h * len(filmstrip_paths)
    
    canvas = PILImage.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(canvas)
    
    try:
        font = ImageFont.load_default()
    except IOError:
        font = None

    for idx, (path, epoch) in enumerate(zip(filmstrip_paths, epoch_numbers)):
        img = PILImage.open(path)
        y_offset = idx * h
        
        canvas.paste(img, (label_margin, y_offset))
        
        text = f"Checkpoint\nEpoch {epoch}"
        draw.text((20, y_offset + h // 2 - 15), text, fill="black", font=font)
        
        if idx > 0:
            draw.line([(0, y_offset), (grid_w, y_offset)], fill="#d0d0d0", width=2)

    grid_path = os.path.join(output_dir, "model_training_evolution_grid.png")
    canvas.save(grid_path, dpi=(300, 300))
    print(f"Success! Combined evolution matrix saved to: {grid_path}")


def jiggle_action(model, dataset, output_dir, step, camera_key):
    """
    Evaluates action perturbation sensitivity over trajectories to verify modeling responsiveness.
    """
    print("Evaluating action perturbation sensitivity")
    demo = random.randint(0, len(dataset.lengths_list) - 1)
    start, end = dataset.get_bounds_of_demo(demo)
    var_list = list() 
    dist_list = list() 
    for i in tqdm.tqdm(range(start, end)):
        sample = dataset.get_labeled_item(i) 
        
        # Dynamische Ermittlung der tatsächlichen Action-Dimensionen vor dem Unsqueeze
        raw_action = prepare_np(sample[1])
        seq_len = raw_action.shape[0]   # z.B. 16 oder 32
        action_dim = raw_action.shape[1] # z.B. 7
        
        state, action, label = prepare_np(sample[0]), raw_action, sample[2]
        state = {k : torch.unsqueeze(v, dim = 0) for k, v in state.items()}
        action = torch.unsqueeze(action, dim = 0) 
        
        with torch.no_grad():
            batch_state = {camera_key : torch.tile(state[camera_key], (11, 1, 1, 1)), "proprio" : torch.tile(state["proprio"], (11, 1))} 
            action_mags = torch.mean(torch.abs(action), dim = 1)
            
            # Hier komplett variabel anstelle von festen Werten (11, 16, 7)
            batch_action = 2 * (torch.rand((11, seq_len, action_dim), device = action.device) - 0.5)
            batch_action = batch_action * action_mags 
            
            # Sicheres Zuweisen der Original-Action (ohne die zusätzliche Batch-Dimension)
            batch_action[0] = action[0] 
            z_hats = model.state_action_embedding(batch_state, batch_action, normalize =False).detach().cpu().numpy()
        
        base_z_hat = z_hats[0]
        other_z_hats = z_hats[1:]
        z_hat_variances = np.mean(np.std(other_z_hats, axis = 0))
        z_hat_distances = np.mean(np.abs(other_z_hats - base_z_hat))   
        var_list.append(z_hat_variances)
        dist_list.append(z_hat_distances)
        
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    steps = np.arange(len(var_list))
    axs[0].plot(steps, var_list, color="#1f77b4", linewidth=1.5)
    axs[0].set_title("Embedding variability under perturbed actions")
    axs[0].set_ylabel("Std. across perturbations")
    axs[0].grid(alpha=0.2)
    axs[1].plot(steps, dist_list, color="#ff7f0e", linewidth=1.5)
    axs[1].set_title("Embedding shift relative to the reference action")
    axs[1].set_xlabel("Timestep in sampled episode")
    axs[1].set_ylabel("Mean absolute embedding difference")
    axs[1].grid(alpha=0.2)
    plt.tight_layout()
    filename = _save_figure(output_dir, f"{step}_action_sensitivity_demo{demo}.png")
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_action_magnitude_vs_error(model, dataset, output_dir, step):
    """
    Plots the structural correlation between magnitude intensities of physical control actions and prediction errors.
    """
    print("Plotting action magnitude vs embedding error")
    sampled_demos = _sample_demo_indices(dataset, max_demos=10)
    action_magnitudes = list()
    prediction_errors = list()

    for demo in sampled_demos:
        start, end = dataset.get_bounds_of_demo(demo)
        for index in range(start, end):
            sample = dataset.__getitem__(index)
            state, action, last_state = prepare_np(sample[0]), prepare_np(sample[1]), prepare_np(sample[2])
            state = {k : torch.unsqueeze(v, dim = 0) for k, v in state.items()}
            last_state = {k : torch.unsqueeze(v, dim = 0) for k, v in last_state.items()}
            action = torch.unsqueeze(action, dim = 0)

            with torch.no_grad():
                target_embed = model.state_embedding(last_state)
                predicted_embed = model.state_action_embedding(state, action)
                error = torch.nn.functional.mse_loss(target_embed, predicted_embed).item()

            action_magnitudes.append(torch.mean(torch.abs(action)).item())
            prediction_errors.append(error)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(action_magnitudes, prediction_errors, s=10, alpha=0.25, color="#4c78a8", edgecolors="none")
    if len(action_magnitudes) > 3:
        bins = np.linspace(min(action_magnitudes), max(action_magnitudes), 12)
        bin_ids = np.digitize(action_magnitudes, bins)
        bin_x = []
        bin_y = []
        for bin_id in range(1, len(bins)):
            values = [prediction_errors[i] for i, current_bin in enumerate(bin_ids) if current_bin == bin_id]
            if values:
                center = 0.5 * (bins[bin_id - 1] + bins[bin_id])
                bin_x.append(center)
                bin_y.append(float(np.mean(values)))
        if bin_x:
            ax.plot(bin_x, bin_y, color="#f58518", linewidth=2.0, label="Binned mean")
            ax.legend(frameon=False)
    ax.set_title("Action magnitude vs. prediction error")
    ax.set_xlabel("Mean absolute action magnitude")
    ax.set_ylabel("MSE between true and predicted embeddings")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    filename = _save_figure(output_dir, f"{step}_action_mag_vs_error.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    

def replay_through_reconstruction(model, dataset, output_dir, step, camera_key):
    """
    Renders temporal visual video trajectories validating multi-step frame prediction and decoding consistency.
    """
    print("Rendering reconstruction rollouts")
    video_path = _save_figure(output_dir, f"{step}_reconstruction_rollout.mp4")
    color_output = imageio.get_writer(video_path)
    img_size = dataset.get_labeled_item(0)[0][camera_key].shape[-1]
    resizer = torchvision.transforms.Resize((img_size, img_size))

    for trial in tqdm.tqdm(range(10)):
        demo = random.randint(0, len(dataset.lengths_list) - 1)
        start, end = dataset.get_bounds_of_demo(demo)
        for i in range(start, end):
            sample = dataset.__getitem__(i)
            state, action, last_state = prepare_np(sample[0]), prepare_np(sample[1]), prepare_np(sample[2])
            state = {k : torch.unsqueeze(v, dim = 0) for k, v in state.items()}
            last_state = {k : torch.unsqueeze(v, dim = 0) for k, v in last_state.items()}
            action = torch.unsqueeze(action, dim = 0) 

            with torch.no_grad():
                pred_last, reco = model(state, action)
                reco = torch.clip(resizer(reco), 0, 1)
            combined_frame = torch.concatenate((state[camera_key][0]  / 255, reco[0], last_state[camera_key][0]  / 255), dim = 1).detach().cpu().numpy()
            combined_frame = np.transpose(combined_frame, (1, 2, 0))
            combined_frame = (combined_frame * 255).astype(np.uint8)

            color_output.append_data(combined_frame)
    color_output.close()


def plot_prediction_similarity(model, dataset, output_dir, step, camera_key, action_mod = None): 
    """
    Graphs structural alignment distance errors separating computed goal state estimations from reference outputs.
    """
    print("Plotting end-state prediction error")
    sampled_demos = _sample_demo_indices(dataset, max_demos=8)
    curves = list()

    for demo in sampled_demos:
        start, end = dataset.get_bounds_of_demo(demo)
        error_list = list()
        for j in tqdm.tqdm(range(start, end)):
            sample = dataset.__getitem__(j)
            state, action, last_state = prepare_np(sample[0]), prepare_np(sample[1]), prepare_np(sample[2])
            state = {k : torch.unsqueeze(v, dim = 0) for k, v in state.items()}
            last_state = {k : torch.unsqueeze(v, dim = 0) for k, v in last_state.items()}
            action = torch.unsqueeze(action, dim = 0) 
            if action_mod == "noise":
                action_noised = torch.randn_like(action)
                action_noised[-1] = action[-1]
                action = action_noised 
            if action_mod == "negate":
                action[0:-1] = -action[0:-1]

            with torch.no_grad():
                last_state_embed = model.state_embedding(last_state)
                predicted_last_state_embed = model.state_action_embedding(state, action)
                mse_loss = torch.nn.MSELoss()
                loss = mse_loss(last_state_embed, predicted_last_state_embed)
            error_list.append(loss.item())

        curves.append(_resample_curve(error_list))

    curves = np.asarray(curves)
    mean_curve = curves.mean(axis=0)
    std_curve = curves.std(axis=0)
    progress = np.linspace(0.0, 1.0, num=len(mean_curve))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for curve in curves:
        ax.plot(progress, curve, color="#b0b0b0", alpha=0.25, linewidth=1.0)
    ax.plot(progress, mean_curve, color="#1f77b4", linewidth=2.0, label="Mean error")
    ax.fill_between(progress, mean_curve - std_curve, mean_curve + std_curve, color="#1f77b4", alpha=0.15, label="±1 std")
    ax.set_title(f"End-state embedding error{' (' + action_mod + ')' if action_mod else ''}")
    ax.set_xlabel("Normalized episode progress")
    ax.set_ylabel("MSE between true and predicted embeddings")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    mod = _safe_name(action_mod)
    plt.savefig(_save_figure(output_dir, f"{step}_prediction_error_{mod}.png"), dpi=300)
    plt.close()


def plot_reconstruction_error(model, dataset, output_dir, step, camera_key):
    """
    Computes structural pixel matching error distributions across evaluated validation episodes.
    """
    print("Plotting reconstruction error distributions...")
    sampled_demos = _sample_demo_indices(dataset, max_demos=8)
    errors = list()

    img_size = dataset.get_labeled_item(0)[0][camera_key].shape[-1]
    resizer = torchvision.transforms.Resize((img_size, img_size))

    for demo in sampled_demos:
        start, end = dataset.get_bounds_of_demo(demo)
        for index in range(start, end):
            sample = dataset.__getitem__(index)
            state, action, _ = prepare_np(sample[0]), prepare_np(sample[1]), prepare_np(sample[2])
            state = {k : torch.unsqueeze(v, dim = 0) for k, v in state.items()}
            action = torch.unsqueeze(action, dim = 0)
            with torch.no_grad():
                _, reconstructed = model(state, action)
                reconstructed = torch.clip(resizer(reconstructed), 0, 1)
                target = state[camera_key] / 255.0
                errors.append(torch.nn.functional.mse_loss(reconstructed, target).item())
                
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(errors, bins=30, color="#4c78a8", alpha=0.85)
    ax.set_title("Reconstruction error distribution")
    ax.set_xlabel("MSE between reconstructed and input image")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(_save_figure(output_dir, f"{step}_reconstruction_error_hist.png"), dpi=300)
    plt.close()


def plot_history_curve(epochs, train_values, val_values, title, y_label, filename, output_dir):
    """
    Generates and saves a clean evaluation curve comparing Train vs Val across epochs.
    """
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, train_values, label="Train", color="#1f77b4", linewidth=2, marker='o')
    plt.plot(epochs, val_values, label="Validation", color="#ff7f0e", linewidth=2, marker='s')
    
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel(y_label, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()


def calculate_metrics_for_checkpoint(model, dataset, camera_key, num_samples=100):
    """
    Computes the average Embedding MSE and Reconstruction Loss for a given model state.
    """
    mse_loss_fn = torch.nn.MSELoss()
    embedding_errors = []
    reco_errors = []
    
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    img_size = dataset.get_labeled_item(0)[0][camera_key].shape[-1]
    resizer = torchvision.transforms.Resize((img_size, img_size))

    for idx in indices:
        sample = dataset.__getitem__(idx)
        state, action, last_state = prepare_np(sample[0]), prepare_np(sample[1]), prepare_np(sample[2])
        state = {k : torch.unsqueeze(v, dim = 0) for k, v in state.items()}
        last_state = {k : torch.unsqueeze(v, dim = 0) for k, v in last_state.items()}
        action = torch.unsqueeze(action, dim = 0)

        with torch.no_grad():
            target_embed = model.state_embedding(last_state)
            predicted_embed = model.state_action_embedding(state, action)
            emb_mse = mse_loss_fn(target_embed, predicted_embed).item()
            embedding_errors.append(emb_mse)
            
            _, reconstructed = model(state, action)
            reconstructed = torch.clip(resizer(reconstructed), 0, 1)
            target_img = state[camera_key] / 255.0
            rec_mse = mse_loss_fn(reconstructed, target_img).item()
            reco_errors.append(rec_mse)

    return np.mean(embedding_errors), np.mean(reco_errors)


def main(args):
    """
    Main execution loop.
    """
    cameras = [args.camera]
    
    val_dataset = MultiviewDataset(
        args.dataset, action_chunk_length=args.action_chunk_length, cameras=cameras, padding=True, proprio=args.proprio_key
    )
    train_path = args.dataset.replace("valid.hdf5", "train.hdf5")
    if not os.path.exists(train_path):
        train_path = args.dataset 
    train_dataset = MultiviewDataset(
        train_path, action_chunk_length=args.action_chunk_length, cameras=cameras, padding=True, proprio=args.proprio_key
    )

    model = FinalStatePredictionDino(
        args.action_dim, args.action_chunk_length, cameras=cameras, reconstruction=True, proprio=args.proprio_key, proprio_dim=args.proprio_dim
    )
    model.to("cuda")

    checkpoint_dir = args.checkpoint_dir
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    
    def extract_epoch(filename):
        nums = [int(s) for s in filename.replace(".", "_").split("_") if s.isdigit()]
        return nums[0] if nums else 0
    checkpoint_files.sort(key=extract_epoch)

    log_dir = os.path.join(args.exp_dir, "final_eval_evolution")
    os.makedirs(log_dir, exist_ok=True)

    fixed_eval_sample = val_dataset.__getitem__(random.randint(0, len(val_dataset) - 1))

    epochs = []
    train_mses, val_mses = [], []
    train_recos, val_recos = [], []
    generated_strips = []
    epoch_tracking = []

    for ckpt_file in checkpoint_files:
        ckpt_path = os.path.join(checkpoint_dir, ckpt_file)
        epoch = extract_epoch(ckpt_file)
        epochs.append(epoch)
        epoch_tracking.append(epoch)
        
        print(f"\nProcessing Epoch {epoch}...")
        model.load_state_dict(torch.load(ckpt_path))
        model.eval()

        t_mse, t_reco = calculate_metrics_for_checkpoint(model, train_dataset, args.camera)
        v_mse, v_reco = calculate_metrics_for_checkpoint(model, val_dataset, args.camera)
        
        train_mses.append(t_mse)
        val_mses.append(v_mse)
        train_recos.append(t_reco)
        val_recos.append(v_reco)

        strip_path = generate_labeled_filmstrip(model, val_dataset, log_dir, str(epoch), args.camera, static_sample=fixed_eval_sample)
        replay_through_reconstruction(model, val_dataset, log_dir, str(epoch), args.camera)
        generated_strips.append(strip_path)

        if ckpt_file == checkpoint_files[-1]:
            print("Generating final snapshot analytical plots...")
            plot_prediction_similarity(model, val_dataset, log_dir, str(epoch), args.camera, action_mod="negate")
            plot_prediction_similarity(model, val_dataset, log_dir, str(epoch), args.camera)
            plot_action_magnitude_vs_error(model, val_dataset, log_dir, epoch)
            jiggle_action(model, val_dataset, log_dir, str(epoch), args.camera)

    plot_history_curve(
        epochs, train_mses, val_mses, 
        title="Embedding Target Error over Epochs", 
        y_label="Embedding MSE", 
        filename="history_embedding_mse.png", 
        output_dir=log_dir
    )
    
    plot_history_curve(
        epochs, train_recos, val_recos, 
        title="Reconstruction Loss over Epochs", 
        y_label="Image Reco MSE", 
        filename="history_reconstruction_mse.png", 
        output_dir=log_dir
    )

    generate_evolution_grid(generated_strips, epoch_tracking, log_dir)
    
    print(f"\n Evaluation completely finished! All assets generated in: {log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True, help="Directory path of the experiment logs.")
    parser.add_argument("--dataset", type=str, required=True, help="Single labeled dataset hdf5 dataset filepath.")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory containing all .pth checkpoints.")
    parser.add_argument("--action_chunk_length", type=int, default=16, help="The action length chunk setup configuration.")
    parser.add_argument("--action_dim", type=int, default=7, help="Dimension of control command operations vectors.")
    parser.add_argument("--proprio_key", type=str, default="proprio", help="Lookup token key tracking low-dim physical states data.")
    parser.add_argument("--proprio_dim", type=int, default=9, help="Total elements size width dimension of proprioception array.")
    parser.add_argument("--camera", type=str, default="agentview_image", help="Target active camera dataset string key identifier.")
    
    args = parser.parse_args()
    main(args)
