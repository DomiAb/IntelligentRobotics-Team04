"""
Train the DynaGuide dynamics model from robomimic-style HDF5 datasets.

This script trains a final-state prediction model that takes a state observation
sequence and an action chunk, then predicts a latent embedding and reconstructs
the final camera frame. It is designed for DynaGuide-style training with
multiview image observations and optional proprioception.

The script now uses configurable training and evaluation schedules:
- `num_epochs` controls how many epochs to train.
- `steps_per_epoch` controls how many optimization steps are run per epoch.
- `eval_every` controls how often validation is run.
- `save_every` controls how often model checkpoints are written.

Training and validation use robomimic-style HDF5 datasets with the following
structure:
- `/data/demo_*/actions`
- `/data/demo_*/obs/<camera_keys>`
- `/data/demo_*/obs/<optional_proprio_key>`

Key features:
- supports standard or noised action inputs,
- logs training and validation metrics to TensorBoard,
- periodically saves validation filmstrips for qualitative monitoring,
- saves model checkpoints to `exp_dir/checkpoints/`,
- writes the full run configuration to `exp_dir/args.json`,
- supports one or more camera observation keys,
- supports an optional proprioception key.

The script expects the proprio feature to have dimension 9, built by
concatenating:
- `robot0_eef_pos` (3),
- `robot0_eef_quat` (4),
- `robot0_gripper_qpos` (2).

Typical usage:
    python train_dynaguide.py \
        --exp_dir runs/dynaguide_square \
        --train_hdf5 dataset/square_splits/train.hdf5 \
        --test_hdf5 dataset/square_splits/valid.hdf5 \
        --cameras robot0_eye_in_hand_image \
        --action_dim 7 \
        --proprio_key proprio \
        --proprio_dim 9 \
        --action_chunk_length 16 \
        --batch_size 16 \
        --num_epochs 200 \
        --steps_per_epoch 1000 \
        --eval_every 5 \
        --save_every 20
"""

import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
# Force Full Precision (FP32) to prevent AMD NaN overflows
torch.set_default_dtype(torch.float32)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import argparse
import os
import sys
# Resolve the absolute path of the project root directory (one level up from dyna_scripts/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "dyna_scripts" else SCRIPT_DIR

# Ensure the root project folder is searched first for local modules like 'core'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Handle local repository overrides for robosuite and mimicgen cleanly
for local_repo in ["robosuite", "mimicgen"]:
    repo_path = os.path.join(PROJECT_ROOT, local_repo)
    if os.path.exists(repo_path) and not os.path.exists(os.path.join(repo_path, "__init__.py")):
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)



from PIL import Image
import numpy as np
import torch
import h5py
import tqdm 
import numpy as np
import torch.nn as nn

from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
import argparse 
import os 
from torch.utils.tensorboard import SummaryWriter
import time 

from core.dynamics_models import FinalStatePredictionDino, FinalStatePredictionDinoCLS
from core.embedder_datasets import MultiviewDataset

import torchvision 
import shutil 
import json 
import random 


def make_filmstrip(current_state, reco_states, true_states, save_dir):
    """
    Generates a visual comparison grid (filmstrip) saved as an image.
    
    Args:
        current_state (torch.Tensor): Visual observation at timestep t.
        reco_states (torch.Tensor): Reconstructed visual prediction for timestep t+1.
        true_states (torch.Tensor): Ground truth visual observation at timestep t+1.
        save_dir (str): Filepath where the output image will be saved.
    """
    imsize = current_state.shape[-1]
    resizer = torchvision.transforms.Resize((imsize, imsize))

    reco_states = resizer(reco_states) 
    reco_states = torch.clip(reco_states, 0, 1) # this makes it legal, although as the model gets better, it shouldn't need a lot of clipping 
    reco_states = reco_states.detach().cpu().numpy()
    current_state = current_state.detach().cpu().numpy()
    true_states = true_states.detach().cpu().numpy()
  
    reco_states = np.transpose(reco_states, (0, 2, 3, 1))
    true_states = np.transpose(true_states, (0, 2, 3, 1))
    current_state = np.transpose(current_state, (0, 2, 3, 1))
    reco_img = np.concatenate([reco_states[i] for i in range(reco_states.shape[0])], axis = 0) 
    real_img = np.concatenate([true_states[i] for i in range(true_states.shape[0])], axis = 0) 
    current_img = np.concatenate([current_state[i] for i in range(current_state.shape[0])], axis = 0) 
    final_img = np.concatenate((current_img, reco_img, real_img), axis = 1)
    plt.imsave(save_dir, final_img)


def get_valid_stats(model, sampler, generator, exp_dir, step, camera = "robot0_eye_in_hand_image"): 
    """
    Evaluates the model on validation data and tracks MSE and Reconstruction losses.

    Args:
        model (nn.Module): The dynamics model being trained.
        sampler (DataLoader): The validation data loader.
        generator (iterator): Iterator over the validation dataloader.
        exp_dir (str): Log directory for saving artifact filmstrips.
        step (int): Current training epoch or step index.
        camera (str): Dict key identifier for the active camera frame.

    Returns:
        tuple: (dict of averaged validation metrics, numpy array of standard deviations, 
                dummy tuple of latent parameters, updated validation generator)
    """
    
    loss_count = 0
    embedding_list = list()
    mse_loss = torch.nn.MSELoss()

    info = {"overall" : 0, "mse_loss" : 0, "reco_loss" : 0}
    resizer = None 
    for j in tqdm.tqdm(range(50)): # takes 50 points for validation 
        try:
            sample = next(generator)
        except StopIteration:
            generator = iter(sampler)
            sample = next(generator)

        state, action, last_state = prepare(sample[0]), prepare(sample[1]), prepare(sample[2]) # there may not be a negative (in that case, it give syou none)
        kld_prior_loss = 0
        with torch.no_grad():      
            z_hat_last, reco_last = model(state, action) # the image is 200x200, we resize to 224 for dino 
            embedding_list.append(z_hat_last)
            if resizer is None:
                resizer = torchvision.transforms.Resize((last_state[camera].shape[-1], last_state[camera].shape[-1]))
           
            reco_loss = mse_loss(resizer(reco_last), last_state[camera] / 255) # inputs are at 0 to 255 but outputs are 0-1
            info["reco_loss"] += reco_loss.item()

            z_last = model.state_embedding(last_state)
            mse_loss_value = mse_loss(z_last, z_hat_last)
            info["mse_loss"] += mse_loss_value.item() 

            loss =  mse_loss_value + reco_loss
           

            info["overall"] += loss.item() 

            
        
        if loss_count % 50 == 0 and step % 40 == 0:
            print("Making filmstrip!")
            make_filmstrip(state[camera] / 255, reco_last, last_state[camera] / 255, exp_dir + f"/rc_{step}_{loss_count}.png")
 
        loss_count += 1

    cat_embed = torch.concatenate(embedding_list, dim = 0)
    variances = torch.std(cat_embed, dim = 0).detach().cpu().numpy() # batch variance
    info = {k : v / loss_count for k, v in info.items()}
    print(f"Average Validation Losses: {info}")
    info["min_var"] = np.min(variances)
    info["max_var"] = np.max(variances)
    print(f"Min batch variance is: {np.min(variances)} and maximum is {np.max(variances)}")
    z_hat_mean = z_hat_logvar = 0

    return info, variances, (z_hat_mean, z_hat_logvar), generator # return variances for logging purposes 

def save_scalar_stats(writer, info_dict, epoch, mode = "train"):
    """
    Logs a dictionary of scalar values to TensorBoard under a designated mode prefix.

    Args:
        writer (SummaryWriter): PyTorch Tensorboard logging utility instance.
        info_dict (dict): Keys represent metrics names, values represent the scalar values.
        epoch (int): The global step or epoch index tracker.
        mode (str): Logging scope category, e.g., 'train' or 'valid'.
    """
    for key, value in info_dict.items():
        writer.add_scalar(f"{mode}/{key}", value, epoch)

def prepare(data, device = "cuda"):
    """
    Moves input batches or dictionaries of tensors recursively to the target device and formats them to float32.

    Args:
        data (dict, torch.Tensor, or None): The input tensor structure extracted from dataset samples.
        device (str): Destination hardware device identifier, defaults to 'cuda'.

    Returns:
        dict or torch.Tensor: Casted and relocated input structures matching original datatype format.
    """

    if data is None:
        return None # for passthroughs 
    
    if type(data) == dict:
        return {k : v.to(device).to(torch.float32) for k, v in data.items()}
    return data.to(device).to(torch.float32)

def find_latest_checkpoint(checkpoint_dir):
    """Return the path to the highest-epoch checkpoint in checkpoint_dir, or None."""
    import glob
    ckpts = glob.glob(os.path.join(checkpoint_dir, "model_epoch_*.pth"))
    if not ckpts:
        return None
    return max(ckpts, key=lambda p: int(os.path.basename(p).replace("model_epoch_", "").replace(".pth", "")))


def main(args):
    """
    Executes the training and validation loops for the vision-based robotic dynamics model.
    Configures datasets, optimizes parameters via Adam, and pushes scalar logs to Tensorboard.
    """
    from torch.utils.data import ConcatDataset

    main_camera = args.cameras[0] # arbitrarily use the first camera for visualization
    padding = True
    pad_mode = "zeros" #"zeros"  # use "repeat" for absolute control, "zeros" for delta control to make it proper

    if args.exp_dir is not None and not os.path.isdir(args.exp_dir):
        os.mkdir(args.exp_dir)

    # Define the model architecture and move it to the GPU
    model = FinalStatePredictionDino(args.action_dim,
                                     args.action_chunk_length,
                                     cameras=args.cameras,
                                     reconstruction = True,
                                     proprio = args.proprio_key,
                                     proprio_dim = args.proprio_dim)

    model.to("cuda")
    print(model.trainable_parameters())

    # Support one or more training HDF5 files (e.g. original + red-square rollouts)
    train_paths = args.train_hdf5 if isinstance(args.train_hdf5, list) else [args.train_hdf5]
    _ds_kwargs = dict(action_chunk_length=args.action_chunk_length,
                      cameras=args.cameras,
                      proprio=args.proprio_key,
                      padding=padding,
                      pad_mode=pad_mode)
    if len(train_paths) == 1:
        dataset = MultiviewDataset(train_paths[0], **_ds_kwargs)
    else:
        dataset = ConcatDataset([MultiviewDataset(p, **_ds_kwargs) for p in train_paths])
        print(f"[Dataset] Concatenated {len(train_paths)} training files → {len(dataset)} samples total:")
        for p in train_paths:
            print(f"  {p}")

    sampler = DataLoader(dataset,
                         batch_size=args.batch_size,        # number of samples per batch to load
                         shuffle=True,                      # set to True to have the data reshuffled at every epoch (default: False)
                         sampler=None,
                         batch_sampler=None,                # custom batch sampler (default: None)
                         num_workers=4,
                         collate_fn=None,
                         pin_memory=False,
                         drop_last=False,
                         timeout=0,
                         worker_init_fn=None,               # prefetch_factor=2,
                         persistent_workers=False)
    sample_generator = iter(sampler)
    
    test_paths = args.test_hdf5 if isinstance(args.test_hdf5, list) else [args.test_hdf5]
    _valid_kwargs = dict(action_chunk_length=args.action_chunk_length,
                         cameras=args.cameras,
                         proprio=args.proprio_key,
                         padding=padding,
                         pad_mode=pad_mode)
    if len(test_paths) == 1:
        valid_dataset = MultiviewDataset(test_paths[0], **_valid_kwargs)
    else:
        from torch.utils.data import ConcatDataset as _ConcatDataset
        valid_dataset = _ConcatDataset([MultiviewDataset(p, **_valid_kwargs) for p in test_paths])
        print(f"[Dataset] Concatenated {len(test_paths)} validation files → {len(valid_dataset)} samples total:")

        for p in test_paths:
            print(f"  {p}")

    valid_sampler = DataLoader(valid_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 sampler=None,
                                 batch_sampler=None,
                                 num_workers=4,
                                 collate_fn=None,
                                 pin_memory=False,
                                 drop_last=False,
                                 timeout=0,
                                 worker_init_fn=None, # prefetch_factor=2,
                                 persistent_workers=False)
    valid_generator = iter(valid_sampler)

    if args.noised:
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        noise_scheduler = DDIMScheduler(
                                 num_train_timesteps=100,
                                 beta_schedule="squaredcos_cap_v2",
                                 clip_sample=True,
                                 prediction_type="epsilon",
                                 steps_offset=0,
                                 set_alpha_to_one=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    writer = SummaryWriter(args.exp_dir)

    # --- Resume or fresh-start ---
    start_epoch = 0
    checkpoint_dir = os.path.join(args.exp_dir, "checkpoints")

    if args.resume or args.resume_checkpoint:
        ckpt_path = args.resume_checkpoint or find_latest_checkpoint(checkpoint_dir)
        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}. Cannot resume.")
        print(f"[Resume] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cuda")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
        else:
            # Legacy format: plain state dict — infer epoch from filename
            model.load_state_dict(ckpt)
            epoch_num = int(os.path.basename(ckpt_path).replace("model_epoch_", "").replace(".pth", ""))
            start_epoch = epoch_num + 1
            print(f"[Resume] Legacy checkpoint (no optimizer state). Epoch inferred from filename: {epoch_num}")
        print(f"[Resume] Continuing from epoch {start_epoch} for {args.num_epochs} more epochs.")
        resume_args_path = os.path.join(args.exp_dir, f"args_resumed_epoch{start_epoch}.json")
        with open(resume_args_path, "w") as f:
            json.dump(vars(args), f, indent=4)
        print(f"[Resume] Args saved → {resume_args_path}")
    else:
        with open(os.path.join(args.exp_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

    mse_loss = torch.nn.MSELoss()
    resizer = None

    # Main training loop — epoch index is absolute so TensorBoard steps stay continuous on resume
    for i in range(start_epoch, start_epoch + args.num_epochs):
        info = {"overall" : 0, "mse_loss" : 0, "reco_loss" : 0}
        loss_count = 0
        print(f"----------------Training Step {i}------------------")

        # Using costum steps per epoch parameter here
        for j in tqdm.tqdm(range(args.steps_per_epoch)):
            try:
                sample = next(sample_generator)
            except StopIteration:
                sample_generator = iter(sampler)
                sample = next(sample_generator)
            

            state, action, last_state = prepare(sample[0]), prepare(sample[1]), prepare(sample[2])

            if args.noised:
                # this logic will add noise to the actions to simulate the noised action inputs during inference time 
                m = torch.distributions.geometric.Geometric(0.05 * torch.ones(action.shape[0])) 
                timesteps = torch.clip(m.sample(), 0, 99).long() # this samples from a geometric distribtuion of expected value 20 
                noise = torch.randn(action.shape, device=action.device)
                noised_action = noise_scheduler.add_noise(action, noise, timesteps)

                chance_of_mask = min(0.5, i / args.num_epochs) # this ensures a ramping effect of the noise operation 
                mask = torch.rand(action.shape[0]) < chance_of_mask 
                action[mask] = noised_action[mask] 
            
            # the actual dynamics model call 
            z_hat_last, reco_last = model(state, action)
            if resizer is None: # resizing the reconstruction so they can be compared 
                resizer = torchvision.transforms.Resize((last_state[main_camera].shape[-1], last_state[main_camera].shape[-1]))
           
            # computing model loss 
            reco_loss = mse_loss(resizer(reco_last), last_state[main_camera] / 255)
            info["reco_loss"] += reco_loss.item()

            z_last = model.state_embedding(last_state)
            mse_loss_value = mse_loss(z_last, z_hat_last)
            info["mse_loss"] += mse_loss_value.item() 

            loss =  mse_loss_value + reco_loss # recall that these two losses are not connected through gradient 

            info["overall"] += loss.item() 

            optimizer.zero_grad() #gradients add up, so you must reset
            loss.backward() #backpropagation. Put a vector into the backward() to compute the jacobian product
            optimizer.step() #applies change

            loss_count += 1

        info = {k : v / loss_count for k, v in info.items()}
        save_scalar_stats(writer, info, i, "train")
        print("Average training losses: ", info) 

        if i % args.eval_every == 0:  # so we don't have to spend that much time evaluating something 
            print("--------------Evaluating-----------------")
            model.eval()
            stats, embeddings_std, (mean, logvar), valid_generator = get_valid_stats(model, valid_sampler, valid_generator, args.exp_dir, step = i,camera = main_camera)
            model.train()
            save_scalar_stats(writer, stats, i, "valid")
            writer.add_histogram("valid/embeddings_std", embeddings_std, i)
            writer.add_histogram("valid/mean", mean, i)
            writer.add_histogram("valid/logvar", logvar, i)

        # Save model checkpoints at specified intervals and at the final epoch
        if i % args.save_every == 0 or i == (start_epoch + args.num_epochs - 1):
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{i}.pth")
            torch.save({
                "epoch": i,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, checkpoint_path)
            print(f"--> Saved checkpoint to {checkpoint_path}")

        
if __name__ == "__main__":
    torch.set_printoptions(sci_mode = False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default=None, help="Path to the experiment directory.",)
    parser.add_argument("--train_hdf5", nargs='+', default=None,
                        help="One or more HDF5 files for training (multiple files are concatenated).")
    parser.add_argument("--test_hdf5", nargs='+', default=None,
                        help="One or more HDF5 files for validation (multiple files are concatenated).")
    parser.add_argument("--resume", action='store_true',
                        help="Resume training from the latest checkpoint in exp_dir/checkpoints/.")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path to a specific checkpoint to resume from (overrides --resume auto-detection).")
    parser.add_argument("--noised", action='store_true', help="enables noise augmentation for actions",)
    parser.add_argument("--cameras", nargs='+', default=["third_person"], help="List of camera names",)
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of training steps x100 to run",)
    parser.add_argument("--steps_per_epoch", type=int, default=1000, help="Number of steps to run in each epoch",)
    parser.add_argument("--action_dim", type=int, default=None, help="How large the actions are",)
    parser.add_argument("--proprio_key", type=str, default=None, help="The key to use in the states dict for proprio",)
    parser.add_argument("--proprio_dim", type=int, default=None, help="How large the proprioception is",)
    parser.add_argument("--action_chunk_length", type=int, default=16, help="How long to take an action chunk", )
    parser.add_argument("--batch_size", type=int, default=16, help="",)
    parser.add_argument("--eval_every", type=int, default=5, help="Number of epochs to wait before running validation loop.", )
    parser.add_argument("--save_every", type=int, default=20, help="Number of epochs to wait before saving a model checkpoint.", )
    
    args = parser.parse_args()

    main(args)