"""Inspect a checkpoint file and print whether DDPM or DDIM scheduler inference is enabled.

Usage:
    python helper_scripts/check_scheduler.py path/to/checkpoint.pth

Mandatory arguments:
    path/to/checkpoint.pth    PyTorch checkpoint file to inspect.
"""

import torch
import functools

# Monkey-patch torch.load to handle PyTorch 2.6+ weight strictness safely
orig_load = torch.load
@functools.wraps(orig_load)
def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return orig_load(*args, **kwargs)
torch.load = safe_load

# Now import robomimic utilities safely
import robomimic.utils.file_utils as FileUtils
import sys

if len(sys.argv) < 2:
    print("Usage: python check_scheduler.py <path_to_checkpoint.pth>")
    sys.exit(1)

checkpoint_path = sys.argv[1]

# Extract configuration metadata from checkpoint
_, ckpt_dict = FileUtils.algo_name_from_checkpoint(ckpt_path=checkpoint_path)
config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)

# Print active scheduler parameters
print("--- Scheduler Optimization Status ---")
print(f"DDPM (Standard 100-step) Enabled: {config.algo.ddpm.enabled}")
if config.algo.ddpm.enabled:
    print(f"  Inference Timesteps: {config.algo.ddpm.num_inference_timesteps}")

print(f"DDIM (Accelerated Skip-step) Enabled: {config.algo.ddim.enabled}")
if config.algo.ddim.enabled:
    print(f"  Inference Timesteps: {config.algo.ddim.num_inference_timesteps}")
