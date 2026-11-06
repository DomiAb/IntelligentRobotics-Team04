"""
General-purpose utilities for DSRL training.

Functions
---------
set_seed           : Fix random seeds for reproducibility.
soft_update        : Polyak-average target network parameters.
hard_update        : Copy source network parameters into target exactly.
save_checkpoint    : Persist actor / critic / alpha state to disk.
ema_smooth         : Exponential moving average smoothing for scalar sequences.
plot_success_curves: Save success-rate curves as PNG and JSON.
"""

import os
import json

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")   # non-interactive backend – avoids _Stack compatibility issues
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Network parameter helpers
# ---------------------------------------------------------------------------

def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak averaging: target = tau * source + (1 - tau) * target."""
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    """Copy all parameters from source into target (exact copy)."""
    target.load_state_dict(source.state_dict())


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_checkpoint(path, actor, critic, critic_target, log_alpha, args, meta, extra):
    payload = {
        "actor":         actor.state_dict(),
        "critic":        critic.state_dict(),
        "critic_target": critic_target.state_dict(),
        "log_alpha":     float(log_alpha.detach().cpu().item()),
        "args":          vars(args),
        "meta":          meta,
        "extra":         extra,
    }
    torch.save(payload, path)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def ema_smooth(values, beta: float = 0.95):
    """Exponential moving average smoothing over a list of scalar values."""
    if len(values) == 0:
        return values
    out = []
    m = values[0]
    for v in values:
        m = beta * m + (1.0 - beta) * v
        out.append(m)
    return out


def plot_success_curves(out_dir, train_sr_hist, eval_x, eval_sr_hist,
                        smooth_beta: float = 0.95) -> None:
    """
    Save an EMA-smoothed success-rate plot (PNG) and the raw data (JSON).

    Parameters
    ----------
    out_dir       : Directory where outputs are written.
    train_sr_hist : Per-episode rolling training success rates.
    eval_x        : Episode indices at which deterministic evaluation was run.
    eval_sr_hist  : Corresponding deterministic evaluation success rates.
    smooth_beta   : EMA decay factor used for smoothing.
    """
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, "success_curve.png")
    json_path = os.path.join(out_dir, "success_curve.json")

    train_smooth = ema_smooth(train_sr_hist, beta=smooth_beta)
    eval_smooth  = ema_smooth(eval_sr_hist,  beta=smooth_beta)

    plt.figure()
    if train_smooth:
        plt.plot(range(1, len(train_smooth) + 1), train_smooth, label="train_SR(EMA)")
    if eval_smooth:
        plt.plot(eval_x, eval_smooth, label="eval_SR(EMA)")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.title("DSRL Success Rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    with open(json_path, "w") as f:
        json.dump(
            {"train_sr": train_sr_hist, "eval_x": eval_x, "eval_sr": eval_sr_hist},
            f,
            indent=2,
        )

    print(f"[PLOT] saved to {plot_path}")
    print(f"[LOG]  saved to {json_path}")
