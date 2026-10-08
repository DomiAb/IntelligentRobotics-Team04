"""Extract scalar curves from a TensorBoard log directory and plot training metrics.

This script currently reads a hard-coded LOGDIR and writes training_curves.csv plus
an overfitting_vs_rollout.png plot. Edit LOGDIR before running.
"""

import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOGDIR = "/home/smlr_g2/smlr_g2_SS_26_Niclas/ss26_g2_smlr/robomimic/robomimic/results/base_policy_square_d0_final/base_policy_square_d0_final/20260622221939/logs/tb"  # ggf. anpassen

ea = EventAccumulator(LOGDIR, size_guidance={"scalars": 0})
ea.Reload()

TAGS = {
    "train_loss":   "Train/Loss",
    "val_loss":     "Valid/Loss",
    "success_rate": "Rollout/Success_Rate/Square_D0-mean",
}

merged = None
for col, tag in TAGS.items():
    events = ea.Scalars(tag)
    df = pd.DataFrame({"step": [e.step for e in events], col: [e.value for e in events]})
    merged = df if merged is None else merged.merge(df, on="step", how="outer")

merged = merged.sort_values("step").reset_index(drop=True)
merged.to_csv("training_curves.csv", index=False)

# Plot
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(merged["step"], merged["train_loss"], label="Train Loss", color="tab:blue")
ax1.plot(merged["step"], merged["val_loss"], label="Val Loss", color="tab:orange")
ax1.set_xlabel("Epoche"); ax1.set_ylabel("Loss"); ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
sr = merged.dropna(subset=["success_rate"])
ax2.plot(sr["step"], sr["success_rate"], label="Success Rate",
         color="tab:green", linestyle="--", marker="o", markersize=3)
ax2.set_ylabel("Success Rate"); ax2.set_ylim(0, 1)

l1, lab1 = ax1.get_legend_handles_labels()
l2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, lab1 + lab2, loc="center right")
plt.title("Square_D0 — Train/Val Loss & Success Rate")
plt.tight_layout()
plt.savefig("overfitting_vs_rollout.png", dpi=150)
plt.show()
