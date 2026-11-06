"""
Initial-state pose distribution analysis for reset-pose JSON files
(as produced by helper_scripts/create_reset_poses.py, e.g.
dataset/dual_square/initial_states.json).

Reproduces the nut-pose plot (position + rotation + spawn-density heatmap)
from analyze_dataset_distribution.py, but works directly off a JSON file of
raw simulator states rather than an HDF5 dataset of demos. Since dual-nut
initial states contain two nuts (gold SquareNut + red RedSquareNut), one
plot is produced per nut.

Usage:
    python helper_scripts/analyze_initial_state_distribution.py \
        dataset/dual_square/initial_states.json \
        --env-meta-source dataset/dual_square/gold_nut_on_peg.hdf5
"""
import json
import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
from scipy import stats
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
try:
    import mimicgen
except ImportError:
    pass

# Resolve the absolute path of the project root directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "helper_scripts" else SCRIPT_DIR

if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

# Handle local repository overrides cleanly
for local_repo in ["robomimic", "robosuite", "mimicgen"]:
	repo_path = os.path.join(PROJECT_ROOT, local_repo)
	if os.path.exists(repo_path) and repo_path not in sys.path:
		sys.path.insert(0, repo_path)


NUT_STYLE = {
	"SquareNut":    dict(label="Gold SquareNut",    color="#B8860B", peg_attr="peg1_body_id", peg_label="Peg1 (Goal)"),
	"RedSquareNut": dict(label="Red RedSquareNut",  color="#C44E52", peg_attr="peg2_body_id", peg_label="Peg2 (Goal)"),
}


def _unwrap_env(env):
	rs_env = env
	while hasattr(rs_env, "env"):
		rs_env = rs_env.env
	return rs_env


def load_states(json_path):
	with open(json_path, "r") as f:
		payload = json.load(f)
	return [np.asarray(s, dtype=np.float64) for s in payload["robot_states"]]


def build_env(env_meta_source):
	try:
		dummy_spec = dict(
			obs=dict(
				low_dim=['robot0_eef_pos'],
				rgb=[],
			),
		)
		ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)
	except Exception:
		pass

	print(f"[Env] Loading environment metadata from: {env_meta_source}")
	env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=env_meta_source)
	print(f"[Env] Creating headless environment ({env_meta['env_name']})...")
	env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=False)
	return env


def extract_nut_poses(env, states):
	"""Resets `env` to every state and reads out each nut's pose (pos + quat_xyzw)."""
	rs_env = _unwrap_env(env)
	nut_names = [nut.name for nut in rs_env.nuts]
	print(f"[Extraction] Nuts found in environment: {nut_names}")

	poses = {name: {"pos": [], "quat": []} for name in nut_names}
	for state in states:
		env.reset_to({"states": state})
		for name in nut_names:
			pos = rs_env.sim.data.body_xpos[rs_env.obj_body_id[name]].copy()
			quat_wxyz = rs_env.sim.data.body_xquat[rs_env.obj_body_id[name]].copy()
			quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
			poses[name]["pos"].append(pos)
			poses[name]["quat"].append(quat_xyzw)

	for name in nut_names:
		poses[name]["pos"] = np.stack(poses[name]["pos"])    # (N, 3)
		poses[name]["quat"] = np.stack(poses[name]["quat"])  # (N, 4) xyzw

	robot_base_pos = rs_env.sim.data.body_xpos[
		rs_env.sim.model.body_name2id(rs_env.robots[0].robot_model.naming_prefix + "base")
	].copy()

	peg_pos = {}
	for name in nut_names:
		peg_attr = NUT_STYLE.get(name, {}).get("peg_attr", "peg1_body_id")
		if hasattr(rs_env, peg_attr):
			peg_pos[name] = rs_env.sim.data.body_xpos[getattr(rs_env, peg_attr)].copy()
		else:
			peg_pos[name] = rs_env.sim.data.body_xpos[rs_env.peg1_body_id].copy()

	return poses, robot_base_pos, peg_pos


def print_stats(nut_name, pos_all, euler_all, n):
	print(f"\n=== {nut_name} (N={n} initial states) ===")
	print(f"\nPosition (world frame):")
	print(f"  {'Axis':<6} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
	print("  " + "-" * 38)
	for i, axis in enumerate(["x", "y", "z"]):
		col = pos_all[:, i]
		print(f"  {axis:<6} {col.mean():>8.4f} {col.std():>8.4f} {col.min():>8.4f} {col.max():>8.4f}")

	print(f"\nRotation — Euler XYZ (degrees):")
	print(f"  {'Axis':<6} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
	print("  " + "-" * 38)
	for i, axis in enumerate(["roll(x)", "pitch(y)", "yaw(z)"]):
		col = euler_all[:, i]
		print(f"  {axis:<8} {col.mean():>8.2f} {col.std():>8.2f} {col.min():>8.2f} {col.max():>8.2f}")


def plot_nut_pose(nut_name, pos_all, euler_all, robot_base_pos, peg_pos, out_path, source_label, color):
	yaw_rad = np.deg2rad(euler_all[:, 2])

	fig2 = plt.figure(figsize=(18, 15))

	# ── 1. Position Plots (Slots 1, 2, 3, 4) ──────────────────────────────

	# Slot 1: XY scatter coloured by yaw
	ax_xy = fig2.add_subplot(3, 3, 1)
	sc = ax_xy.scatter(
		pos_all[:, 0], pos_all[:, 1],
		c=euler_all[:, 2], cmap="hsv", vmin=-180, vmax=180,
		alpha=0.7, s=20, edgecolors="none",
	)
	plt.colorbar(sc, ax=ax_xy, label="yaw (°)", shrink=0.8)
	arrow_len = max(pos_all[:, :2].ptp(axis=0)) * 0.06 if pos_all[:, :2].ptp(axis=0).max() > 0 else 0.01
	for px, py, yr in zip(pos_all[:, 0], pos_all[:, 1], yaw_rad):
		ax_xy.annotate(
			"", xy=(px + arrow_len * np.cos(yr), py + arrow_len * np.sin(yr)),
			xytext=(px, py),
			arrowprops=dict(arrowstyle="->", color="k", lw=0.5, alpha=0.5),
		)
	ax_xy.set_xlabel("x (m)")
	ax_xy.set_ylabel("y (m)")
	ax_xy.set_title("Spawn XY + facing direction\n(arrow = nut front, colour = yaw)")
	ax_xy.set_aspect("equal")
	ax_xy.grid(True, alpha=0.3)

	# Slots 2, 3, 4: X, Y, Z histograms
	position_data = zip(
		[pos_all[:, 0], pos_all[:, 1], pos_all[:, 2]],
		["x", "y", "z"],
		["#4C72B0", "#DD8452", "#55A868"]
	)
	for subplot_idx, (col, label, pcolor) in zip([2, 3, 4], position_data):
		ax = fig2.add_subplot(3, 3, subplot_idx)
		ax.hist(col, bins=30, color=pcolor, edgecolor="white", density=True)
		if col.std() > 1e-8:
			xs = np.linspace(col.min(), col.max(), 200)
			ax.plot(xs, stats.gaussian_kde(col)(xs), "k-", linewidth=1.5)
		ax.set_title(f"Nut position — {label}")
		ax.set_xlabel("meters")

	# ── 2. Rotation Plots (Slots 5, 6, 7, 8) ──────────────────────────────

	# Slot 7: Polar yaw histogram
	ax_polar = fig2.add_subplot(3, 3, 7, projection="polar")
	yaw_rad_plot = (yaw_rad - np.pi / 2) % (2 * np.pi)
	weights = np.ones_like(yaw_rad_plot) * 100.0 / len(yaw_rad_plot)
	ax_polar.hist(yaw_rad_plot, bins=36, weights=weights, color="#C44E52", edgecolor="white")
	ax_polar.set_theta_zero_location("E")   # 0° = +x axis
	ax_polar.set_theta_direction(1)          # CCW = positive yaw
	ax_polar.set_xticks(np.deg2rad([0, 90, 180, 270]))
	ax_polar.set_xticklabels(["Right\n(90°)", "Back\n(Robot, 180°)", "Left\n(270°)", "Front\n(Peg, 0°)"])

	ax_polar.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
	ax_polar.set_title("Nut spawn orientation", pad=15)

	r_lim = ax_polar.get_ylim()
	r_marker = r_lim[1] * 1.05 if r_lim[1] > 0 else 1.05
	ax_polar.plot(np.pi/2, r_marker, marker='s', color='grey', markersize=10, clip_on=False)
	ax_polar.plot(3*np.pi/2, r_marker, marker='o', color='green', markersize=10, clip_on=False)

	# Slots 5, 6, 8: Roll, Pitch, and Linear Yaw histograms
	rotation_data = zip(
		[euler_all[:, 0], euler_all[:, 1], euler_all[:, 2]],
		["roll (x)", "pitch (y)", "yaw (z)"],
		["#8172B2", "#C44E52", "#937860"]
	)
	for subplot_idx, (col, label, rcolor) in zip([5, 6, 8], rotation_data):
		ax = fig2.add_subplot(3, 3, subplot_idx)
		ax.hist(col, bins=30, color=rcolor, edgecolor="white", density=True)
		if col.std() > 1e-8:
			xs = np.linspace(col.min(), col.max(), 200)
			ax.plot(xs, stats.gaussian_kde(col)(xs), "k-", linewidth=1.5)
		ax.set_title(f"Nut rotation — {label}")
		ax.set_xlabel("degrees")

		if "yaw" in label:
			ax.axvline(45, color="gray", linestyle="--", alpha=0.5)
			ax.axvline(135, color="gray", linestyle="--", alpha=0.5)
			ax.axvline(-45, color="gray", linestyle="--", alpha=0.5)
			ax.axvline(-135, color="gray", linestyle="--", alpha=0.5)
			ylim = ax.get_ylim()
			y_pos = ylim[1] * 0.9 if ylim[1] > 0 else 0.9
			ax.text(0, y_pos, "Front", color="dimgray", ha="center", va="top", fontsize=8)
			ax.text(90, y_pos, "Right", color="dimgray", ha="center", va="top", fontsize=8)
			ax.text(-90, y_pos, "Left", color="dimgray", ha="center", va="top", fontsize=8)
			ax.text(157.5, y_pos, "Back", color="dimgray", ha="center", va="top", fontsize=8)
			ax.text(-157.5, y_pos, "Back", color="dimgray", ha="center", va="top", fontsize=8)

	# ── 3. Heatmap Plot (Slot 9) ──────────────────────────────────────────

	# Slot 9: 2-D spawn density heatmap
	ax_heat = fig2.add_subplot(3, 3, 9)
	x_data, y_data = pos_all[:, 0], pos_all[:, 1]

	x_min = min(x_data.min(), robot_base_pos[0] - 0.1, peg_pos[0] - 0.05) - 0.02
	x_max = max(x_data.max(), robot_base_pos[0] + 0.1, peg_pos[0] + 0.05) + 0.02
	y_min = min(y_data.min(), robot_base_pos[1] - 0.15, peg_pos[1] - 0.05) - 0.02
	y_max = max(y_data.max(), robot_base_pos[1] + 0.15, peg_pos[1] + 0.05) + 0.02

	h2d, xedges, yedges = np.histogram2d(
		x_data, y_data, bins=60,
		range=[[x_min, x_max], [y_min, y_max]],
	)
	h2d_smooth = gaussian_filter(h2d.astype(float), sigma=1.5)
	mesh = ax_heat.pcolormesh(xedges, yedges, h2d_smooth.T, cmap="YlOrRd")
	plt.colorbar(mesh, ax=ax_heat, label="# states (smoothed)")

	rw, rh = 0.15, 0.2
	rect = patches.Rectangle(
		(robot_base_pos[0] - rw/2, robot_base_pos[1] - rh/2),
		rw, rh,
		linewidth=1.5, edgecolor='dimgray', facecolor='darkgray', alpha=0.8,
		label="Robot Base"
	)
	ax_heat.add_patch(rect)
	ax_heat.text(
		robot_base_pos[0], robot_base_pos[1], "Robot\nBase",
		color="white", ha="center", va="center", fontsize=8, weight="bold"
	)

	ax_heat.plot(peg_pos[0], peg_pos[1], marker='*', color='green', markersize=8, label="Peg (Goal)")
	ax_heat.text(peg_pos[0], peg_pos[1] + 0.02, "Peg", color="green", ha="center", va="bottom", fontsize=8, weight="bold")

	ax_heat.set_xlabel("x (m)")
	ax_heat.set_ylabel("y (m)")
	ax_heat.set_title("Spawn density heatmap\n(top-down, Gaussian-smoothed)")
	ax_heat.set_aspect("equal")
	ax_heat.grid(True, alpha=0.25, color="gray", linestyle="--")

	fig2.suptitle(f"{nut_name} Initial Pose — {source_label}", fontsize=13)
	fig2.tight_layout()
	fig2.savefig(out_path, dpi=120)
	plt.close(fig2)
	print(f"Nut pose plot saved to: {out_path}")


def analyze(json_path, env_meta_source, save_plots=True):
	print(f"\n{'='*50}")
	print(f"Initial states: {json_path}")
	print(f"{'='*50}")

	states = load_states(json_path)
	print(f"Total initial states: {len(states)}")

	env = build_env(env_meta_source)
	poses, robot_base_pos, peg_pos = extract_nut_poses(env, states)

	base = json_path.replace(".json", "")
	source_label = os.path.basename(json_path)

	for nut_name, nut_data in poses.items():
		pos_all = nut_data["pos"]
		euler_all = Rotation.from_quat(nut_data["quat"]).as_euler("xyz", degrees=True)

		print_stats(nut_name, pos_all, euler_all, len(states))

		if save_plots:
			style = NUT_STYLE.get(nut_name, dict(color="#4C72B0"))
			out_path = f"{base}_{nut_name}_pose.png"
			plot_nut_pose(
				nut_name, pos_all, euler_all,
				robot_base_pos, peg_pos[nut_name],
				out_path, source_label, style.get("color", "#4C72B0"),
			)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("path", type=str, help="Path to reset-poses JSON file (e.g. dataset/dual_square/initial_states.json)")
	parser.add_argument(
		"--env-meta-source", type=str, default="dataset/dual_square/gold_nut_on_peg.hdf5",
		help="HDF5 file to pull robosuite/robomimic env metadata from (must match the model used to record the states)",
	)
	parser.add_argument("--no-plots", action="store_true", help="Skip saving plots")
	args = parser.parse_args()
	analyze(args.path, args.env_meta_source, save_plots=not args.no_plots)
