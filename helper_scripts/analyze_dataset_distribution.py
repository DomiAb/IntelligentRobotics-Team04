"""
Distribution analysis for robomimic HDF5 datasets.
Includes nut pose (position + rotation) analysis for square_d0_visual.
Usage: python helper_scripts/analyze_dataset_distribution.py <path_to_dataset.hdf5>
"""
import h5py
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter
import os
import sys

import matplotlib.ticker as ticker
import create_guidance_dataset as cgd
import show_hdf5 as sh

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
try:
    import mimicgen
except ImportError:
    pass

import matplotlib.patches as patches

import imageio

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




def get_orientation_cluster(yaw_deg: float) -> str:
	"""Map a yaw angle (degrees) to one of 4 insertion-direction clusters (robot-relative)."""
	y = yaw_deg % 360
	if 45 <= y < 135:
		return "Right (45-135°)"
	elif 135 <= y < 225:
		return "Back (135-225°)"
	elif 225 <= y < 315:
		return "Left (225-315°)"
	else:
		return "Front (315-45°)"


def analyze(file_path, save_plots=True, save_failed_demos=False):
    with h5py.File(file_path, "r") as f:
        data = f["data"]
        demos = cgd.sorted_demo_names(data)
        print(f"\n{'='*50}")
        print(f"Dataset: {file_path}")
        print(f"{'='*50}")
        print(f"Total demos: {len(demos)}")

        # ── Trajectory lengths ──────────────────────────────
        lengths = sh.get_trajectory_lengths(data, demos)
        print(f"\nTrajectory lengths:")
        print(f"  min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}, std={np.std(lengths):.1f}")
        print(f"  skewness={stats.skew(lengths):.3f}, kurtosis={stats.kurtosis(lengths):.3f}")

        # ── Actions ─────────────────────────────────────────
        all_actions = np.concatenate([data[d]["actions"][:] for d in demos], axis=0)
        action_dim = all_actions.shape[1]
        print(f"\nActions: {all_actions.shape} (total steps x dim)")
        print(f"{'Dim':<6} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Skew':>8} {'Kurt':>8}")
        print("-" * 58)
        for i in range(action_dim):
            col = all_actions[:, i]
            print(f"{i:<6} {col.mean():>8.3f} {col.std():>8.3f} {col.min():>8.3f} {col.max():>8.3f} "
                  f"{stats.skew(col):>8.3f} {stats.kurtosis(col):>8.3f}")

        # ── Rewards ─────────────────────────────────────────
        has_rewards = all("rewards" in data[d] for d in demos)
        if has_rewards:
            all_rewards = np.concatenate([data[d]["rewards"][:] for d in demos])
            unique, counts = np.unique(all_rewards, return_counts=True)
            print(f"\nRewards distribution:")
            for v, c in zip(unique, counts):
                print(f"  {v:.2f}: {c} ({100*c/len(all_rewards):.1f}%)")
        else:
            print(f"\nRewards distribution: Not found in dataset")

        # ── Proprioception (if available) ────────────────────
        obs_keys = list(data[demos[0]]["obs"].keys())
        print(f"\nObs keys: {obs_keys}")
        proprio_keys = [k for k in obs_keys if "state" in k or "proprio" in k or "joint" in k or "eef" in k]
        for key in proprio_keys:
            arr = np.concatenate([data[d]["obs"][key][:] for d in demos], axis=0)
            print(f"\nObs [{key}]: shape={arr.shape}")
            print(f"  mean={arr.mean(axis=0).round(3)}")
            print(f"  std={arr.std(axis=0).round(3)}")

        # ── Nut pose (square_d0_visual specific) ─────────────
        nut_pos_key = next((k for k in obs_keys if "SquareNut" in k and k.endswith("_pos")), None)
        nut_quat_key = next((k for k in obs_keys if "SquareNut" in k and k.endswith("_quat")), None)

        nut_pos_all = nut_quat_all = nut_euler_all = None
        env = None

        if nut_pos_key and nut_quat_key:
            try:
                nut_pos_list = []
                nut_quat_list = []
                for d in demos:
                    obs_grp = data[d]["obs"]
                    if nut_pos_key in obs_grp and nut_quat_key in obs_grp:
                        nut_pos_list.append(obs_grp[nut_pos_key][0])
                        nut_quat_list.append(obs_grp[nut_quat_key][0])
                if len(nut_pos_list) > 0:
                    nut_pos_all  = np.stack(nut_pos_list)  # (N, 3)
                    nut_quat_all = np.stack(nut_quat_list)  # (N, 4) xyzw
                    print(f"\nFound explicit nut keys: '{nut_pos_key}', '{nut_quat_key}'")
            except Exception as e:
                print(f"\nError reading explicit nut keys: {e}")

        if nut_pos_all is None and "object" in obs_keys:
            try:
                # In square_d0_visual the nut pose is packed into the 'object' observation.
                # Standard robosuite NutAssemblySquare layout: [pos(3), quat_xyzw(4), ...]
                obj0 = data[demos[0]]["obs"]["object"][0]
                print(f"\nNo explicit SquareNut keys found. Using 'object' key (shape per step: {obj0.shape}).")
                print(f"  First-step values: {obj0.round(4)}")
                print(f"  Assuming layout: object[0:3]=pos, object[3:7]=quat (xyzw)")
                nut_pos_list = []
                nut_quat_list = []
                for d in demos:
                    obs_grp = data[d]["obs"]
                    if "object" in obs_grp:
                        nut_pos_list.append(obs_grp["object"][0, :3])
                        nut_quat_list.append(obs_grp["object"][0, 3:7])
                if len(nut_pos_list) > 0:
                    nut_pos_all  = np.stack(nut_pos_list)  # (N, 3)
                    nut_quat_all = np.stack(nut_quat_list)  # (N, 4) xyzw
            except Exception as e:
                print(f"\nError reading packed object keys: {e}")

        if nut_pos_all is None and any("states" in data[d] for d in demos):
            try:
                print("\nNo nut pose in observations. Attempting to extract from simulator states...")
                
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
                
                print("[Nut Pose Extraction] Loading environment metadata from dataset...")
                env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=file_path)
                print("[Nut Pose Extraction] Creating headless environment...")
                env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=False)
                
                rs_env = env
                while hasattr(rs_env, 'env'):
                    rs_env = rs_env.env
                
                # Find square nut name
                square_nut = next((nut for nut in rs_env.nuts if "Square" in nut.name), None)
                if square_nut is None:
                    square_nut = rs_env.nuts[0]
                nut_name = square_nut.name
                print(f"[Nut Pose Extraction] Target nut object: {nut_name}")
                
                nut_pos_list = []
                nut_quat_list = []
                for d in demos:
                    states = data[d]["states"][:]
                    env.reset_to({"states": states[0]})
                    pos = rs_env.sim.data.body_xpos[rs_env.obj_body_id[nut_name]].copy()
                    quat_wxyz = rs_env.sim.data.body_xquat[rs_env.obj_body_id[nut_name]].copy()
                    # Convert wxyz to xyzw for scipy Rotation
                    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
                    nut_pos_list.append(pos)
                    nut_quat_list.append(quat_xyzw)
                
                if len(nut_pos_list) > 0:
                    nut_pos_all  = np.stack(nut_pos_list)  # (N, 3)
                    nut_quat_all = np.stack(nut_quat_list)  # (N, 4) xyzw
                    print(f"Successfully extracted nut poses from simulator states (N={len(demos)}).")
            except Exception as e:
                print(f"Error extracting nut pose from states: {e}")

        if nut_pos_all is None:
            print(f"\nCould not find nut pose. Available obs keys: {obs_keys}")

        if nut_pos_all is not None:
            nut_euler_all = Rotation.from_quat(nut_quat_all).as_euler("xyz", degrees=True)  # (N, 3)

            print(f"\nNut initial position (world frame, N={len(demos)} demos):")
            print(f"  {'Axis':<6} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
            print("  " + "-" * 38)
            for i, axis in enumerate(["x", "y", "z"]):
                col = nut_pos_all[:, i]
                print(f"  {axis:<6} {col.mean():>8.4f} {col.std():>8.4f} {col.min():>8.4f} {col.max():>8.4f}")

            print(f"\nNut initial rotation — Euler XYZ (degrees):")
            print(f"  {'Axis':<6} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
            print("  " + "-" * 38)
            for i, axis in enumerate(["roll(x)", "pitch(y)", "yaw(z)"]):
                col = nut_euler_all[:, i]
                print(f"  {axis:<8} {col.mean():>8.2f} {col.std():>8.2f} {col.min():>8.2f} {col.max():>8.2f}")

        # ── Plots ────────────────────────────────────────────
        if save_plots:
            base = file_path.replace(".hdf5", "").replace(".h5", "")

            # -- Plot 1: actions + trajectory lengths --
            out_actions = base + "_actions.png"
            n_cols = 4
            n_rows = (action_dim + 1 + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
            axes = axes.flatten()

            axes[0].hist(lengths, bins=30, color="steelblue", edgecolor="white")
            axes[0].set_title("Trajectory Lengths")
            axes[0].set_xlabel("steps")

            for i in range(action_dim):
                ax = axes[i + 1]
                ax.hist(all_actions[:, i], bins=60, color="coral", edgecolor="white", density=True)
                xs = np.linspace(all_actions[:, i].min(), all_actions[:, i].max(), 200)
                kde = stats.gaussian_kde(all_actions[:, i])
                ax.plot(xs, kde(xs), "k-", linewidth=1.5)
                ax.set_title(f"Action dim {i}")
                ax.set_xlabel("value")

            for j in range(action_dim + 1, len(axes)):
                axes[j].set_visible(False)

            fig.suptitle(f"Actions — {file_path.split('/')[-1]}", fontsize=13)
            fig.tight_layout()
            fig.savefig(out_actions, dpi=120)
            plt.close(fig)
            print(f"\nAction plots saved to: {out_actions}")
            
            # -- Plot 2: nut pose distribution --
            if nut_pos_all is not None:
                out_nut = base + "_nut_pose.png"
                yaw_rad = np.deg2rad(nut_euler_all[:, 2])

                # Get robot and peg positions for visualization
                try:
                    if env is not None:
                        rs_env = env
                        while hasattr(rs_env, 'env'):
                            rs_env = rs_env.env
                        robot_base_pos = rs_env.sim.data.body_xpos[rs_env.sim.model.body_name2id(rs_env.robots[0].robot_model.naming_prefix + "base")].copy()
                    else:
                        robot_base_pos = np.array([-0.385, 0.0, 0.9])
                except Exception:
                    robot_base_pos = np.array([-0.385, 0.0, 0.9])

                try:
                    if env is not None:
                        rs_env = env
                        while hasattr(rs_env, 'env'):
                            rs_env = rs_env.env
                        peg_pos = rs_env.sim.data.body_xpos[rs_env.peg1_body_id].copy()
                    else:
                        peg_pos = np.array([0.23, 0.1, 0.85])
                except Exception:
                    peg_pos = np.array([0.23, 0.1, 0.85])

                fig2 = plt.figure(figsize=(18, 15))

                # ── 1. Position Plots (Slots 1, 2, 3, 4) ──────────────────────────────

                # Slot 1: XY scatter coloured by yaw
                ax_xy = fig2.add_subplot(3, 3, 1)
                sc = ax_xy.scatter(
                    nut_pos_all[:, 0], nut_pos_all[:, 1],
                    c=nut_euler_all[:, 2], cmap="hsv", vmin=-180, vmax=180,
                    alpha=0.7, s=20, edgecolors="none",
                )
                plt.colorbar(sc, ax=ax_xy, label="yaw (°)", shrink=0.8)
                arrow_len = max(nut_pos_all[:, :2].ptp(axis=0)) * 0.06
                for px, py, yr in zip(nut_pos_all[:, 0], nut_pos_all[:, 1], yaw_rad):
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
                    [nut_pos_all[:, 0], nut_pos_all[:, 1], nut_pos_all[:, 2]],
                    ["x", "y", "z"],
                    ["#4C72B0", "#DD8452", "#55A868"]
                )
                for subplot_idx, (col, label, color) in zip([2, 3, 4], position_data):
                    ax = fig2.add_subplot(3, 3, subplot_idx)
                    ax.hist(col, bins=30, color=color, edgecolor="white", density=True)
                    if col.std() > 1e-8:
                        xs = np.linspace(col.min(), col.max(), 200)
                        ax.plot(xs, stats.gaussian_kde(col)(xs), "k-", linewidth=1.5)
                    ax.set_title(f"Nut position — {label}")
                    ax.set_xlabel("meters")

                # ── 2. Rotation Plots (Slots 5, 6, 7, 8) ──────────────────────────────

                # Slot 7: Polar yaw histogram
                ax_polar = fig2.add_subplot(3, 3, 7, projection="polar")
                yaw_rad_plot = (yaw_rad - np.pi / 2) % (2 * np.pi)
                # Calculate weights so that bin heights represent percentages
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
                    [nut_euler_all[:, 0], nut_euler_all[:, 1], nut_euler_all[:, 2]],
                    ["roll (x)", "pitch (y)", "yaw (z)"],
                    ["#8172B2", "#C44E52", "#937860"]
                )
                for subplot_idx, (col, label, color) in zip([5, 6, 8], rotation_data):
                    ax = fig2.add_subplot(3, 3, subplot_idx)
                    ax.hist(col, bins=30, color=color, edgecolor="white", density=True)
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
                x_data, y_data = nut_pos_all[:, 0], nut_pos_all[:, 1]
                
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
                plt.colorbar(mesh, ax=ax_heat, label="# demos (smoothed)")
                
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

                fig2.suptitle(f"SquareNut Initial Pose — {file_path.split('/')[-1]}", fontsize=13)
                fig2.tight_layout()
                fig2.savefig(out_nut, dpi=120)
                plt.close(fig2)
                print(f"Nut pose plots saved to: {out_nut}")

        # -- Nut orientation classification and distribution analysis --
        if env is None:
            try:                
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
                
                print("\n[Orientation Classification] Loading environment metadata from dataset...")
                env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=file_path)
                print("[Orientation Classification] Creating headless environment...")
                env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=False)
            except Exception as e:
                print(f"[Warning] Could not initialize environment for classifying nut orientation: {e}")

        if env is not None:
            print("[Orientation Classification] Categorizing demos by nut insertion direction...")
            demos_by_dir = {
                "left": [],
                "right": [],
                "back": [],
                "front": [],
                "unknown": []
            }
            
            rs_env = env
            while hasattr(rs_env, 'env'):
                rs_env = rs_env.env

            base = file_path.replace(".hdf5", "").replace(".h5", "")

            for demo_name in demos:
                demo_in = data[demo_name]
                if "states" not in demo_in:
                    demos_by_dir["unknown"].append((demo_name, None))
                    continue
                
                # Check where success / max reward is achieved
                if "rewards" in demo_in:
                    rewards = np.asarray(demo_in["rewards"][...])
                    if rewards.size > 0 and np.max(rewards) > 0:
                        step_idx = int(np.argmax(rewards))
                    else:
                        step_idx = len(rewards) - 1 if rewards.size > 0 else 0
                else:
                    step_idx = 0
                
                states = np.asarray(demo_in["states"][...])
                yaw = None
                for check_idx in range(step_idx, len(states)):
                    try:
                        env.reset_to({"states": states[check_idx]})
                        if hasattr(rs_env, '_check_success'):
                            rs_env._check_success()
                        if hasattr(rs_env, 'assembled_nut_yaw'):
                            yaw = rs_env.assembled_nut_yaw()
                            if yaw is not None:
                                break
                    except Exception:
                        pass
                
                if yaw is not None:
                    cluster = get_orientation_cluster(yaw)
                    if "Left" in cluster:
                        demos_by_dir["left"].append(demo_name)
                    elif "Right" in cluster:
                        demos_by_dir["right"].append(demo_name)
                    elif "Back" in cluster:
                        demos_by_dir["back"].append(demo_name)
                    else:
                        demos_by_dir["front"].append(demo_name)
                else:
                    demos_by_dir["unknown"].append((demo_name, yaw))
                    print(f"[Warning] Failed to find successful yaw for {demo_name}. Recorded yaw: {yaw}")
                    if(save_failed_demos):
                        # Try to save a picture and video of this failed sample
                        try:
                            image_key = next((k for k in obs_keys if "image" in k), None)
                            if image_key is not None and image_key in demo_in["obs"]:
                                frames = demo_in["obs"][image_key][...]
                                # If the frames are float/normalized, convert to uint8
                                if frames.dtype == np.float32 or frames.dtype == np.float64:
                                    if np.max(frames) <= 1.0:
                                        frames = (frames * 255).astype(np.uint8)
                                
                                # If channels-first (robomimic standard), transpose to channels-last
                                if frames.ndim == 4 and frames.shape[-3] in [1, 3]: # (T, C, H, W)
                                    frames = np.transpose(frames, (0, 2, 3, 1))
                                
                                # Save final frame as PNG
                                image_path = f"{base}_{demo_name}_failed.png"
                                imageio.imwrite(image_path, frames[-1])
                                
                                # Save whole sequence as MP4
                                video_path = f"{base}_{demo_name}_failed.mp4"
                                imageio.mimsave(video_path, frames, fps=20, macro_block_size=1)
                                print(f"  -> Saved failed demo visualization to:")
                                print(f"     - Image: {image_path}")
                                print(f"     - Video: {video_path}")
                        except Exception as e:
                            print(f"  -> Could not save visualization for {demo_name}: {e}")

            print("\n=== Orientation Distribution Summary ===")
            total_classified = sum(len(demos_by_dir[k]) for k in ["left", "right", "back", "front"])
            for direction in ["left", "right", "back", "front"]:
                count = len(demos_by_dir[direction])
                pct = 100 * count / total_classified if total_classified > 0 else 0
                print(f"  {direction.capitalize():<8}: {count:<4} ({pct:.1f}%)")
            print(f"  Unknown : {len(demos_by_dir['unknown'])} demos")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to HDF5 dataset")
    parser.add_argument("--no-plots", action="store_true", help="Skip saving plots")
    parser.add_argument("--save_failed", action="store_true", help="Saves videos of insertions where it faild to classify the direction")
    args = parser.parse_args()
    analyze(args.path, save_plots=not args.no_plots, save_failed_demos=args.save_failed)
