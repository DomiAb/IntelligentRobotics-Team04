import cv2
import time
import numpy as np
import pyrealsense2 as rs
import torch
import traceback
import json
import os
from collections import deque
from scipy.spatial.transform import Rotation as R

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface


@torch.no_grad()
def extract_state_feature(dp_algo, rollout_policy, raw_obs, raw_goal=None):
    ob = rollout_policy._prepare_observation(raw_obs, batched_ob=False)
    goal = None
    if raw_goal is not None:
        goal = rollout_policy._prepare_observation(raw_goal, batched_ob=False)

    nets = dp_algo.nets
    if dp_algo.ema is not None:
        nets = dp_algo.ema.averaged_model

    inputs = {"obs": ob, "goal": goal}
    obs_features = TensorUtils.time_distributed(
        inputs,
        nets["policy"]["obs_encoder"],
        inputs_as_kwargs=True,
    )
    obs_cond = obs_features.flatten(start_dim=1)
    return obs_cond


def rotmat_to_quat_wxyz(rot):
    quat_xyzw = R.from_matrix(rot).as_quat()
    quat_wxyz = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float32,
    )
    return quat_wxyz


def list_realsense_devices():
    ctx = rs.context()
    devices = ctx.query_devices()

    print("===== RealSense devices =====")
    print("device count:", len(devices))
    for i, dev in enumerate(devices):
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        print(f"[{i}] {name}, serial={serial}")

    return devices


def start_realsense_by_serial(serial, width=1280, height=720, fps=30, name="camera"):
    ctx = rs.context()
    devices = ctx.query_devices()

    if len(devices) == 0:
        raise RuntimeError("No RealSense device found by pyrealsense2")

    found = False
    for dev in devices:
        dev_serial = dev.get_info(rs.camera_info.serial_number)
        if dev_serial == serial:
            found = True
            break

    if not found:
        raise RuntimeError(f"Target serial {serial} not found")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    profile = pipeline.start(config)
    print(f"{name} started successfully, serial={serial}")
    return pipeline, profile


def crop_and_resize_rgb(rgb, crop_box, out_size=84):
    x0, y0, x1, y1 = crop_box
    crop = rgb[y0:y1, x0:x1]
    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return crop


def clamp_crop_box(crop_box, img_shape):
    h, w = img_shape[:2]
    x0, y0, x1, y1 = crop_box

    x0 = max(0, min(x0, w - 2))
    x1 = max(x0 + 1, min(x1, w))
    y0 = max(0, min(y0, h - 2))
    y1 = max(y0 + 1, min(y1, h))

    return [x0, y0, x1, y1]


def get_robot_state(robot_interface):
    if len(robot_interface._state_buffer) == 0:
        return None

    last_eef_rot, last_eef_pos = robot_interface.last_eef_rot_and_pos

    eef_pos = np.array(last_eef_pos, dtype=np.float32).reshape(-1)
    if eef_pos.shape[0] > 3:
        eef_pos = eef_pos[:3]

    eef_rot = np.array(last_eef_rot, dtype=np.float32)
    if eef_rot.shape != (3, 3):
        eef_rot = eef_rot.reshape(3, 3)

    eef_quat = rotmat_to_quat_wxyz(eef_rot)

    gripper_qpos = None
    if hasattr(robot_interface, "last_gripper_q") and robot_interface.last_gripper_q is not None:
        g = np.array(robot_interface.last_gripper_q, dtype=np.float32).reshape(-1)
        if g.size == 1:
            gripper_qpos = np.array([g[0], g[0]], dtype=np.float32)
        elif g.size >= 2:
            gripper_qpos = g[:2]

    if gripper_qpos is None or np.any(np.isnan(gripper_qpos)):
        gripper_qpos = np.array([0.0, 0.0], dtype=np.float32)

    return {
        "robot0_eef_pos": eef_pos,
        "robot0_eef_quat": eef_quat,
        "robot0_gripper_qpos": gripper_qpos,
    }


def make_stacked_raw_obs(agent_img_hist, wrist_img_hist, eef_pos_hist, eef_quat_hist, gripper_hist):
    assert len(agent_img_hist) == 2
    assert len(wrist_img_hist) == 2
    assert len(eef_pos_hist) == 2
    assert len(eef_quat_hist) == 2
    assert len(gripper_hist) == 2

    raw_obs = {
        "agentview_image": np.stack(list(agent_img_hist), axis=0).astype(np.uint8),
        # 注意: 这里的键名 "wristview_image" 需与你训练模型时 config 中的键名保持一致
        "wristview_image": np.stack(list(wrist_img_hist), axis=0).astype(np.uint8),
        "robot0_eef_pos": np.stack(list(eef_pos_hist), axis=0).astype(np.float32),
        "robot0_eef_quat": np.stack(list(eef_quat_hist), axis=0).astype(np.float32),
        "robot0_gripper_qpos": np.stack(list(gripper_hist), axis=0).astype(np.float32),
    }
    return raw_obs


def print_raw_obs_info(raw_obs):
    print("\n===== raw_obs =====")
    for k, v in raw_obs.items():
        arr = np.array(v)
        print(f"{k:22s} shape={arr.shape} dtype={arr.dtype}")


def draw_crop_box_with_label(img, crop_box, label, color=(0, 255, 0)):
    vis = img.copy()
    x0, y0, x1, y1 = crop_box
    cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
    cv2.putText(
        vis,
        label,
        (x0, max(20, y0 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    return vis


def main():
    ckpt_path = "/home/smlr_g2/smlr_g2_WS25/model_epoch_400_square_d0_success_1.0.pth"
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    WRIST_SERIAL = "141722074396"
    AGENT_SERIAL = "233522077069"

    # 读取或初始化配置文件
    crop_config_path = "crop_config.json"
    agent_crop_box = [320, 70, 880, 630]
    wrist_crop_box = [320, 70, 880, 630]

    if os.path.exists(crop_config_path):
        try:
            with open(crop_config_path, "r") as f:
                saved_config = json.load(f)
                agent_crop_box = saved_config.get("agent_crop_box", agent_crop_box)
                wrist_crop_box = saved_config.get("wrist_crop_box", wrist_crop_box)
            print(f"\n[INFO] 已加载保存的裁剪框坐标: Agent={agent_crop_box}, Wrist={wrist_crop_box}")
        except Exception as e:
            print(f"\n[WARNING] 读取配置文件失败，将使用默认坐标。错误信息: {e}")

    rollout_policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=ckpt_path,
        device=device,
        verbose=True,
    )
    dp_algo = rollout_policy.policy
    rollout_policy.start_episode()

    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    ObsUtils.initialize_obs_utils_with_config(config)

    print("\n===== expected obs keys =====")
    print(config.all_obs_keys)
    print("\n===== obs modalities =====")
    print(config.observation.modalities)

    list_realsense_devices()

    wrist_pipeline, _ = start_realsense_by_serial(
        WRIST_SERIAL, width=1280, height=720, fps=30, name="wrist_cam"
    )
    agent_pipeline, _ = start_realsense_by_serial(
        AGENT_SERIAL, width=1280, height=720, fps=30, name="agent_cam"
    )

    robot_interface = FrankaInterface(
        config_root + "/charmander.yml",
        use_visualizer=False,
    )

    print("waiting for robot state...")
    while len(robot_interface._state_buffer) == 0:
        time.sleep(0.05)
    print("robot state ready")

    active_crop = "agent"

    agent_img_hist = deque(maxlen=2)
    wrist_img_hist = deque(maxlen=2)
    eef_pos_hist = deque(maxlen=2)
    eef_quat_hist = deque(maxlen=2)
    gripper_hist = deque(maxlen=2)

    error_once = False
    frame_idx = 0

    print("\nControls:")
    print("  ESC : quit and save config")
    print("  s   : save current images and config")
    print("  1   : select AGENT crop box")
    print("  2   : select WRIST crop box")
    print("  i/k : y0 -=/+ 10")
    print("  j/l : x0 -=/+ 10")
    print("  t/g : y1 -=/+ 10")
    print("  f/h : x1 -=/+ 10")

    try:
        while True:
            wrist_frames = wrist_pipeline.wait_for_frames()
            wrist_color_frame = wrist_frames.get_color_frame()

            agent_frames = agent_pipeline.wait_for_frames()
            agent_color_frame = agent_frames.get_color_frame()

            if not wrist_color_frame or not agent_color_frame:
                continue

            wrist_bgr = np.asanyarray(wrist_color_frame.get_data())
            agent_bgr = np.asanyarray(agent_color_frame.get_data())

            wrist_rgb = cv2.cvtColor(wrist_bgr, cv2.COLOR_BGR2RGB)
            agent_rgb = cv2.cvtColor(agent_bgr, cv2.COLOR_BGR2RGB)

            wrist_crop_box = clamp_crop_box(wrist_crop_box, wrist_rgb.shape)
            agent_crop_box = clamp_crop_box(agent_crop_box, agent_rgb.shape)

            wrist_rgb_hwc = crop_and_resize_rgb(wrist_rgb, wrist_crop_box, out_size=84).astype(np.uint8)
            agent_rgb_hwc = crop_and_resize_rgb(agent_rgb, agent_crop_box, out_size=84).astype(np.uint8)

            robot_state = get_robot_state(robot_interface)
            if robot_state is None:
                continue

            eef_pos = robot_state["robot0_eef_pos"]
            eef_quat = robot_state["robot0_eef_quat"]
            gripper_qpos = robot_state["robot0_gripper_qpos"]

            agent_img_hist.append(agent_rgb_hwc)
            wrist_img_hist.append(wrist_rgb_hwc)
            eef_pos_hist.append(eef_pos)
            eef_quat_hist.append(eef_quat)
            gripper_hist.append(gripper_qpos)

            if len(agent_img_hist) < 2:
                while len(agent_img_hist) < 2:
                    agent_img_hist.append(agent_rgb_hwc.copy())
                    wrist_img_hist.append(wrist_rgb_hwc.copy())
                    eef_pos_hist.append(eef_pos.copy())
                    eef_quat_hist.append(eef_quat.copy())
                    gripper_hist.append(gripper_qpos.copy())

            raw_obs = make_stacked_raw_obs(
                agent_img_hist=agent_img_hist,
                wrist_img_hist=wrist_img_hist,
                eef_pos_hist=eef_pos_hist,
                eef_quat_hist=eef_quat_hist,
                gripper_hist=gripper_hist,
            )

            try:
                ob = rollout_policy._prepare_observation(raw_obs, batched_ob=False)
                feat = extract_state_feature(dp_algo, rollout_policy, raw_obs)
                act = rollout_policy(ob=raw_obs)

                if frame_idx % 30 == 0:
                    print_raw_obs_info(raw_obs)
                    print("eef_pos:", eef_pos)
                    print("eef_quat:", eef_quat)
                    print("gripper_qpos:", gripper_qpos)
                    print("feature shape:", tuple(feat.shape))
                    print("action:", np.array(act), np.array(act).shape)
                    print("active_crop:", active_crop)
                    print("agent_crop_box:", agent_crop_box)
                    print("wrist_crop_box:", wrist_crop_box)

                print(
                    f"\rframe={frame_idx:05d} active_crop={active_crop} "
                    f"agent_crop={agent_crop_box} wrist_crop={wrist_crop_box} "
                    f"feature={tuple(feat.shape)} act_shape={np.array(act).shape}",
                    end=""
                )

            except Exception:
                if not error_once:
                    print("\n\n[PIPELINE ERROR FULL]")
                    print_raw_obs_info(raw_obs)
                    traceback.print_exc()
                    error_once = True
                cv2.waitKey(0)
                break

            agent_color = (0, 255, 0) if active_crop == "agent" else (255, 0, 0)
            wrist_color = (0, 255, 0) if active_crop == "wrist" else (255, 0, 0)

            wrist_vis = draw_crop_box_with_label(
                wrist_bgr, wrist_crop_box, f"wrist crop ({active_crop})", wrist_color
            )
            agent_vis = draw_crop_box_with_label(
                agent_bgr, agent_crop_box, f"agent crop ({active_crop})", agent_color
            )

            wrist_crop_bgr = cv2.cvtColor(wrist_rgb_hwc, cv2.COLOR_RGB2BGR)
            agent_crop_bgr = cv2.cvtColor(agent_rgb_hwc, cv2.COLOR_RGB2BGR)

            cv2.imshow("wrist_raw_with_crop", wrist_vis)
            cv2.imshow("agent_raw_with_crop", agent_vis)
            cv2.imshow("wristview_84x84", wrist_crop_bgr)
            cv2.imshow("agentview_84x84", agent_crop_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                with open(crop_config_path, "w") as f:
                    json.dump({
                        "agent_crop_box": agent_crop_box,
                        "wrist_crop_box": wrist_crop_box
                    }, f, indent=4)
                print(f"\n[SAVED] 裁剪框坐标已自动保存至 {crop_config_path}")
                break
            elif key == ord('s'):
                cv2.imwrite("wrist_raw_with_crop.png", wrist_vis)
                cv2.imwrite("agent_raw_with_crop.png", agent_vis)
                cv2.imwrite("wristview_84x84.png", wrist_crop_bgr)
                cv2.imwrite("agentview_84x84.png", agent_crop_bgr)
                
                with open(crop_config_path, "w") as f:
                    json.dump({
                        "agent_crop_box": agent_crop_box,
                        "wrist_crop_box": wrist_crop_box
                    }, f, indent=4)
                print(f"\n[SAVED] 图片已保存，裁剪框坐标已保存至 {crop_config_path}")
            elif key == ord('1'):
                active_crop = "agent"
                print("\n[INFO] active crop switched to AGENT")
            elif key == ord('2'):
                active_crop = "wrist"
                print("\n[INFO] active crop switched to WRIST")
            elif key == ord('i'):
                if active_crop == "agent":
                    agent_crop_box[1] -= 10
                else:
                    wrist_crop_box[1] -= 10
            elif key == ord('k'):
                if active_crop == "agent":
                    agent_crop_box[1] += 10
                else:
                    wrist_crop_box[1] += 10
            elif key == ord('j'):
                if active_crop == "agent":
                    agent_crop_box[0] -= 10
                else:
                    wrist_crop_box[0] -= 10
            elif key == ord('l'):
                if active_crop == "agent":
                    agent_crop_box[0] += 10
                else:
                    wrist_crop_box[0] += 10
            elif key == ord('t'):
                if active_crop == "agent":
                    agent_crop_box[3] -= 10
                else:
                    wrist_crop_box[3] -= 10
            elif key == ord('g'):
                if active_crop == "agent":
                    agent_crop_box[3] += 10
                else:
                    wrist_crop_box[3] += 10
            elif key == ord('f'):
                if active_crop == "agent":
                    agent_crop_box[2] -= 10
                else:
                    wrist_crop_box[2] -= 10
            elif key == ord('h'):
                if active_crop == "agent":
                    agent_crop_box[2] += 10
                else:
                    wrist_crop_box[2] += 10

            frame_idx += 1

    finally:
        try:
            wrist_pipeline.stop()
        except Exception:
            pass
        try:
            agent_pipeline.stop()
        except Exception:
            pass
        try:
            robot_interface.close()
        except Exception:
            pass

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()