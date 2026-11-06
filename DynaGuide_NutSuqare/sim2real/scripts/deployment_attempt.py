import argparse
import os
import json
import h5py
import imageio
import sys
import time
import traceback
import numpy as np
from copy import deepcopy
from tqdm import tqdm

import torch

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

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.utils.log_utils import log_warning
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper
from robomimic.algo import RolloutPolicy
from robomimic.scripts.playback_dataset import DEFAULT_CAMERAS

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig

def list_realsense_devices():
    ctx = rs.context()
    devices = ctx.query_devices()
    infos = []
    print("===== RealSense devices =====")
    print("device count:", len(devices))
    for i, dev in enumerate(devices):
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        print(f"[{i}] name={name}, serial={serial}")
        infos.append((name, serial))
    return infos

def make_stacked_raw_obs(
    agent_img_hist,
    wrist_img_hist,
    eef_pos_hist,
    eef_quat_hist,
    gripper_hist,
):
    assert len(agent_img_hist) == 2
    assert len(wrist_img_hist) == 2
    assert len(eef_pos_hist) == 2
    assert len(eef_quat_hist) == 2
    assert len(gripper_hist) == 2

    raw_obs = {
        "agentview_image": np.stack(list(agent_img_hist), axis=0).astype(np.uint8),
        "robot0_eye_in_hand_image": np.stack(list(wrist_img_hist), axis=0).astype(np.uint8),
        "robot0_eef_pos": np.stack(list(eef_pos_hist), axis=0).astype(np.float32),
        "robot0_eef_quat": np.stack(list(eef_quat_hist), axis=0).astype(np.float32),
        "robot0_gripper_qpos": np.stack(list(gripper_hist), axis=0).astype(np.float32),
    }
    return raw_obs


def print_raw_obs_info(raw_obs):
    print("\n===== raw_obs =====")
    for k, v in raw_obs.items():
        arr = np.array(v)
        print(f"{k:28s} shape={arr.shape} dtype={arr.dtype}")


def postprocess_policy_action(policy_action, gripper_threshold=0.0):
    a = np.array(policy_action, dtype=np.float32).reshape(-1)
    if a.size < 7:
        raise ValueError(f"Policy action dim={a.size}, expected >= 7")

    a = a[:7].copy()
    a[:3] = np.clip(a[:3], -1.0, 1.0)
    # 如需关掉旋转，取消下一行注释
    # a[3:6] = 0.0
    a[6] = 1.0 if a[6] > gripper_threshold else -1.0
    return a.astype(np.float32)


def safe_zero_action():
    return np.zeros(7, dtype=np.float32)


def start_realsense_by_serial(serial, width=1280, height=720, fps=30):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(config)
    print(f"pipeline started successfully for serial={serial}")
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

    # 修改点 2：调用 xyzw 的转换函数
    eef_quat = R.from_matrix(eef_rot).as_quat().astype(np.float32)

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

def warmup_history_two_views(
    agent_img_hist,
    wrist_img_hist,
    eef_pos_hist,
    eef_quat_hist,
    gripper_hist,
    agent_rgb_hwc,
    wrist_rgb_hwc,
    eef_pos,
    eef_quat,
    gripper_qpos,
):
    agent_img_hist.append(agent_rgb_hwc)
    wrist_img_hist.append(wrist_rgb_hwc)
    eef_pos_hist.append(eef_pos)
    eef_quat_hist.append(eef_quat)
    gripper_hist.append(gripper_qpos)

    while len(agent_img_hist) < 2:
        agent_img_hist.append(agent_rgb_hwc.copy())
        wrist_img_hist.append(wrist_rgb_hwc.copy())
        eef_pos_hist.append(eef_pos.copy())
        eef_quat_hist.append(eef_quat.copy())
        gripper_hist.append(gripper_qpos.copy())

ckpt_path = '/home/smlr_g2/smlr_g2_SS_26/diffusion_robert/ss26_g2_smlr/sim2real/models/ep_520_Square_RealWorldDeployment.pth'
algo_name, ckpt_dict = FileUtils.algo_name_from_checkpoint(ckpt_path)
device = TorchUtils.get_torch_device(try_to_use_cuda=True)
rollout_policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_dict=ckpt_dict, device=device, verbose=True)
dp_algo = rollout_policy.policy
rollout_policy.start_episode()
config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
ObsUtils.initialize_obs_utils_with_config(config)

realsense_width = 1280
realsense_height = 720
realsense_fps = 30

agent_serial = "213322073743"
wrist_serial = "141722074396"

agent_crop_box = [420, 70, 980, 630]
wrist_crop_box = [420, 120, 980, 680]

img_out_size = 84

print("\n===== expected obs keys =====")
print(config.all_obs_keys)
print("\n===== obs modalities =====")
print(config.observation.modalities)

# -----------------------------------------------------
# List and start cameras
# -----------------------------------------------------
list_realsense_devices()

pipeline_agent, _ = start_realsense_by_serial(
    serial=agent_serial,
    width=realsense_width,
    height=realsense_height,
    fps=realsense_fps,
)
pipeline_wrist, _ = start_realsense_by_serial(
    serial=wrist_serial,
    width=realsense_width,
    height=realsense_height,
    fps=realsense_fps,
)

# -----------------------------------------------------
# Start robot
# -----------------------------------------------------
robot_interface = FrankaInterface(
    config_root + "/charmander.yml",
    use_visualizer=False,
)

print("waiting for robot state...")
while len(robot_interface._state_buffer) == 0:
    time.sleep(0.05)
print("robot state ready")


agent_img_hist = deque(maxlen=2)
wrist_img_hist = deque(maxlen=2)
eef_pos_hist = deque(maxlen=2)
eef_quat_hist = deque(maxlen=2)
gripper_hist = deque(maxlen=2)


frames_agent = pipeline_agent.wait_for_frames()
color_frame_agent = frames_agent.get_color_frame()

bgr_agent = np.asanyarray(color_frame_agent.get_data())
rgb_agent = cv2.cvtColor(bgr_agent, cv2.COLOR_BGR2RGB)

agent_crop_box = clamp_crop_box(agent_crop_box, rgb_agent.shape)
ax0, ay0, ax1, ay1 = agent_crop_box

agent_rgb_hwc = crop_and_resize_rgb(
    rgb_agent,
    crop_box=agent_crop_box,
    out_size=img_out_size,
).astype(np.uint8)


frames_wrist = pipeline_wrist.wait_for_frames()
color_frame_wrist = frames_wrist.get_color_frame()


bgr_wrist = np.asanyarray(color_frame_wrist.get_data())
rgb_wrist = cv2.cvtColor(bgr_wrist, cv2.COLOR_BGR2RGB)

wrist_crop_box = clamp_crop_box(wrist_crop_box, rgb_wrist.shape)
wx0, wy0, wx1, wy1 = wrist_crop_box

wrist_rgb_hwc = crop_and_resize_rgb(
    rgb_wrist,
    crop_box=wrist_crop_box,
    out_size=img_out_size,
).astype(np.uint8)

robot_state = get_robot_state(robot_interface)

eef_pos = robot_state["robot0_eef_pos"]
eef_quat = robot_state["robot0_eef_quat"]
gripper_qpos = robot_state["robot0_gripper_qpos"]

if len(agent_img_hist) == 0:
    warmup_history_two_views(
        agent_img_hist,
        wrist_img_hist,
        eef_pos_hist,
        eef_quat_hist,
        gripper_hist,
        agent_rgb_hwc,
        wrist_rgb_hwc,
        eef_pos,
        eef_quat,
        gripper_qpos,
    )
else:
    agent_img_hist.append(agent_rgb_hwc)
    wrist_img_hist.append(wrist_rgb_hwc)
    eef_pos_hist.append(eef_pos)
    eef_quat_hist.append(eef_quat)
    gripper_hist.append(gripper_qpos)


raw_obs = make_stacked_raw_obs(
    agent_img_hist=agent_img_hist,
    wrist_img_hist=wrist_img_hist,
    eef_pos_hist=eef_pos_hist,
    eef_quat_hist=eef_quat_hist,
    gripper_hist=gripper_hist,
)

print(raw_obs["agentview_image"].shape)

action = rollout_policy(ob=raw_obs)