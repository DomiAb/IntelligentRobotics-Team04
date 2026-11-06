import cv2
import time
import numpy as np
import pyrealsense2 as rs
import torch
import traceback
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


def start_realsense(width=1280, height=720, fps=30):
    ctx = rs.context()
    devices = ctx.query_devices()

    print("===== RealSense devices =====")
    print("device count:", len(devices))
    if len(devices) == 0:
        raise RuntimeError("No RealSense device found by pyrealsense2")

    dev = devices[0]
    name = dev.get_info(rs.camera_info.name)
    serial = dev.get_info(rs.camera_info.serial_number)
    print(f"using device: {name}, serial={serial}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    profile = pipeline.start(config)
    print("pipeline started successfully")
    return pipeline, profile


def crop_and_resize_rgb(rgb, crop_box, out_size=84):
    x0, y0, x1, y1 = crop_box
    crop = rgb[y0:y1, x0:x1]
    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return crop


def get_robot_state(robot_interface):
    """
    从 deoxys 读取 policy 需要的 low-dim state:
      robot0_eef_pos      -> (3,)
      robot0_eef_quat     -> (4,) in wxyz
      robot0_gripper_qpos -> (2,)
    """
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

    # gripper
    gripper_qpos = None
    if hasattr(robot_interface, "last_gripper_q") and robot_interface.last_gripper_q is not None:
        g = np.array(robot_interface.last_gripper_q, dtype=np.float32).reshape(-1)
        if g.size == 1:
            gripper_qpos = np.array([g[0], g[0]], dtype=np.float32)
        elif g.size >= 2:
            gripper_qpos = g[:2]

    # fallback
    if gripper_qpos is None or np.any(np.isnan(gripper_qpos)):
        gripper_qpos = np.array([0.0, 0.0], dtype=np.float32)

    return {
        "robot0_eef_pos": eef_pos,
        "robot0_eef_quat": eef_quat,
        "robot0_gripper_qpos": gripper_qpos,
    }


def make_stacked_raw_obs(agent_img_hist, eef_pos_hist, eef_quat_hist, gripper_hist):
    assert len(agent_img_hist) == 2
    assert len(eef_pos_hist) == 2
    assert len(eef_quat_hist) == 2
    assert len(gripper_hist) == 2

    raw_obs = {
        "agentview_image": np.stack(list(agent_img_hist), axis=0).astype(np.uint8),
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


def main():
    ckpt_path = "/home/smlr_g2/smlr_g2_WS25/model_epoch_400_square_d0_success_1.0.pth"
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # ===== load policy =====
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

    # ===== start devices =====
    pipeline, _ = start_realsense(width=1280, height=720, fps=30)

    robot_interface = FrankaInterface(
        config_root + "/charmander.yml",
        use_visualizer=False,
    )

    print("waiting for robot state...")
    while len(robot_interface._state_buffer) == 0:
        time.sleep(0.05)
    print("robot state ready")

    # 初始 crop
    crop_box = [320, 70, 880, 630]

    # 2-frame history
    agent_img_hist = deque(maxlen=2)
    eef_pos_hist = deque(maxlen=2)
    eef_quat_hist = deque(maxlen=2)
    gripper_hist = deque(maxlen=2)

    error_once = False
    frame_idx = 0

    print("\nControls:")
    print("  ESC : quit")
    print("  s   : save current raw / crop images")
    print("  i/k : y0 -=/+ 10")
    print("  j/l : x0 -=/+ 10")
    print("  t/g : y1 -=/+ 10")
    print("  f/h : x1 -=/+ 10")

    try:
        while True:
            # ===== camera =====
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            bgr = np.asanyarray(color_frame.get_data())
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # clamp crop box
            x0, y0, x1, y1 = crop_box
            x0 = max(0, min(x0, rgb.shape[1] - 2))
            x1 = max(x0 + 1, min(x1, rgb.shape[1]))
            y0 = max(0, min(y0, rgb.shape[0] - 2))
            y1 = max(y0 + 1, min(y1, rgb.shape[0]))
            crop_box = [x0, y0, x1, y1]

            agent_rgb_hwc = crop_and_resize_rgb(rgb, crop_box, out_size=84).astype(np.uint8)

            # ===== robot =====
            robot_state = get_robot_state(robot_interface)
            if robot_state is None:
                continue

            eef_pos = robot_state["robot0_eef_pos"]
            eef_quat = robot_state["robot0_eef_quat"]
            gripper_qpos = robot_state["robot0_gripper_qpos"]

            # push history
            agent_img_hist.append(agent_rgb_hwc)
            eef_pos_hist.append(eef_pos)
            eef_quat_hist.append(eef_quat)
            gripper_hist.append(gripper_qpos)

            if len(agent_img_hist) < 2:
                while len(agent_img_hist) < 2:
                    agent_img_hist.append(agent_rgb_hwc.copy())
                    eef_pos_hist.append(eef_pos.copy())
                    eef_quat_hist.append(eef_quat.copy())
                    gripper_hist.append(gripper_qpos.copy())

            raw_obs = make_stacked_raw_obs(
                agent_img_hist=agent_img_hist,
                eef_pos_hist=eef_pos_hist,
                eef_quat_hist=eef_quat_hist,
                gripper_hist=gripper_hist,
            )

            # ===== pipeline forward =====
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

                print(
                    f"\rframe={frame_idx:05d} crop={crop_box} feature={tuple(feat.shape)} act_shape={np.array(act).shape}",
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

            # ===== visualization =====
            vis = bgr.copy()
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
            crop_bgr = cv2.cvtColor(agent_rgb_hwc, cv2.COLOR_RGB2BGR)

            cv2.imshow("raw_with_crop", vis)
            cv2.imshow("agentview_84x84", crop_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('s'):
                cv2.imwrite("raw_with_crop.png", vis)
                cv2.imwrite("agentview_84x84.png", crop_bgr)
                print(f"\n[SAVED] raw_with_crop.png, agentview_84x84.png, crop={crop_box}")
            elif key == ord('i'):
                crop_box[1] -= 10
            elif key == ord('k'):
                crop_box[1] += 10
            elif key == ord('j'):
                crop_box[0] -= 10
            elif key == ord('l'):
                crop_box[0] += 10
            elif key == ord('t'):
                crop_box[3] -= 10
            elif key == ord('g'):
                crop_box[3] += 10
            elif key == ord('f'):
                crop_box[2] -= 10
            elif key == ord('h'):
                crop_box[2] += 10

            frame_idx += 1

    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        try:
            robot_interface.close()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()