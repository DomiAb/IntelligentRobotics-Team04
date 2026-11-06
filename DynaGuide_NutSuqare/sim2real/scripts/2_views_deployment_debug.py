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
from deoxys.utils import YamlConfig

from torchvision.transforms.v2 import functional as F


# =========================================================
# Utility
# =========================================================

def show(imgs):
    import matplotlib.pyplot as plt
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

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


# =========================================================
# Main Deployment
# =========================================================

def main():
    # -----------------------------------------------------
    # User configs
    # -----------------------------------------------------
    # ckpt_path = "/home/smlr_g2/smlr_g2_WS25/deoxys_control/deoxys/g2_repo/ckpt/dr_2views_96*96.pth"
    ckpt_path = "/home/smlr_g2/smlr_g2_SS_26/diffusion_paul/ws2526-g2-repo/ep_520_Square_RealWorldDeployment.pth"

    realsense_width = 1280
    realsense_height = 720
    realsense_fps = 30

    agent_serial = "213322073743"
    wrist_serial = "141722074396"

    agent_crop_box = [420, 70, 980, 630]
    wrist_crop_box = [0, 0, 720, 720]
    
    crop_config_path = "crop_config.json"
    if os.path.exists(crop_config_path):
        try:
            with open(crop_config_path, "r") as f:
                saved_config = json.load(f)
                agent_crop_box = saved_config.get("agent_crop_box", agent_crop_box)
                wrist_crop_box = saved_config.get("wrist_crop_box", wrist_crop_box)
            print(f"\n[INFO] 已加载保存的裁剪框坐标: Agent={agent_crop_box}, Wrist={wrist_crop_box}")
        except Exception as e:
            print(f"\n[WARNING] 读取配置文件失败，将使用默认坐标。错误信息: {e}")

    img_out_size = 84

    controller_type = "OSC_POSE"
    controller_cfg = YamlConfig(
        config_root + "/osc-pose-controller.yml"
    ).as_easydict()

    control_dt = 0.05
    print_every = 20
    use_feature_debug = False
    execute_robot = True

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # -----------------------------------------------------
    # Load policy
    # -----------------------------------------------------
    print("===== loading policy =====")
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

    print("\nControls:")
    print("  ESC : quit and save config")
    print("  s   : save current raw / crop images and config")
    print("  p   : pause / resume robot execution")
    print("  z   : send zero action once")
    print("  1/2 : agent y0 -=/+ 10")
    print("  3/4 : agent x0 -=/+ 10")
    print("  5/6 : agent y1 -=/+ 10")
    print("  7/8 : agent x1 -=/+ 10")
    print("  i/k : wrist y0 -=/+ 10")
    print("  j/l : wrist x0 -=/+ 10")
    print("  t/g : wrist y1 -=/+ 10")
    print("  f/h : wrist x1 -=/+ 10")

    agent_img_hist = deque(maxlen=2)
    wrist_img_hist = deque(maxlen=2)
    eef_pos_hist = deque(maxlen=2)
    eef_quat_hist = deque(maxlen=2)
    gripper_hist = deque(maxlen=2)

    frame_idx = 0
    error_once = False
    paused = False

    i = 0
    ACTIONS = torch.load('/home/smlr_g2/smlr_g2_SS_26/diffusion_robert/ss26_g2_smlr/sim2real/dataset_inspect/actions.pt')

    try:
        while True:
            loop_start = time.time()

            # -------------------------------------------------
            # Camera 1: agentview
            # -------------------------------------------------
            frames_agent = pipeline_agent.wait_for_frames()
            color_frame_agent = frames_agent.get_color_frame()
            if not color_frame_agent:
                continue

            bgr_agent = np.asanyarray(color_frame_agent.get_data())
            rgb_agent = cv2.cvtColor(bgr_agent, cv2.COLOR_BGR2RGB)

            agent_crop_box = clamp_crop_box(agent_crop_box, rgb_agent.shape)
            ax0, ay0, ax1, ay1 = agent_crop_box

            agent_rgb_hwc = crop_and_resize_rgb(
                rgb_agent,
                crop_box=agent_crop_box,
                out_size=img_out_size,
            ).astype(np.uint8)

            # -------------------------------------------------
            # Camera 2: eye-in-hand
            # -------------------------------------------------
            frames_wrist = pipeline_wrist.wait_for_frames()
            color_frame_wrist = frames_wrist.get_color_frame()
            if not color_frame_wrist:
                continue

            bgr_wrist = np.asanyarray(color_frame_wrist.get_data())
            # 2026 Fix
            bgr_wrist = np.flip(bgr_wrist, axis=(0, 1))


            rgb_wrist = cv2.cvtColor(bgr_wrist, cv2.COLOR_BGR2RGB)

            wrist_crop_box = clamp_crop_box(wrist_crop_box, rgb_wrist.shape)
            wx0, wy0, wx1, wy1 = wrist_crop_box

            wrist_rgb_hwc = crop_and_resize_rgb(
                rgb_wrist,
                crop_box=wrist_crop_box,
                out_size=img_out_size,
            ).astype(np.uint8)

            # # 2026 Fix
            # wrist_rgb_hwc = np.flip(wrist_rgb_hwc, axis=(0, 1))

            # -------------------------------------------------
            # Robot state
            # -------------------------------------------------
            robot_state = get_robot_state(robot_interface)
            if robot_state is None:
                continue

            eef_pos = robot_state["robot0_eef_pos"]
            eef_quat = robot_state["robot0_eef_quat"]
            gripper_qpos = robot_state["robot0_gripper_qpos"]

            # -------------------------------------------------
            # Fill history
            # -------------------------------------------------
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

            if len(agent_img_hist) < 2:
                continue

            raw_obs = make_stacked_raw_obs(
                agent_img_hist=agent_img_hist,
                wrist_img_hist=wrist_img_hist,
                eef_pos_hist=eef_pos_hist,
                eef_quat_hist=eef_quat_hist,
                gripper_hist=gripper_hist,
            )

            # -------------------------------------------------
            # Policy inference
            # -------------------------------------------------
            try:
                # 2026 Fix (C, H, W)
                agent_image = torch.from_numpy(raw_obs["agentview_image"]).movedim(-1, -3).float() / 255.0
                hand_image = torch.from_numpy(raw_obs["robot0_eye_in_hand_image"]).movedim(-1, -3).float() / 255.0

                # og_agent_image = agent_image.clone()
                # og_hand_image = hand_image.clone()

                # Image editing
                # agent_image = agent_image * torch.tensor((1., .85, .85))[:, None, None]
                # agent_image = F.adjust_brightness(agent_image, brightness_factor=1.3)
                # agent_image = F.adjust_saturation(agent_image, saturation_factor=1.2)

                # hand_image = hand_image * torch.tensor((1., .85, .85))[:, None, None]
                # hand_image = F.adjust_brightness(hand_image, brightness_factor=1.3)
                # hand_image = F.adjust_saturation(hand_image, saturation_factor=1.2)

                raw_obs["agentview_image"] = agent_image
                raw_obs["robot0_eye_in_hand_image"] = hand_image

                policy_action = rollout_policy(ob=raw_obs)
                if i >= len(ACTIONS): quit()
                policy_action = ACTIONS[i]
                i += 1

                policy_action = np.array(policy_action, dtype=np.float32).reshape(-1)

                if use_feature_debug:
                    feat = extract_state_feature(dp_algo, rollout_policy, raw_obs)
                else:
                    feat = None

                ctrl_action = postprocess_policy_action(
                    policy_action,
                    gripper_threshold=0.0,
                )

                # -------------------------------------------------
                # Robot control
                # -------------------------------------------------
                if execute_robot and (not paused):
                    robot_interface.control(
                        controller_type=controller_type,
                        action=ctrl_action,
                        controller_cfg=controller_cfg,
                    )

                # -------------------------------------------------
                # Logging
                # -------------------------------------------------
                if frame_idx % print_every == 0:
                    print_raw_obs_info(raw_obs)
                    print("eef_pos:", eef_pos)
                    print("eef_quat:", eef_quat)
                    print("gripper_qpos:", gripper_qpos)
                    print("policy_action:", policy_action, policy_action.shape)
                    print("ctrl_action:", ctrl_action, ctrl_action.shape)
                    if feat is not None:
                        print("feature shape:", tuple(feat.shape))

                msg = (
                    f"\rframe={frame_idx:05d} "
                    f"agent_crop={agent_crop_box} "
                    f"wrist_crop={wrist_crop_box} "
                    f"policy_shape={policy_action.shape} "
                    f"ctrl={np.round(ctrl_action, 4)} "
                    f"{'[PAUSED]' if paused else '[RUN]'}"
                )
                print(msg, end="")

            except Exception:
                if not error_once:
                    print("\n\n[PIPELINE ERROR FULL]")
                    print_raw_obs_info(raw_obs)
                    traceback.print_exc()
                    error_once = True

                try:
                    robot_interface.control(
                        controller_type=controller_type,
                        action=safe_zero_action(),
                        controller_cfg=controller_cfg,
                    )
                except Exception:
                    pass

                cv2.waitKey(0)
                break

            # -------------------------------------------------
            # Visualization
            # -------------------------------------------------
            vis_agent = bgr_agent.copy()
            cv2.rectangle(vis_agent, (ax0, ay0), (ax1, ay1), (0, 255, 0), 2)
            crop_agent_bgr = cv2.cvtColor(agent_rgb_hwc, cv2.COLOR_RGB2BGR)

            vis_wrist = bgr_wrist.copy()
            cv2.rectangle(vis_wrist, (wx0, wy0), (wx1, wy1), (0, 255, 0), 2)
            crop_wrist_bgr = cv2.cvtColor(wrist_rgb_hwc, cv2.COLOR_RGB2BGR)

            cv2.imshow("agent_raw_with_crop", vis_agent)
            cv2.imshow("agentview_image", crop_agent_bgr)
            cv2.imshow("wrist_raw_with_crop", vis_wrist)
            cv2.imshow("robot0_eye_in_hand_image", crop_wrist_bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                with open(crop_config_path, "w") as f:
                    json.dump({
                        "agent_crop_box": agent_crop_box,
                        "wrist_crop_box": wrist_crop_box
                    }, f, indent=4)
                print(f"\n[SAVED] 裁剪框坐标已自动保存至 {crop_config_path}")
                break
            elif key == ord('s'):
                cv2.imwrite("agent_raw_with_crop.png", vis_agent)
                cv2.imwrite("agentview_image.png", crop_agent_bgr)
                cv2.imwrite("wrist_raw_with_crop.png", vis_wrist)
                cv2.imwrite("robot0_eye_in_hand_image.png", crop_wrist_bgr)
                
                with open(crop_config_path, "w") as f:
                    json.dump({
                        "agent_crop_box": agent_crop_box,
                        "wrist_crop_box": wrist_crop_box
                    }, f, indent=4)
                print(f"\n[SAVED] 图片及裁剪框坐标配置已保存")
            elif key == ord('p'):
                paused = not paused
                print(f"\n[PAUSE TOGGLE] paused={paused}")
            elif key == ord('z'):
                zero_action = safe_zero_action()
                robot_interface.control(
                    controller_type=controller_type,
                    action=zero_action,
                    controller_cfg=controller_cfg,
                )
                print("\n[SENT ZERO ACTION]")

            # agent crop
            elif key == ord('1'):
                agent_crop_box[1] -= 10
            elif key == ord('2'):
                agent_crop_box[1] += 10
            elif key == ord('3'):
                agent_crop_box[0] -= 10
            elif key == ord('4'):
                agent_crop_box[0] += 10
            elif key == ord('5'):
                agent_crop_box[3] -= 10
            elif key == ord('6'):
                agent_crop_box[3] += 10
            elif key == ord('7'):
                agent_crop_box[2] -= 10
            elif key == ord('8'):
                agent_crop_box[2] += 10

            # wrist crop
            elif key == ord('i'):
                wrist_crop_box[1] -= 10
            elif key == ord('k'):
                wrist_crop_box[1] += 10
            elif key == ord('j'):
                wrist_crop_box[0] -= 10
            elif key == ord('l'):
                wrist_crop_box[0] += 10
            elif key == ord('t'):
                wrist_crop_box[3] -= 10
            elif key == ord('g'):
                wrist_crop_box[3] += 10
            elif key == ord('f'):
                wrist_crop_box[2] -= 10
            elif key == ord('h'):
                wrist_crop_box[2] += 10

            frame_idx += 1

            elapsed = time.time() - loop_start
            sleep_t = control_dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    finally:
        print("\n\n[EXIT] sending zero action several times for safety...")
        try:
            zero_action = safe_zero_action()
            for _ in range(5):
                robot_interface.control(
                    controller_type=controller_type,
                    action=zero_action,
                    controller_cfg=controller_cfg,
                )
                time.sleep(0.05)
        except Exception:
            pass

        try:
            pipeline_agent.stop()
        except Exception:
            pass

        try:
            pipeline_wrist.stop()
        except Exception:
            pass

        try:
            robot_interface.close()
        except Exception:
            pass

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()