import cv2
import numpy as np
import pyrealsense2 as rs
import torch

import robomimic.utils.file_utils as FileUtils

# 你的函数，直接复用
import robomimic.utils.tensor_utils as TensorUtils


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
        inputs, nets["policy"]["obs_encoder"], inputs_as_kwargs=True
    )
    obs_cond = obs_features.flatten(start_dim=1)
    return obs_cond


def main():
    ckpt_path = "/home/smlr_g2/smlr_g2_WS25/model_epoch_400_square_d0_success_1.0.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rollout_policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=ckpt_path,
        device=device,
        verbose=True,
    )
    dp_algo = rollout_policy.policy

    # ===== RealSense init =====
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())   # BGR, HWC
            rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # 改成训练时实际分辨率
            rgb_resized = cv2.resize(rgb, (84, 84), interpolation=cv2.INTER_AREA)

            # 先做一个最小 obs dict
            # 还没有 wrist cam，就先拿同一张图占位
            raw_obs = {
                "agentview_image": rgb_resized,
                "robot0_eye_in_hand_image": rgb_resized,
            }

            # 测试 policy 预处理
            ob = rollout_policy._prepare_observation(raw_obs, batched_ob=False)

            # 测试 encoder feature
            feat = extract_state_feature(dp_algo, rollout_policy, raw_obs)

            print("raw_obs keys:", raw_obs.keys())
            print("agentview_image shape:", raw_obs["agentview_image"].shape, raw_obs["agentview_image"].dtype)
            print("prepared keys:", ob.keys() if isinstance(ob, dict) else type(ob))
            print("feature shape:", feat.shape)

            # 显示图像，方便确认
            cv2.imshow("agentview_image", cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1)
            if key == 27:   # ESC
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()