#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DSRL eval script (clean, standalone):
- Loads a frozen diffusion policy checkpoint (robomimic)
- Loads a trained noise-actor checkpoint (SAC actor in noise space)
- Runs N evaluation episodes deterministically
- Records ONE single mp4 that concatenates all episodes (no black screen separators)
- Video filename includes success rate via format string: e.g. ./eval_sr{sr:.3f}_n{n}.mp4
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import imageio
import mimicgen

from dsrl.rollout import extract_state_feature, run_one_chunk, denoise_actions
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.tensor_utils as TensorUtils
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


# -----------------------------
# Networks (must match your saved actor architecture)
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(2048, 2048, 2048)):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SquashedGaussianActor(nn.Module):
    """
    Deterministic eval uses tanh(mu) * action_magnitude.
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden=(2048, 2048, 2048),
        action_magnitude=1.5,
        log_std_min=-5,
        log_std_max=2,
    ):
        super().__init__()
        # backbone outputs hidden[-1]
        self.backbone = MLP(state_dim, out_dim=hidden[-1], hidden=hidden[:-1])
        last_h = hidden[-1]
        self.mu = nn.Linear(last_h, action_dim)
        self.log_std = nn.Linear(last_h, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_magnitude = float(action_magnitude)

    def forward(self, s):
        h = self.backbone(s)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), self.log_std_min, self.log_std_max)
        return mu, log_std

    def deterministic(self, s):
        mu, _ = self.forward(s)
        return self.action_magnitude * torch.tanh(mu)


# -----------------------------
# Feature extraction (freeze DP encoder)
# -----------------------------

def _safe_render_rgb(env, camera_name, h, w):
    """
    Robust offscreen render wrapper (robosuite-style envs).
    """
    try:
        img = env.render(mode="rgb_array", height=h, width=w, camera_name=camera_name)
        # ensure uint8
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    except TypeError:
        # some envs ignore height/width/camera_name
        img = env.render(mode="rgb_array")
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img


@torch.no_grad()
def run_episode(
    env,
    rollout_policy,
    dp_algo,
    actor,
    Ta,
    Tp,
    Da,
    max_steps,
    writer=None,
    video_skip=1,
    camera_name="agentview",
    video_height=512,
    video_width=512,
):
    """
    One episode rollout using DSRL-SAC style:
    - every macro-step: noise actor outputs w_flat -> reshape to [1, Tp, Da]
    - run DDIM denoising once to get the full action trajectory of length Tp
    - execute the first Ta low-level env steps from that trajectory
    - record frames into a single global writer if provided
    """
    obs = env.reset()
    rollout_policy.start_episode()

    ep_return = 0.0
    steps = 0
    success = False
    done = False
    frame_count = 0

    # Record initial frame
    if writer is not None:
        img0 = _safe_render_rgb(env, camera_name, video_height, video_width)
        writer.append_data(img0)
        frame_count = 1

    while steps < max_steps and (not done) and (not success):
        # 1. Sample deterministic noise action from SAC actor
        s_feat = extract_state_feature(dp_algo, rollout_policy, obs)
        a_flat = actor.deterministic(s_feat)
        noise_w = a_flat.view(1, Tp, Da)

        # 2. Denoise with DDIM in noise space to obtain full plan
        actions = denoise_actions(dp_algo, noise_w, s_feat)
        if hasattr(dp_algo, "normalizers") and "actions" in dp_algo.normalizers:
            actions = dp_algo.normalizers["actions"].unnormalize(actions)
        elif hasattr(dp_algo, "unnormalize_actions"):
            actions = dp_algo.unnormalize_actions(actions)

        actions_np = actions.detach().cpu().numpy()

        # 3. Execute the first Ta actions in the environment
        chunk_r = 0.0
        for i in range(Ta):
            act = actions_np[i]
            obs, r, done, _ = env.step(act)
            chunk_r += float(r)
            steps += 1

            is_succ = env.is_success()
            success = bool(is_succ["task"] if isinstance(is_succ, dict) else is_succ)

            if writer is not None:
                if frame_count % max(int(video_skip), 1) == 0:
                    img = _safe_render_rgb(env, camera_name, video_height, video_width)
                    writer.append_data(img)
                frame_count += 1

            if done or success or steps >= max_steps:
                break

        ep_return += chunk_r

    return ep_return, success, steps


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- load diffusion rollout policy (frozen) ----
    rollout_policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args.dp_ckpt,
        device=device,
        verbose=args.verbose,
    )
    dp_algo = rollout_policy.policy

    # ---- force DDIM sampling (DSRL requirement) ----
    with dp_algo.algo_config.values_unlocked():
        dp_algo.algo_config.ddpm.enabled = False
        dp_algo.algo_config.ddim.enabled = True
        dp_algo.algo_config.ddim.num_inference_timesteps = args.ddim_steps

    dp_algo.noise_scheduler = DDIMScheduler(
        num_train_timesteps=dp_algo.algo_config.ddim.num_train_timesteps,
        beta_schedule=dp_algo.algo_config.ddim.beta_schedule,
        clip_sample=dp_algo.algo_config.ddim.clip_sample,
        set_alpha_to_one=dp_algo.algo_config.ddim.set_alpha_to_one,
        steps_offset=dp_algo.algo_config.ddim.steps_offset,
        prediction_type=dp_algo.algo_config.ddim.prediction_type,
    )

    # ---- create env (enable offscreen renderer only if recording) ----
    record = args.video_path is not None
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        env_name=args.env,
        render=False,
        render_offscreen=record,  # IMPORTANT for rgb_array
        verbose=args.verbose,
    )

    # horizons from DP
    Ta = int(dp_algo.algo_config.horizon.action_horizon)
    Tp = int(dp_algo.algo_config.horizon.prediction_horizon)
    Da = int(dp_algo.ac_dim)
    act_dim = Tp * Da

    # probe state_dim
    obs0 = env.reset()
    rollout_policy.start_episode()
    s0 = extract_state_feature(dp_algo, rollout_policy, obs0)
    state_dim = int(s0.shape[-1])

    # ---- load noise actor ckpt ----
    ckpt = torch.load(args.noise_actor_ckpt, map_location=device)
    # try to read settings from checkpoint if present
    action_magnitude = 1.5
    hidden = (2048, 2048, 2048)
    if isinstance(ckpt, dict) and "args" in ckpt and isinstance(ckpt["args"], dict):
        action_magnitude = float(ckpt["args"].get("action_magnitude", action_magnitude))
        # if you saved hidden sizes explicitly you can read them here, otherwise keep default

    actor = SquashedGaussianActor(
        state_dim=state_dim,
        action_dim=act_dim,
        hidden=hidden,
        action_magnitude=action_magnitude,
    ).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    # ---- single video writer for ALL episodes ----

    writer = None
    tmp_video = None
    if record:
        out_dir = args.video_path 
        os.makedirs(out_dir, exist_ok=True)
        tmp_video = os.path.join(out_dir, "video_tmp.mp4")
        writer = imageio.get_writer(tmp_video, fps=args.video_fps)

    returns = []
    succ = 0

    try:
        for i in range(args.n_episodes):
            ep_ret, ep_succ, steps = run_episode(
                env=env,
                rollout_policy=rollout_policy,
                dp_algo=dp_algo,
                actor=actor,
                Ta=Ta,
                Tp=Tp,
                Da=Da,
                max_steps=args.max_steps,
                writer=writer,
                video_skip=args.video_skip,
                camera_name=args.camera_name,
                video_height=args.video_height,
                video_width=args.video_width,
            )
            returns.append(ep_ret)
            succ += int(ep_succ)
            print(f"[EVAL {i:03d}] return={ep_ret:.3f} success={ep_succ} steps={steps}")

    finally:
        if writer is not None:
            writer.close()
        try:
            env.close()
        except Exception:
            pass

    sr = succ / float(args.n_episodes)
    avg_ret = float(np.mean(returns)) if len(returns) else 0.0

    print("==== Summary ====")
    print("Success rate:", sr)
    print("Avg return:", avg_ret)

    # rename video with success rate

    if record and tmp_video is not None:
        out_dir = args.video_path
        os.makedirs(out_dir, exist_ok=True)
        final_path = os.path.join(out_dir, f"video_{sr:.3f}.mp4")
        os.replace(tmp_video, final_path)
        print("[VIDEO] saved to:", final_path)

    # Clean exit
    os._exit(0)


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # required
    p.add_argument("--dp_ckpt", type=str, required=True, help="diffusion policy checkpoint (.pth)")
    p.add_argument("--noise_actor_ckpt", type=str, required=True, help="trained noise-actor checkpoint (.pth)")

    # optional overrides
    p.add_argument("--env", type=str, default=None, help="override env name (default uses checkpoint env)")

    # eval config
    p.add_argument("--n_episodes", type=int, default=27)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--ddim_steps", type=int, default=10)
    p.add_argument("--verbose", action="store_true")

    p.add_argument(
        "--video_path",
        type=str,
        default=None,
    )
    p.add_argument("--camera_name", type=str, default="agentview")
    p.add_argument("--video_fps", type=int, default=20)
    p.add_argument("--video_skip", type=int, default=5, help="write every N frames")
    p.add_argument("--video_height", type=int, default=512)
    p.add_argument("--video_width", type=int, default=512)

    args = p.parse_args()
    main(args)
