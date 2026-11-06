import os
import json
import time
import argparse
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils

from diffusers.schedulers.scheduling_ddim import DDIMScheduler


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


def hard_update(target: nn.Module, source: nn.Module):
    target.load_state_dict(source.state_dict())


# -----------------------------
# Replay Buffer (CPU)
# -----------------------------
class ReplayBuffer:
    def __init__(self, state_dim, act_dim, capacity=int(1e6)):
        self.capacity = int(capacity)
        self.ptr = 0
        self.size = 0
        self.s = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.a = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.r = np.zeros((self.capacity, 1), dtype=np.float32)
        self.s2 = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.d = np.zeros((self.capacity, 1), dtype=np.float32)

    def add(self, s, a, r, s2, d):
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.s2[self.ptr] = s2
        self.d[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx])


# -----------------------------
# Networks
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
    Tanh-squashed Gaussian policy:
      u ~ N(mu, std)
      a = action_magnitude * tanh(u)
    logp uses tanh correction (crucial for SAC stability).
    """
    def __init__(self, state_dim, action_dim, hidden=(2048, 2048, 2048),
                 action_magnitude=1.5, log_std_min=-5, log_std_max=2):
        super().__init__()
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
        log_std = self.log_std(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, s):
        """
        returns:
          a: [B, act_dim] in [-action_magnitude, action_magnitude]
          logp: [B,1] corrected
        """
        mu, log_std = self.forward(s)
        std = torch.exp(log_std)
        eps = torch.randn_like(mu)
        u = mu + std * eps
        a_tanh = torch.tanh(u)
        a = self.action_magnitude * a_tanh

        # log prob of Gaussian (u)
        # sum over action dims
        logp_u = (-0.5 * (((u - mu) / (std + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))).sum(dim=-1, keepdim=True)

        # tanh correction: log|det(da/du)| = sum log(action_mag*(1-tanh(u)^2))
        # => subtract it from logp_u
        corr = torch.log(self.action_magnitude * (1.0 - a_tanh.pow(2)) + 1e-6).sum(dim=-1, keepdim=True)
        logp = logp_u - corr
        return a, logp

    def deterministic(self, s):
        mu, _ = self.forward(s)
        a = self.action_magnitude * torch.tanh(mu)
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=(2048, 2048, 2048)):
        super().__init__()
        self.q1 = MLP(state_dim + action_dim, 1, hidden=hidden)
        self.q2 = MLP(state_dim + action_dim, 1, hidden=hidden)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.q1(x), self.q2(x)


# -----------------------------
# Feature extraction (freeze DP encoder)
# -----------------------------
@torch.no_grad()
def extract_state_feature(dp_algo, rollout_policy, raw_obs, raw_goal=None):
    """
    raw_obs: env 原始 obs dict（numpy）
    return: obs_cond [1, state_dim]
    """
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
    )  # [B, To, D]
    obs_cond = obs_features.flatten(start_dim=1)  # [B, To*D]
    return obs_cond


# -----------------------------
# One macro-step (chunk)
# -----------------------------
def run_one_chunk(env, rollout_policy, dp_algo, actor, obs, Ta, Tp, Da,
                  deterministic=False):
    """
    s -> w -> execute Ta env steps using same w -> s'
    """
    s_feat = extract_state_feature(dp_algo, rollout_policy, obs)  # [1, state_dim]

    if deterministic:
        a_flat = actor.deterministic(s_feat)
        logp = None
    else:
        a_flat, logp = actor.sample(s_feat)

    # reshape -> noise_w [1, Tp, Da]
    noise_w = a_flat.view(1, Tp, Da)

    chunk_reward = 0.0
    done = False
    success = False
    next_obs = obs

    for _ in range(Ta):
        act = rollout_policy(ob=next_obs, noise_w=noise_w)
        next_obs, r, done, _ = env.step(act)
        chunk_reward += float(r)
        success = env.is_success()["task"]
        if done or success:
            break

    next_feat = extract_state_feature(dp_algo, rollout_policy, next_obs)
    return next_obs, chunk_reward, done, success, s_feat, a_flat, logp, next_feat


# -----------------------------
# SAC update
# -----------------------------
def sac_update(actor, critic, critic_target, actor_opt, critic_opt,
               log_alpha, alpha_opt,
               replay: ReplayBuffer,
               batch_size,
               gamma_macro,
               tau,
               device,
               target_entropy,
               auto_alpha=True,
               grad_clip=None,
               amp=False,
               scaler=None):
    s, a, r, s2, d = replay.sample(batch_size)
    s = torch.from_numpy(s).to(device)
    a = torch.from_numpy(a).to(device)
    r = torch.from_numpy(r).to(device)
    s2 = torch.from_numpy(s2).to(device)
    d = torch.from_numpy(d).to(device)

    alpha = log_alpha.exp()

    # --- critic ---
    with torch.no_grad():
        a2, logp2 = actor.sample(s2)
        q1_t, q2_t = critic_target(s2, a2)
        q_t = torch.min(q1_t, q2_t) - alpha * logp2
        y = r + (1.0 - d) * gamma_macro * q_t

    if amp and scaler is not None:
        with torch.cuda.amp.autocast():
            q1, q2 = critic(s, a)
            critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        critic_opt.zero_grad(set_to_none=True)
        scaler.scale(critic_loss).backward()
        if grad_clip is not None:
            scaler.unscale_(critic_opt)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
        scaler.step(critic_opt)
    else:
        q1, q2 = critic(s, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
        critic_opt.step()

    # --- actor ---
    if amp and scaler is not None:
        with torch.cuda.amp.autocast():
            a_new, logp_new = actor.sample(s)
            q1_new, q2_new = critic(s, a_new)
            q_new = torch.min(q1_new, q2_new)
            actor_loss = (alpha * logp_new - q_new).mean()
        actor_opt.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()
        if grad_clip is not None:
            scaler.unscale_(actor_opt)
            torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
        scaler.step(actor_opt)
    else:
        a_new, logp_new = actor.sample(s)
        q1_new, q2_new = critic(s, a_new)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (alpha * logp_new - q_new).mean()
        actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
        actor_opt.step()

    # --- alpha ---
    alpha_loss = None
    if auto_alpha:
        # minimize: -log_alpha * (logp + target_entropy)
        alpha_loss = -(log_alpha * (logp_new + target_entropy).detach()).mean()
        alpha_opt.zero_grad(set_to_none=True)
        if amp and scaler is not None:
            scaler.scale(alpha_loss).backward()
            scaler.step(alpha_opt)
        else:
            alpha_loss.backward()
            alpha_opt.step()
        alpha = log_alpha.exp()

    # --- scaler update ---
    if amp and scaler is not None:
        scaler.update()

    soft_update(critic_target, critic, tau=tau)

    return {
        "critic_loss": float(critic_loss.item()),
        "actor_loss": float(actor_loss.item()),
        "alpha": float(alpha.item()),
        "alpha_loss": float(alpha_loss.item()) if alpha_loss is not None else None,
        "reward": float(r.mean().item()),
    }


def save_checkpoint(path, actor, critic, critic_target, log_alpha, args, meta, extra):
    payload = {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "critic_target": critic_target.state_dict(),
        "log_alpha": float(log_alpha.detach().cpu().item()),
        "args": vars(args),
        "meta": meta,
        "extra": extra,
    }
    torch.save(payload, path)


# -----------------------------
# Main
# -----------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # speed knobs (safe defaults)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # -------- load diffusion policy --------
    rollout_policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args.agent,
        device=device,
        verbose=args.verbose,
    )
    dp_algo = rollout_policy.policy

    # switch to DDIM (required by DSRL doc)
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

    # create env
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        env_name=args.env,
        render=False,
        render_offscreen=False,
        verbose=args.verbose,
    )

    # horizons
    Ta = int(dp_algo.algo_config.horizon.action_horizon)
    Tp = int(dp_algo.algo_config.horizon.prediction_horizon)
    Da = int(dp_algo.ac_dim)
    act_dim = Tp * Da

    # probe state_dim
    obs0 = env.reset()
    rollout_policy.start_episode()
    s0_feat = extract_state_feature(dp_algo, rollout_policy, obs0)
    state_dim = int(s0_feat.shape[-1])

    print(f"[INFO] Ta={Ta}, Tp={Tp}, Da={Da} -> act_dim={act_dim}, state_dim={state_dim}")
    print(f"[INFO] action_magnitude={args.action_magnitude}, utd={args.updates_per_iter}")

    # -------- SAC modules --------
    hidden = (args.hidden, args.hidden, args.hidden)
    actor = SquashedGaussianActor(
        state_dim=state_dim,
        action_dim=act_dim,
        hidden=hidden,
        action_magnitude=args.action_magnitude,
    ).to(device)
    critic = Critic(state_dim=state_dim, action_dim=act_dim, hidden=hidden).to(device)
    critic_target = Critic(state_dim=state_dim, action_dim=act_dim, hidden=hidden).to(device)
    hard_update(critic_target, critic)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    log_alpha = torch.tensor(np.log(args.init_alpha), device=device, requires_grad=True)
    alpha_opt = torch.optim.Adam([log_alpha], lr=args.alpha_lr)

    target_entropy = -float(act_dim) if args.target_entropy is None else float(args.target_entropy)

    replay = ReplayBuffer(state_dim=state_dim, act_dim=act_dim, capacity=args.replay_size)

    gamma_macro = (args.gamma ** Ta)

    # AMP
    amp = args.amp and (device.type == "cuda")
    scaler = torch.amp.GradScaler(enabled=amp)

    # saving
    os.makedirs(args.out_dir, exist_ok=True)
    success_window = deque(maxlen=args.sr_window)

    t0 = time.time()
    total_chunks = 0
    total_env_steps = 0

    try:
        for ep in range(args.num_episodes):
            obs = env.reset()
            rollout_policy.start_episode()

            ep_return = 0.0
            ep_steps = 0
            ep_success = False

            while ep_steps < args.max_steps:
                next_obs, chunk_r, done, success, s_feat, a_flat, logp, next_feat = run_one_chunk(
                    env=env,
                    rollout_policy=rollout_policy,
                    dp_algo=dp_algo,
                    actor=actor,
                    obs=obs,
                    Ta=Ta,
                    Tp=Tp,
                    Da=Da,
                    deterministic=False,
                )

                # store (CPU)
                replay.add(
                    s_feat.detach().cpu().numpy().reshape(-1),
                    a_flat.detach().cpu().numpy().reshape(-1),
                    np.array([chunk_r], dtype=np.float32),
                    next_feat.detach().cpu().numpy().reshape(-1),
                    np.array([float(done or success)], dtype=np.float32),
                )

                obs = next_obs
                ep_return += chunk_r
                ep_steps += Ta
                total_chunks += 1
                total_env_steps += Ta
                ep_success = bool(success)

                # updates (UTD)
                if replay.size >= args.start_updates and (total_chunks % args.update_every == 0):
                    for _ in range(args.updates_per_iter):
                        _info = sac_update(
                            actor=actor,
                            critic=critic,
                            critic_target=critic_target,
                            actor_opt=actor_opt,
                            critic_opt=critic_opt,
                            log_alpha=log_alpha,
                            alpha_opt=alpha_opt,
                            replay=replay,
                            batch_size=args.batch_size,
                            gamma_macro=gamma_macro,
                            tau=args.tau,
                            device=device,
                            target_entropy=target_entropy,
                            auto_alpha=args.auto_alpha,
                            grad_clip=args.grad_clip,
                            amp=amp,
                            scaler=scaler,
                        )

                if done or success:
                    break

            success_window.append(1.0 if ep_success else 0.0)
            sr = float(np.mean(success_window)) if len(success_window) > 0 else 0.0

            print(f"[EP {ep:04d}] return={ep_return:.3f} steps~{ep_steps} success={ep_success} "
                  f"SR({args.sr_window})={sr:.3f} replay={replay.size} chunks={total_chunks} env_steps~{total_env_steps}")

            # save
            if (ep + 1) % args.save_every == 0:
                fname = f"noise_actor_ep{ep+1:06d}_sr{sr:.3f}.pth"
                save_path = os.path.join(args.out_dir, fname)
                save_checkpoint(
                    save_path,
                    actor, critic, critic_target, log_alpha,
                    args,
                    meta={"Ta": Ta, "Tp": Tp, "Da": Da, "state_dim": state_dim, "act_dim": act_dim},
                    extra={"rolling_success_rate": sr, "episode": ep + 1, "elapsed_sec": time.time() - t0},
                )
                # latest_path = os.path.join(args.out_dir, "noise_actor_latest.pth")
                # save_checkpoint(
                #     latest_path,
                #     actor, critic, critic_target, log_alpha,
                #     args,
                #     meta={"Ta": Ta, "Tp": Tp, "Da": Da, "state_dim": state_dim, "act_dim": act_dim},
                #     extra={"rolling_success_rate": sr, "episode": ep + 1, "elapsed_sec": time.time() - t0},
                # )
                print(f"[SAVE] {save_path}")

        print(f"[DONE] elapsed={time.time() - t0:.1f}s")

    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--agent", type=str, required=True, help="diffusion policy checkpoint (.pth)")
    p.add_argument("--env", type=str, default=None, help="override env name (optional)")

    # schedule
    p.add_argument("--num_episodes", type=int, default=600)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)

    # replay + updates
    p.add_argument("--replay_size", type=int, default=200000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--start_updates", type=int, default=2000)
    p.add_argument("--update_every", type=int, default=1)
    p.add_argument("--updates_per_iter", type=int, default=20)  # utd

    # SAC hyperparams
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--actor_lr", type=float, default=3e-4)
    p.add_argument("--critic_lr", type=float, default=3e-4)

    p.add_argument("--auto_alpha", action="store_true")
    p.add_argument("--init_alpha", type=float, default=0.2)
    p.add_argument("--alpha_lr", type=float, default=3e-4)
    p.add_argument("--target_entropy", type=float, default=None)

    # DSRL knobs
    p.add_argument("--action_magnitude", type=float, default=0.5)  
    p.add_argument("--ddim_steps", type=int, default=10)

    # net size (official tip: 2048)
    p.add_argument("--hidden", type=int, default=2048)

    # stability / speed
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--amp", action="store_true", help="use mixed precision for actor/critic updates (GPU)")
    p.add_argument("--verbose", action="store_true")

    # saving
    p.add_argument("--out_dir", type=str, default="./dsrl_outputs")
    p.add_argument("--save_every", type=int, default=27)
    p.add_argument("--sr_window", type=int, default=27, help="rolling success rate window")

    args = p.parse_args()
    main(args)
