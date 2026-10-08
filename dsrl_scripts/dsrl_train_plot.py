"""
DSRL training entry point.

Loads a pre-trained Diffusion Policy checkpoint, freezes its obs_encoder,
and fine-tunes a SAC actor in the latent noise space using online RL.

Usage
-----
python dsrl_train_plot.py --agent <path/to/dp_checkpoint.pth> [options]
"""

import os
import time
import argparse
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

import robomimic.utils.file_utils as FileUtils
import mimicgen

from diffusers.schedulers.scheduling_ddim import DDIMScheduler

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from dsrl.networks import SquashedGaussianActor, Critic
from dsrl.buffer import ReplayBuffer
from dsrl.utils import (
    set_seed, soft_update, hard_update,
    save_checkpoint, plot_success_curves,
)
from dsrl.rollout import extract_state_feature, run_one_chunk, evaluate_success_rate



# ---------------------------------------------------------------------------
# SAC update
# ---------------------------------------------------------------------------

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
    """
    One gradient step of SAC: update critic, actor, and optionally the
    entropy temperature alpha.

    Returns
    -------
    dict with keys: critic_loss, actor_loss, alpha, alpha_loss, reward
    """
    s, a, r, s2, d = replay.sample(batch_size)
    s  = torch.from_numpy(s).to(device)
    a  = torch.from_numpy(a).to(device)
    r  = torch.from_numpy(r).to(device)
    s2 = torch.from_numpy(s2).to(device)
    d  = torch.from_numpy(d).to(device)

    alpha = log_alpha.exp()

    # --- critic target ---
    with torch.no_grad():
        a2, logp2 = actor.sample(s2)
        q1_t, q2_t = critic_target(s2, a2)
        q_t = torch.min(q1_t, q2_t) - alpha * logp2
        y   = r + (1.0 - d) * gamma_macro * q_t

    # --- critic update ---
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

    # --- actor update ---
    if amp and scaler is not None:
        with torch.cuda.amp.autocast():
            a_new, logp_new = actor.sample(s)
            q1_new, q2_new  = critic(s, a_new)
            actor_loss = (alpha * logp_new - torch.min(q1_new, q2_new)).mean()
        actor_opt.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()
        if grad_clip is not None:
            scaler.unscale_(actor_opt)
            torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
        scaler.step(actor_opt)
    else:
        a_new, logp_new = actor.sample(s)
        q1_new, q2_new  = critic(s, a_new)
        actor_loss = (alpha * logp_new - torch.min(q1_new, q2_new)).mean()
        actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
        actor_opt.step()

    # --- alpha (entropy temperature) update ---
    alpha_loss = None
    if auto_alpha:
        alpha_loss = -(log_alpha * (logp_new + target_entropy).detach()).mean()
        alpha_opt.zero_grad(set_to_none=True)
        if amp and scaler is not None:
            scaler.scale(alpha_loss).backward()
            scaler.step(alpha_opt)
        else:
            alpha_loss.backward()
            alpha_opt.step()
        alpha = log_alpha.exp()

    if amp and scaler is not None:
        scaler.update()

    soft_update(critic_target, critic, tau=tau)

    return {
        "critic_loss": float(critic_loss.item()),
        "actor_loss":  float(actor_loss.item()),
        "alpha":       float(alpha.item()),
        "alpha_loss":  float(alpha_loss.item()) if alpha_loss is not None else None,
        "reward":      float(r.mean().item()),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # ---- load diffusion policy ----
    rollout_policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=args.agent,
        device=device,
        verbose=args.verbose,
    )
    dp_algo = rollout_policy.policy

    # switch DDPM -> DDIM (required by DSRL)
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

    # ---- create environment ----
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        env_name=args.env,
        render=False,
        render_offscreen=False,
        verbose=args.verbose,
    )

    # ---- infer dimensions ----
    Ta      = int(dp_algo.algo_config.horizon.action_horizon)
    Tp      = int(dp_algo.algo_config.horizon.prediction_horizon)
    Da      = int(dp_algo.ac_dim)
    act_dim = Tp * Da

    obs0 = env.reset()
    rollout_policy.start_episode()
    s0_feat   = extract_state_feature(dp_algo, rollout_policy, obs0)
    state_dim = int(s0_feat.shape[-1])

    print(f"[INFO] Ta={Ta}, Tp={Tp}, Da={Da} -> act_dim={act_dim}, state_dim={state_dim}")
    print(f"[INFO] action_magnitude={args.action_magnitude}, utd={args.updates_per_iter}")

    # ---- SAC modules ----
    hidden = (args.hidden, args.hidden, args.hidden)
    actor = SquashedGaussianActor(
        state_dim=state_dim,
        action_dim=act_dim,
        hidden=hidden,
        action_magnitude=args.action_magnitude,
    ).to(device)
    critic        = Critic(state_dim=state_dim, action_dim=act_dim, hidden=hidden).to(device)
    critic_target = Critic(state_dim=state_dim, action_dim=act_dim, hidden=hidden).to(device)
    hard_update(critic_target, critic)

    actor_opt  = torch.optim.Adam(actor.parameters(),  lr=args.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    log_alpha = torch.tensor(np.log(args.init_alpha), device=device, requires_grad=True)
    alpha_opt = torch.optim.Adam([log_alpha], lr=args.alpha_lr)

    target_entropy = -float(act_dim) if args.target_entropy is None else float(args.target_entropy)

    replay      = ReplayBuffer(state_dim=state_dim, act_dim=act_dim, capacity=args.replay_size)
    gamma_macro = args.gamma ** Ta

    amp    = args.amp and (device.type == "cuda")
    scaler = torch.amp.GradScaler(enabled=amp)

    os.makedirs(args.out_dir, exist_ok=True)
    success_window = deque(maxlen=args.sr_window)

    # ---- wandb ----
    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        print("[WANDB] wandb not installed, skipping. Run: pip install wandb")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=vars(args),
        )
        print(f"[WANDB] run: {wandb.run.name}  url: {wandb.run.url}")

    # ---- curve buffers ----
    train_sr_hist = []
    eval_x        = []
    eval_sr_hist  = []

    # optional initial evaluation before training
    if args.eval_before_train:
        init_eval = evaluate_success_rate(
            env=env, rollout_policy=rollout_policy, dp_algo=dp_algo,
            actor=actor, Ta=Ta, Tp=Tp, Da=Da,
            max_steps=args.max_steps, num_episodes=args.eval_episodes,
        )
        eval_x.append(0)
        eval_sr_hist.append(init_eval)
        print(f"[EVAL] ep=0 eval_SR({args.eval_episodes})={init_eval:.3f}")
        if use_wandb:
            wandb.log({"eval/SR": init_eval}, step=0)

    t0              = time.time()
    total_chunks    = 0
    total_env_steps = 0

    try:
        for ep in range(args.num_episodes):
            # Check if the environment or any underlying wrapper implements set_bound_idx
            target_env = env
            while hasattr(target_env, "env") and not hasattr(target_env, "set_bound_idx"):
                target_env = target_env.env

            if hasattr(target_env, "set_bound_idx"):
                # Query number of bounds dynamically
                num_bounds = len(getattr(target_env, "ALL_PLACEMENT_BOUNDS", [0]))
                bound_idx = int(np.random.randint(num_bounds))
                print(f"[EP {ep:04d}] choose bound {bound_idx + 1}/{num_bounds}")
                target_env.set_bound_idx(bound_idx)

            obs = env.reset()
            rollout_policy.start_episode()

            ep_return  = 0.0
            ep_steps   = 0
            ep_success = False

            while ep_steps < args.max_steps:
                next_obs, chunk_r, done, success, s_feat, a_flat, logp, next_feat = run_one_chunk(
                    env=env, rollout_policy=rollout_policy, dp_algo=dp_algo,
                    actor=actor, obs=obs, Ta=Ta, Tp=Tp, Da=Da, deterministic=False,
                )

                replay.add(
                    s_feat.detach().cpu().numpy().reshape(-1),
                    a_flat.detach().cpu().numpy().reshape(-1),
                    np.array([chunk_r],               dtype=np.float32),
                    next_feat.detach().cpu().numpy().reshape(-1),
                    np.array([float(done or success)], dtype=np.float32),
                )

                obs          = next_obs
                ep_return   += chunk_r
                ep_steps    += Ta
                total_chunks += 1
                total_env_steps += Ta
                ep_success   = bool(success)

                if (replay.size >= args.start_updates
                        and total_chunks % args.update_every == 0):
                    last_info = None
                    for _ in range(args.updates_per_iter):
                        last_info = sac_update(
                            actor=actor, critic=critic, critic_target=critic_target,
                            actor_opt=actor_opt, critic_opt=critic_opt,
                            log_alpha=log_alpha, alpha_opt=alpha_opt,
                            replay=replay, batch_size=args.batch_size,
                            gamma_macro=gamma_macro, tau=args.tau, device=device,
                            target_entropy=target_entropy, auto_alpha=args.auto_alpha,
                            grad_clip=args.grad_clip, amp=amp, scaler=scaler,
                        )
                    if use_wandb and last_info is not None:
                        wandb.log({
                            "train/critic_loss": last_info["critic_loss"],
                            "train/actor_loss":  last_info["actor_loss"],
                            "train/alpha":       last_info["alpha"],
                            **( {"train/alpha_loss": last_info["alpha_loss"]}
                                if last_info["alpha_loss"] is not None else {} ),
                        }, step=ep)

                if done or success:
                    break

            success_window.append(1.0 if ep_success else 0.0)
            sr = float(np.mean(success_window)) if success_window else 0.0
            train_sr_hist.append(sr)

            print(
                f"[EP {ep:04d}] return={ep_return:.3f} steps~{ep_steps} "
                f"success={ep_success} SR({args.sr_window})={sr:.3f} "
                f"replay={replay.size} chunks={total_chunks} env_steps~{total_env_steps}"
            )

            if use_wandb:
                wandb.log({
                    "train/rolling_SR":  sr,
                    "train/ep_return":   ep_return,
                    "train/ep_steps":    ep_steps,
                    "train/success":     float(ep_success),
                    "train/replay_size": replay.size,
                    "train/env_steps":   total_env_steps,
                }, step=ep)

            # periodic deterministic evaluation
            if (ep + 1) % args.eval_every == 0:
                eval_sr = evaluate_success_rate(
                    env=env, rollout_policy=rollout_policy, dp_algo=dp_algo,
                    actor=actor, Ta=Ta, Tp=Tp, Da=Da,
                    max_steps=args.max_steps, num_episodes=args.eval_episodes,
                )
                eval_x.append(ep + 1)
                eval_sr_hist.append(eval_sr)
                print(f"[EVAL] ep={ep+1} eval_SR({args.eval_episodes})={eval_sr:.3f}")
                if use_wandb:
                    wandb.log({"eval/SR": eval_sr}, step=ep + 1)

            # checkpoint
            if (ep + 1) % args.save_every == 0:
                fname     = f"noise_actor_ep{ep+1:06d}_sr{sr:.3f}.pth"
                save_path = os.path.join(args.out_dir, fname)
                save_checkpoint(
                    save_path, actor, critic, critic_target, log_alpha, args,
                    meta={
                        "Ta": Ta, "Tp": Tp, "Da": Da,
                        "state_dim": state_dim, "act_dim": act_dim,
                    },
                    extra={
                        "rolling_success_rate": sr,
                        "episode":              ep + 1,
                        "elapsed_sec":          time.time() - t0,
                    },
                )
                print(f"[SAVE] {save_path}")

        print(f"[DONE] elapsed={time.time() - t0:.1f}s")

    finally:
        # plot before env.close() to avoid SIGABRT from MuJoCo teardown
        try:
            plot_success_curves(
                out_dir=args.out_dir,
                train_sr_hist=train_sr_hist,
                eval_x=eval_x,
                eval_sr_hist=eval_sr_hist,
                smooth_beta=args.smooth_beta,
            )
        except Exception as e:
            print(f"[PLOT] failed: {e}")

        if use_wandb:
            try:
                wandb.finish()
            except Exception:
                pass

        try:
            env.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="DSRL: fine-tune a Diffusion Policy with SAC in noise space."
    )
    p.add_argument("--agent", type=str, required=True,
                   help="Path to the Diffusion Policy checkpoint (.pth)")
    p.add_argument("--env",   type=str, default=None,
                   help="Override environment name (optional)")

    # training schedule
    p.add_argument("--num_episodes", type=int,   default=600)
    p.add_argument("--max_steps",    type=int,   default=400)
    p.add_argument("--seed",         type=int,   default=0)

    # replay buffer & update schedule
    p.add_argument("--replay_size",      type=int, default=200_000)
    p.add_argument("--batch_size",       type=int, default=256)
    p.add_argument("--start_updates",    type=int, default=2000)
    p.add_argument("--update_every",     type=int, default=1)
    p.add_argument("--updates_per_iter", type=int, default=20,
                   help="Update-to-data (UTD) ratio")

    # SAC hyperparameters
    p.add_argument("--gamma",     type=float, default=0.99)
    p.add_argument("--tau",       type=float, default=0.005)
    p.add_argument("--actor_lr",  type=float, default=3e-4)
    p.add_argument("--critic_lr", type=float, default=3e-4)

    p.add_argument("--auto_alpha",     action="store_true")
    p.add_argument("--init_alpha",     type=float, default=0.2)
    p.add_argument("--alpha_lr",       type=float, default=3e-4)
    p.add_argument("--target_entropy", type=float, default=None)

    # DSRL-specific
    p.add_argument("--action_magnitude", type=float, default=0.5,
                   help="Tanh squashing scale for the noise actor")
    p.add_argument("--ddim_steps",       type=int,   default=10,
                   help="DDIM inference steps (replaces DDPM for speed)")

    # network size
    p.add_argument("--hidden", type=int, default=2048,
                   help="Hidden-layer width for actor and critic MLPs")

    # stability & speed
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--amp",     action="store_true",
                   help="Mixed-precision training for actor/critic (GPU only)")
    p.add_argument("--verbose", action="store_true")

    # checkpointing
    p.add_argument("--out_dir",    type=str, default="./dsrl_outputs")
    p.add_argument("--save_every", type=int, default=27)
    p.add_argument("--sr_window",  type=int, default=27,
                   help="Window size for rolling training success rate")

    # evaluation
    p.add_argument("--eval_every",        type=int,   default=10,
                   help="Run deterministic evaluation every N episodes")
    p.add_argument("--eval_episodes",     type=int,   default=10,
                   help="Number of episodes per evaluation")
    p.add_argument("--smooth_beta",       type=float, default=0.95,
                   help="EMA decay for success-rate curves")
    p.add_argument("--eval_before_train", action="store_true",
                   help="Evaluate at episode 0 before any training")

    # wandb
    p.add_argument("--wandb",          action="store_true")
    p.add_argument("--wandb_project",  type=str, default="dsrl")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_entity",   type=str, default=None)

    args = p.parse_args()
    main(args)
