"""
Environment interaction and evaluation utilities for DSRL.

The Diffusion Policy (DP) obs_encoder is kept frozen throughout; only the
SAC actor (which lives in networks.py) is trained in the latent noise space.
"""

import torch
import robomimic.utils.tensor_utils as TensorUtils


@torch.no_grad()
def extract_state_feature(dp_algo, rollout_policy, raw_obs, raw_goal=None):
    """
    Encode raw_obs into a flat conditioning feature vector using the frozen DP encoder.
    """
    ob = rollout_policy._prepare_observation(raw_obs)
    goal = None
    if raw_goal is not None:
        goal = rollout_policy._prepare_observation(raw_goal)

    nets = dp_algo.nets
    if hasattr(dp_algo, "ema") and dp_algo.ema is not None:
        if hasattr(dp_algo.ema, "averaged_model"):
            nets = dp_algo.ema.averaged_model

    if "policy" in nets:
        policy_net = nets["policy"]
        if isinstance(policy_net, (dict, torch.nn.ModuleDict)):
            encoder = policy_net["obs_encoder"]
        else:
            encoder = getattr(policy_net, "obs_encoder", policy_net)
    elif "obs_encoder" in nets:
        encoder = nets["obs_encoder"]
    else:
        encoder = getattr(dp_algo, "obs_encoder", nets)

    inputs = {"obs": ob, "goal": goal}
    obs_features = TensorUtils.time_distributed(
        inputs, encoder, inputs_as_kwargs=True
    )  # [1, To, D]
    obs_cond = obs_features.flatten(start_dim=1)  # [1, To*D]
    return obs_cond


@torch.no_grad()
def denoise_actions(dp_algo, noise_w, obs_cond):
    """
    Denoise initial noise vector noise_w [1, Tp, Da] into an action trajectory [Tp, Da]
    conditioned on obs_cond [1, state_dim] using DDIM.
    """
    device = noise_w.device
    scheduler = dp_algo.noise_scheduler
    scheduler.set_timesteps(dp_algo.algo_config.ddim.num_inference_timesteps)

    nets = dp_algo.nets
    if hasattr(dp_algo, "ema") and dp_algo.ema is not None:
        if hasattr(dp_algo.ema, "averaged_model"):
            nets = dp_algo.ema.averaged_model

    if "policy" in nets:
        policy_net = nets["policy"]
        if isinstance(policy_net, (dict, torch.nn.ModuleDict)):
            noise_pred_net = policy_net["noise_pred_net"]
        else:
            noise_pred_net = getattr(policy_net, "noise_pred_net", policy_net)
    elif "noise_pred_net" in nets:
        noise_pred_net = nets["noise_pred_net"]
    else:
        noise_pred_net = getattr(dp_algo, "noise_pred_net", nets)

    sample = noise_w.clone().to(device)
    for t in scheduler.timesteps:
        timesteps = t
        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.tensor([t], dtype=torch.long, device=device)
        elif timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0).to(device)
        else:
            timesteps = timesteps.to(device)

        noise_pred = noise_pred_net(sample=sample, timestep=timesteps, global_cond=obs_cond)
        sample = scheduler.step(noise_pred, t, sample).prev_sample

    return sample.squeeze(0)  # [Tp, Da]


def run_one_chunk(env, rollout_policy, dp_algo, actor, obs, Ta, Tp, Da,
                  deterministic=False):
    """
    Execute one macro-step:
    1. Extract observation feature s_feat.
    2. Sample initial noise vector w from SAC actor.
    3. Run DDIM denoising starting from w to get Tp actions.
    4. Step the environment for Ta steps using the generated trajectory.
    """
    s_feat = extract_state_feature(dp_algo, rollout_policy, obs)

    if deterministic:
        a_flat = actor.deterministic(s_feat)
        logp   = None
    else:
        a_flat, logp = actor.sample(s_feat)

    noise_w = a_flat.view(1, Tp, Da)

    # Generate full action plan via DDIM denoising
    actions = denoise_actions(dp_algo, noise_w, s_feat)
    actions_np = actions.detach().cpu().numpy()

    chunk_reward = 0.0
    done         = False
    success      = False
    next_obs     = obs

    for i in range(Ta):
        act = actions_np[i]
        next_obs, r, done, _ = env.step(act)
        chunk_reward += float(r)

        is_succ = env.is_success()
        success = is_succ["task"] if isinstance(is_succ, dict) else bool(is_succ)
        if done or success:
            break

    next_feat = extract_state_feature(dp_algo, rollout_policy, next_obs)
    return next_obs, chunk_reward, done, success, s_feat, a_flat, logp, next_feat


@torch.no_grad()
def evaluate_success_rate(env, rollout_policy, dp_algo, actor, Ta, Tp, Da,
                          max_steps, num_episodes=10):
    """
    Deterministic evaluation over num_episodes episodes.
    """
    actor.eval()
    successes = 0.0

    for _ in range(num_episodes):
        obs = env.reset()
        rollout_policy.start_episode()

        steps   = 0
        done    = False
        success = False

        while steps < max_steps and not done and not success:
            next_obs, _, done, success, *_ = run_one_chunk(
                env=env,
                rollout_policy=rollout_policy,
                dp_algo=dp_algo,
                actor=actor,
                obs=obs,
                Ta=Ta,
                Tp=Tp,
                Da=Da,
                deterministic=True,
            )
            obs    = next_obs
            steps += Ta

        successes += 1.0 if success else 0.0

    actor.train()
    return successes / float(num_episodes)