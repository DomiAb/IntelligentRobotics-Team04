"""
Neural network definitions for DSRL.

Classes
-------
MLP                  : Generic multi-layer perceptron with ReLU activations.
SquashedGaussianActor: Tanh-squashed Gaussian policy for SAC.
Critic               : Twin Q-network (double-Q) for SAC.
"""

import numpy as np
import torch
import torch.nn as nn


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
        Returns
        -------
        a    : [B, act_dim] in [-action_magnitude, action_magnitude]
        logp : [B, 1] log-probability with tanh Jacobian correction
        """
        mu, log_std = self.forward(s)
        std = torch.exp(log_std)
        eps = torch.randn_like(mu)
        u = mu + std * eps
        a_tanh = torch.tanh(u)
        a = self.action_magnitude * a_tanh

        logp_u = (
            -0.5 * (((u - mu) / (std + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        ).sum(dim=-1, keepdim=True)

        # Jacobian correction: log|det(da/du)| = sum log(action_mag * (1 - tanh(u)^2))
        corr = torch.log(
            self.action_magnitude * (1.0 - a_tanh.pow(2)) + 1e-6
        ).sum(dim=-1, keepdim=True)
        logp = logp_u - corr
        return a, logp

    def deterministic(self, s):
        mu, _ = self.forward(s)
        return self.action_magnitude * torch.tanh(mu)


class Critic(nn.Module):
    """Twin Q-network: returns (Q1, Q2) to reduce overestimation bias."""

    def __init__(self, state_dim, action_dim, hidden=(2048, 2048, 2048)):
        super().__init__()
        self.q1 = MLP(state_dim + action_dim, 1, hidden=hidden)
        self.q2 = MLP(state_dim + action_dim, 1, hidden=hidden)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.q1(x), self.q2(x)
