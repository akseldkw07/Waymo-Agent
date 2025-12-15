from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.distributions import TransformedDistribution, ExponentialFamily, Categorical


class SubActorHeadNN(nn.Module, ABC):
    @abstractmethod
    def dist(
        self, h: torch.Tensor, obs: dict[str, torch.Tensor] | None = None
    ) -> TransformedDistribution | ExponentialFamily | Categorical:
        """Return a torch Distribution (or something equivalent) parameterized by h."""
        raise NotImplementedError

    @torch.no_grad()
    def act(self, h: torch.Tensor, obs: dict[str, torch.Tensor] | None = None, deterministic: bool = False):
        d = self.dist(h, obs)
        return d.mode if deterministic and hasattr(d, "mode") else d.sample()

    @abstractmethod
    def log_prob_and_entropy(self, h: torch.Tensor, action: torch.Tensor, obs: dict[str, torch.Tensor] | None = None):
        raise NotImplementedError


class SharedEncoder(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        h = self.net(x)
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        return h
