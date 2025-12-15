from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import LogNormal

from waymo_agent.data_classes.config import EnvConfig
from waymo_agent.models.submodel_base import SubActorHeadNN


class PricingHead(SubActorHeadNN):
    def __init__(self, hidden: int, cfg: EnvConfig, init_logstd: float = -0.5):
        super().__init__()
        self.max_pending = cfg.max_pending_requests
        self.mu = nn.Linear(hidden, self.max_pending)
        self.logstd = nn.Parameter(torch.full((self.max_pending,), init_logstd))
        # “masked” prices should still be valid for log_prob
        self.eps_price = 1.0

    def dist(self, h: torch.Tensor, obs: dict[str, torch.Tensor] | None = None):
        mu = self.mu(h).clamp(min=-2.0, max=10.0)  # stabilize
        sigma = torch.exp(self.logstd).clamp(1e-4, 10.0)  # stabilize
        return LogNormal(mu, sigma)

    @torch.no_grad()
    def act(self, h: torch.Tensor, obs: dict[str, torch.Tensor] | None = None, deterministic: bool = False):
        d = self.dist(h, obs)
        if deterministic:
            # LogNormal median = exp(mu)
            a = torch.exp(self.mu(h).clamp(min=-2.0, max=10.0))
        else:
            a = d.rsample()

        a = torch.nan_to_num(a, nan=self.eps_price, posinf=1e6, neginf=self.eps_price)
        a = a.clamp(min=self.eps_price)

        if obs is not None and "pricing_mask" in obs:
            pm = obs["pricing_mask"].to(device=a.device, dtype=a.dtype)  # (..., 50)
            a = torch.where(pm > 0.0, a, a.new_full(a.shape, self.eps_price))

        return a

    def log_prob_and_entropy(self, h: torch.Tensor, action: torch.Tensor, obs: dict[str, torch.Tensor] | None = None):
        d = self.dist(h, obs)
        a = action.to(device=h.device, dtype=torch.float32)
        a = torch.nan_to_num(a, nan=1e-6, posinf=1e6, neginf=1e-6).clamp(min=1e-6)

        logp = d.log_prob(a)  # (..., 50)
        ent = d.entropy()  # (..., 50)

        if obs is not None and "pricing_mask" in obs:
            pm = obs["pricing_mask"].to(device=logp.device, dtype=logp.dtype)
            logp = (logp * pm).sum(dim=-1)
            ent = (ent * pm).sum(dim=-1)
        else:
            logp = logp.sum(dim=-1)
            ent = ent.sum(dim=-1)

        return logp, ent
