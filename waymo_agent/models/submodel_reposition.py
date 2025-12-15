from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from waymo_agent.models.submodel_base import SubActorHeadNN


class RepositionHead(SubActorHeadNN):
    """
    Outputs reposition targets in [-1, 1] for each vehicle:
      action shape: (B, num_veh, 2)

    Uses tanh-squashed Normal:
      z ~ Normal(mu, sigma)
      a = tanh(z)

    Masking:
      If obs contains "dispatch_mask" (shape (B, num_veh) or (num_veh,)),
      we zero-out actions for masked (non-idle) vehicles and exclude them from logp/entropy sums.
    """

    def __init__(self, hidden: int, num_veh: int, init_logstd: float = -1.0, eps: float = 1e-6):
        super().__init__()
        self.num_veh = int(num_veh)
        self.eps = float(eps)

        self.mu = nn.Linear(hidden, self.num_veh * 2)
        self.logstd = nn.Parameter(torch.full((self.num_veh, 2), float(init_logstd)))

    def _dist_params(self, h: torch.Tensor):
        # h: (B, hidden) or (hidden,)
        if h.ndim == 1:
            h = h.unsqueeze(0)

        mu: torch.Tensor = self.mu(h).view(h.shape[0], self.num_veh, 2)
        mu = mu.clamp(min=-5.0, max=5.0)  # stabilize logits a bit

        sigma = torch.exp(self.logstd).clamp(min=1e-4, max=10.0)  # (num_veh, 2)
        sigma = sigma.unsqueeze(0).expand_as(mu)  # (B, num_veh, 2)

        return mu, sigma

    def dist(self, h: torch.Tensor, obs: dict[str, torch.Tensor] | None = None):
        mu, sigma = self._dist_params(h)
        return Normal(mu, sigma)

    @torch.no_grad()
    def act(self, h: torch.Tensor, obs: dict[str, torch.Tensor] | None = None, deterministic: bool = False):
        base = self.dist(h, obs)
        z = base.mean if deterministic else base.rsample()
        a = torch.tanh(z)

        a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)

        # mask out non-idle vehicles
        if obs is not None and "dispatch_mask" in obs:
            m = obs["dispatch_mask"]
            if m.ndim == 1:
                m = m.unsqueeze(0)  # (1, num_veh)
            m = m.to(device=a.device, dtype=a.dtype)
            a = a * m.unsqueeze(-1)  # (B, num_veh, 2)

        # return (num_veh,2) if no batch in, else (B,num_veh,2)
        return a.squeeze(0) if h.ndim == 1 else a

    def log_prob_and_entropy(self, h: torch.Tensor, action: torch.Tensor, obs: dict[str, torch.Tensor] | None = None):
        base = self.dist(h, obs)  # Normal over z
        if h.ndim == 1:
            pass
        else:
            h.shape[0]

        a = action
        if a.ndim == 2:
            a = a.unsqueeze(0)  # (1, num_veh, 2)
        a = a.to(device=base.mean.device, dtype=torch.float32)
        a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0 + self.eps, 1.0 - self.eps)

        # inverse tanh: atanh(a)
        z = 0.5 * (torch.log1p(a) - torch.log1p(-a))  # (B, num_veh, 2)

        # log p(a) = log p(z) + log|dz/da|
        # dz/da = 1 / (1 - a^2)
        log_pz: torch.Tensor = base.log_prob(z)  # (B, num_veh, 2)
        log_det = -torch.log(1.0 - a * a + self.eps)  # (B, num_veh, 2)
        logp_elem: torch.Tensor = log_pz + log_det

        # entropy: no clean closed form after tanh; use base entropy as a proxy (works fine for debugging)
        ent_elem = base.entropy()

        # apply mask: only idle vehicles contribute
        if obs is not None and "dispatch_mask" in obs:
            m = obs["dispatch_mask"]
            if m.ndim == 1:
                m = m.unsqueeze(0)
            m = m.to(device=logp_elem.device, dtype=logp_elem.dtype)  # (B, num_veh)
            logp: torch.Tensor = (logp_elem * m.unsqueeze(-1)).sum(dim=(-1, -2))  # sum over veh,xy
            ent = (ent_elem * m.unsqueeze(-1)).sum(dim=(-1, -2))
        else:
            logp = logp_elem.sum(dim=(-1, -2))
            ent = ent_elem.sum(dim=(-1, -2))

        # shapes: (B,)
        return logp, ent
