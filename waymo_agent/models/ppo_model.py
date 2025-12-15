from __future__ import annotations

import typing as t
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Normal

from waymo_agent.constants import DEVICE_TORCH_STR
from waymo_agent.data_classes.space_dicts import ActionDict, ObservationDict
from waymo_agent.graph_env.ENV import RideShareEnv

from .pricing_model import PricingHead
from .sub_actor import SharedEncoder
from .torch_np_utils import _flat_obs

DEVICE = torch.device(DEVICE_TORCH_STR)


class TanhNormal:
    """
    Squashed Normal distribution: a = tanh(z),  z ~ Normal(mu, sigma).

    We use the change-of-variables correction for log_prob:
      log p(a) = log p(z) - sum log(1 - tanh(z)^2)
    where z = atanh(a).

    Entropy has no clean closed-form; return None and treat entropy bonus as 0
    for this head (or approximate via Monte Carlo later).
    """

    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor, eps: float = 1e-6):
        self.mu = mu
        self.sigma = sigma
        self.base = Normal(mu, sigma)
        self.eps = eps

    def sample(self) -> torch.Tensor:
        z = self.base.rsample()
        return torch.tanh(z)

    def mode(self) -> torch.Tensor:
        return torch.tanh(self.mu)

    def log_prob(self, a: torch.Tensor) -> torch.Tensor:
        a = a.clamp(-1.0 + self.eps, 1.0 - self.eps)
        z = 0.5 * (torch.log1p(a) - torch.log1p(-a))  # atanh(a)
        log_pz = self.base.log_prob(z)
        log_det = -torch.log(1.0 - a * a + self.eps)  # -log(1-a^2)
        return log_pz + log_det

    def entropy(self):
        return None


@dataclass
class ActorCriticConfig: ...


class RideShareActorCritic(nn.Module):
    def __init__(self, env: RideShareEnv, hidden: int = 256):
        super().__init__()
        self.max_pending = env.config.max_pending_requests
        self.num_veh = env.num_vehicles
        self.dispatch_n = self.num_veh + 1  # 25 (includes "no-action")
        self.obs_space = t.cast(ObservationDict, env.observation_space)
        self.action_space = t.cast(ActionDict, env.action_space)
        self.dispatch_n = self.num_veh + 1  # 25 (includes "no-action")

        size_globals: int = self.obs_space["globals"].shape[0]  # 5
        size_sd_ratio: int = self.obs_space["supply_demand_ratio"].shape[0]  # 3

        # compute input dim from size_sd_ratio + sz_cars + sz_reqs + sz_rides + sz_dispatch_mask + sz_pricing_mask size_sd_ratio = self.obs_space["supply_demand_ratio"].shape[0]  # 3
        sz_cars = self.obs_space["vehicles"].shape[0] * self.obs_space["vehicles"].shape[1]  # 24*4
        sz_reqs = self.obs_space["pending_requests"].shape[0] * self.obs_space["pending_requests"].shape[1]  # 50*9
        sz_rides = self.obs_space["active_rides"].shape[0] * self.obs_space["active_rides"].shape[1]  # 24*9
        sz_dispatch_mask: int = self.obs_space["dispatch_mask"].shape[0]  # 24
        sz_pricing_mask: int = self.obs_space["pricing_mask"].shape[0]  # 50

        obs_dim = size_globals + size_sd_ratio + sz_cars + sz_reqs + sz_rides + sz_dispatch_mask + sz_pricing_mask

        self.encoder = SharedEncoder(obs_dim, hidden)

        self.pricing = PricingHead(hidden, self.max_pending)

        # placeholders for your next heads
        self.reposition = RepositionHead(hidden, self.num_veh)  # you’ll implement similarly
        self.dispatch = DispatchHead(hidden, self.max_pending, self.num_veh)  # masking lives here

        self.value = nn.Linear(hidden, 1)

    def forward(self, obs: dict[str, torch.Tensor]):
        x = _flat_obs(obs)
        h = self.encoder(x)
        v = self.value(h).squeeze(-1)
        return h, v

    @torch.no_grad()
    def act(self, obs: dict[str, torch.Tensor], deterministic: bool = False):
        h, _v = self.forward(obs)
        return {
            "prices": self.pricing.act(h, obs, deterministic=deterministic),
            "reposition": self.reposition.act(h, obs, deterministic=deterministic),
            "dispatch": self.dispatch.act(h, obs, deterministic=deterministic),
        }

    def log_prob_and_entropy(self, obs: dict[str, torch.Tensor], action: dict[str, torch.Tensor]):
        h, v = self.forward(obs)

        lp_p, ent_p = self.pricing.log_prob_and_entropy(h, action["prices"], obs)
        lp_r, ent_r = self.reposition.log_prob_and_entropy(h, action["reposition"], obs)
        lp_d, ent_d = self.dispatch.log_prob_and_entropy(h, action["dispatch"], obs)

        logp = lp_p + lp_r + lp_d
        ent = ent_p + ent_r + ent_d
        return logp, ent, v
