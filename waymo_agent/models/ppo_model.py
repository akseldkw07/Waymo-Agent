from __future__ import annotations

import typing as t
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Categorical, LogNormal, Normal


from waymo_agent.constants import DEVICE_TORCH_STR
from waymo_agent.data_classes.space_dicts import ActionDict, ObservationDict
from waymo_agent.graph_env.ENV import RideShareEnv


DEVICE = torch.device(DEVICE_TORCH_STR)


def save_weights(model: torch.nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_weights(model: torch.nn.Module, path: str | Path, device: str | torch.device = "cpu") -> torch.nn.Module:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    return model


def _flat_obs(obs: dict[str, torch.Tensor]) -> torch.Tensor:
    """
    obs keys (per your space):
      globals:            (5,)
      supply_demand_ratio:(3,)
      vehicles:           (24,4)
      pending_requests:   (50,9)
      active_rides:       (24,9)
      dispatch_mask:      (24,)
      pricing_mask:       (50,)
    Returns: flat vector (batch, D) or (D,) if unbatched.
    """
    parts = [
        obs["globals"].reshape(obs["globals"].shape[:-1] + (-1,)),
        obs["supply_demand_ratio"].reshape(obs["supply_demand_ratio"].shape[:-1] + (-1,)),
        obs["vehicles"].reshape(obs["vehicles"].shape[:-2] + (-1,)),
        obs["pending_requests"].reshape(obs["pending_requests"].shape[:-2] + (-1,)),
        obs["active_rides"].reshape(obs["active_rides"].shape[:-2] + (-1,)),
        obs["dispatch_mask"].reshape(obs["dispatch_mask"].shape[:-1] + (-1,)),
        obs["pricing_mask"].reshape(obs["pricing_mask"].shape[:-1] + (-1,)),
    ]
    return torch.cat(parts, dim=-1)


def _to_device(d: dict[str, torch.Tensor], device: torch.device = DEVICE) -> dict[str, torch.Tensor]:
    """Move all tensors in a dict to the desired device."""
    return {k: v.to(device, non_blocking=True) for k, v in d.items()}


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


class RideShareActorCritic(nn.Module):
    """
    Actor-Critic for:
      prices:     Box(0, inf, (50,))
      reposition: Box(-1, 1, (24,2))
      dispatch:   MultiDiscrete([25]*50, start=[-1]*50)  # emit categories 0..24, then shift to -1..23
    """

    def __init__(self, env: RideShareEnv, hidden: int = 256):
        super().__init__()
        self.max_pending = env.config.max_pending_requests
        self.num_veh = env.num_vehicles
        self.dispatch_n = self.num_veh + 1  # 25 (includes "no-action")
        self.obs_space = t.cast(ObservationDict, env.observation_space)
        self.action_space = t.cast(ActionDict, env.action_space)

        # compute input dim from your space
        size_globals = self.obs_space["globals"].shape[0]  # 5
        size_sd_ratio = self.obs_space["supply_demand_ratio"].shape[0]  # 3
        sz_cars = self.obs_space["vehicles"].shape[0] * self.obs_space["vehicles"].shape[1]  # 24*4
        sz_reqs = self.obs_space["pending_requests"].shape[0] * self.obs_space["pending_requests"].shape[1]  # 50*9
        sz_rides = self.obs_space["active_rides"].shape[0] * self.obs_space["active_rides"].shape[1]  # 24*9
        sz_dispatch_mask = self.obs_space["dispatch_mask"].shape[0]  # 24
        sz_pricing_mask = self.obs_space["pricing_mask"].shape[0]  # 50

        obs_dim = size_globals + size_sd_ratio + sz_cars + sz_reqs + sz_rides + sz_dispatch_mask + sz_pricing_mask

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # --- policy heads ---
        # prices
        self.price_mu = nn.Linear(hidden, self.max_pending)
        self.price_logstd = nn.Parameter(torch.full((self.max_pending,), -0.5))  # learned global log-std per dim

        # reposition (num_veh,2)
        self.repo_mu = nn.Linear(hidden, self.num_veh * 2)
        self.repo_logstd = nn.Parameter(torch.full((self.num_veh * 2,), -0.5))

        # dispatch logits (max_pending, num_veh+1)
        self.dispatch_logits = nn.Linear(hidden, self.max_pending * self.dispatch_n)

        # --- baseline ---
        self.value = nn.Linear(hidden, 1)
        self.to(DEVICE)

    def forward(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = _flat_obs(obs)
        print(f"X:{x}\n\n")
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"X:{x}\n\n")
        h = self.encoder(x)
        print(f"H:{h}\n\n")
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"H:{h}\n\n")

        out: dict[str, torch.Tensor] = {}

        out["price_mu"] = self.price_mu(h)  # (..., 50)
        out["repo_mu"] = self.repo_mu(h).view(*h.shape[:-1], self.num_veh, 2)  # (..., 24, 2)
        out["dispatch_logits"] = self.dispatch_logits(h).view(
            *h.shape[:-1], self.max_pending, self.dispatch_n
        )  # (..., 50, 25)
        print(f"DISPATCH: {out['dispatch_logits']}\n\n")

        out["value"] = self.value(h).squeeze(-1)  # (...,)
        return out

    @torch.no_grad()
    def act(self, obs: dict[str, torch.Tensor], deterministic: bool = False) -> dict[str, torch.Tensor]:
        """
        Samples an action dict matching your env.
        If env expects dispatch in [-1..23] (start=-1), we output that shift.
        """
        obs = _to_device(obs)

        out = self.forward(obs)
        # Safety: if anything upstream produced NaNs, zero them out before building dists
        for kk in ("price_mu", "repo_mu", "dispatch_logits", "value"):
            out[kk] = torch.nan_to_num(out[kk], nan=0.0, posinf=0.0, neginf=0.0)

        # --- prices: LogNormal ensures positivity with correct log-prob ---
        eps_price = 1.0

        price_mu = out["price_mu"].clamp(min=-2.0, max=10.0)
        price_sigma = torch.exp(self.price_logstd.detach().clone()).clamp(min=1e-4, max=10.0)
        price_dist = LogNormal(price_mu, price_sigma)

        if deterministic:
            prices = torch.exp(price_mu.detach().clone())  # median (stable)
        else:
            prices = price_dist.rsample()  # rsample for PPO friendliness

        prices = torch.nan_to_num(prices, nan=eps_price, posinf=1e6, neginf=eps_price)
        prices = prices.clamp(min=eps_price)

        if "pricing_mask" in obs:
            pm = obs["pricing_mask"].to(device=prices.device, dtype=prices.dtype)
            # IMPORTANT: masked dims become eps (not 0) so log_prob is defined
            prices = torch.where(pm > 0.0, prices, prices.new_full(prices.shape, eps_price))

        # --- reposition: tanh-squashed Normal matches Box([-1,1]) ---
        repo_sigma = torch.exp(self.repo_logstd.detach().clone()).view(self.num_veh, 2)
        repo_dist = TanhNormal(out["repo_mu"], repo_sigma)
        reposition = repo_dist.mode() if deterministic else repo_dist.sample()  # (..., 24, 2)

        # --- dispatch: Categorical per request over (num_veh+1) choices ---
        logits = out["dispatch_logits"].clone()  # (..., 50, 25)

        # mask unavailable vehicles (dispatch_mask is (..., 24))
        # allow "no-action" always (the last column)
        if "dispatch_mask" in obs:
            veh_mask = obs["dispatch_mask"].to(device=logits.device, dtype=logits.dtype)  # (..., 24)
            # expand to (..., 50, 24) using batch dims only
            veh_mask = veh_mask.unsqueeze(-2).expand(*logits.shape[:-2], self.max_pending, self.num_veh)
            full_mask = torch.cat([veh_mask, torch.ones_like(veh_mask[..., :1])], dim=-1)  # (..., 50, 25)
            logits = logits.masked_fill(full_mask <= 0.0, -1e9)

        dispatch_dist = Categorical(logits=logits)
        dispatch_cat = logits.argmax(dim=-1) if deterministic else dispatch_dist.sample()  # (..., 50) in [0..24]

        # shift so "no-action" is -1 and vehicles are 0..23 (matches start=-1 convention)
        dispatch = dispatch_cat - 1  # (..., 50) in [-1..23]

        return {
            "prices": prices,
            "reposition": reposition,
            "dispatch": dispatch.to(torch.int64),
        }

    def log_prob_and_entropy(
        self, obs: dict[str, torch.Tensor], action: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For PPO update: returns (logp, entropy, value)
        logp/entropy are summed across action components.
        """
        obs = _to_device(obs)
        action = _to_device(action)
        out = self.forward(obs)
        # Safety: if anything upstream produced NaNs, zero them out before building dists
        for kk in ("price_mu", "repo_mu", "dispatch_logits", "value"):
            out[kk] = torch.nan_to_num(out[kk], nan=0.0, posinf=0.0, neginf=0.0)

        # prices (LogNormal): consistent with act()
        eps_price = 1e-6

        price_mu = out["price_mu"].clamp(min=-10.0, max=10.0)
        price_sigma = torch.exp(self.price_logstd.detach()).clamp(min=1e-4, max=10.0)
        price_dist = LogNormal(price_mu, price_sigma)

        prices_a = action["prices"].to(device=price_mu.device, dtype=price_mu.dtype)
        prices_a = torch.nan_to_num(prices_a, nan=eps_price, posinf=1e6, neginf=eps_price)
        prices_a = prices_a.clamp(min=eps_price)

        if "pricing_mask" in obs:
            pm = obs["pricing_mask"].to(device=prices_a.device, dtype=prices_a.dtype)
            price_logp = (price_dist.log_prob(prices_a) * pm).sum(dim=-1)
            price_ent = (price_dist.entropy() * pm).sum(dim=-1)
        else:
            price_logp = price_dist.log_prob(prices_a).sum(dim=-1)
            price_ent = price_dist.entropy().sum(dim=-1)

        # reposition (tanh-squashed Normal): consistent with act()
        repo_mu = out["repo_mu"]
        repo_sigma = torch.exp(self.repo_logstd).detach().view(self.num_veh, 2)
        repo_dist = TanhNormal(repo_mu, repo_sigma)

        repo_logp = repo_dist.log_prob(action["reposition"]).sum(dim=(-2, -1))
        repo_ent = repo_logp.new_zeros(repo_logp.shape)

        # dispatch
        logits = out["dispatch_logits"].clone()

        if "dispatch_mask" in obs:
            veh_mask = obs["dispatch_mask"].to(device=logits.device, dtype=logits.dtype)  # (..., 24)
            veh_mask = veh_mask.unsqueeze(-2).expand(
                *logits.shape[:-2], self.max_pending, self.num_veh
            )  # (..., 50, 24)
            full_mask = torch.cat([veh_mask, torch.ones_like(veh_mask[..., :1])], dim=-1)  # (..., 50, 25)
            logits = logits.masked_fill(full_mask <= 0.0, -1e9)

        dispatch_dist = Categorical(logits=logits)
        dispatch_cat = action["dispatch"] + 1  # unshift back to [0..24]
        disp_logp = dispatch_dist.log_prob(dispatch_cat).sum(dim=-1)
        disp_ent = dispatch_dist.entropy().sum(dim=-1)

        logp = price_logp + repo_logp + disp_logp
        ent = price_ent + repo_ent + disp_ent
        value = out["value"]
        return logp, ent, value
