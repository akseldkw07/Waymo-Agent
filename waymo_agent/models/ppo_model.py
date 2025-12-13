from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


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


class RideShareActorCritic(nn.Module):
    """
    Actor-Critic for:
      prices:     Box(0, inf, (50,))
      reposition: Box(-1, 1, (24,2))
      dispatch:   MultiDiscrete([25]*50, start=[-1]*50)  # we'll emit categories 0..24, then shift to -1..23 if needed
    """

    def __init__(self, hidden: int = 256, max_pending: int = 50, num_veh: int = 24):
        super().__init__()
        self.max_pending = max_pending
        self.num_veh = num_veh
        self.dispatch_n = num_veh + 1  # 25 (includes "no-action")

        # compute input dim from your space
        obs_dim = 5 + 3 + (num_veh * 4) + (max_pending * 9) + (num_veh * 9) + num_veh + max_pending

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # --- policy heads ---
        # prices
        self.price_mu = nn.Linear(hidden, max_pending)
        self.price_logstd = nn.Parameter(torch.full((max_pending,), -0.5))  # learned global log-std per dim

        # reposition (24,2)
        self.repo_mu = nn.Linear(hidden, num_veh * 2)
        self.repo_logstd = nn.Parameter(torch.full((num_veh * 2,), -0.5))

        # dispatch logits (50, 25)
        self.dispatch_logits = nn.Linear(hidden, max_pending * self.dispatch_n)

        # --- baseline ---
        self.value = nn.Linear(hidden, 1)

    def forward(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = _flat_obs(obs)
        h = self.encoder(x)
        out = {}

        out["price_mu"] = self.price_mu(h)  # (..., 50)
        out["repo_mu"] = self.repo_mu(h).view(*h.shape[:-1], self.num_veh, 2)  # (..., 24, 2)
        out["dispatch_logits"] = self.dispatch_logits(h).view(
            *h.shape[:-1], self.max_pending, self.dispatch_n
        )  # (..., 50, 25)

        out["value"] = self.value(h).squeeze(-1)  # (...,)
        return out

    @torch.no_grad()
    def act(self, obs: dict[str, torch.Tensor], deterministic: bool = False) -> dict[str, torch.Tensor]:
        """
        Samples an action dict matching your env.
        If your env expects dispatch in [-1..23] (start=-1), we output that shift.
        """
        out = self.forward(obs)

        # --- prices: Gaussian then squash to [0, inf) via softplus ---
        price_std = out["price_mu"].new_tensor(self.price_logstd).exp()
        price_dist = Normal(out["price_mu"], price_std)
        price_raw = out["price_mu"] if deterministic else price_dist.sample()
        prices = F.softplus(price_raw)  # (..,50), >=0

        # optional: enforce pricing_mask (zero out where mask=0)
        if "pricing_mask" in obs:
            prices = prices * obs["pricing_mask"]

        # --- reposition: Gaussian then clamp to [-1,1] via tanh ---
        repo_std = out["repo_mu"].new_tensor(self.repo_logstd).exp().view(self.num_veh, 2)
        repo_dist = Normal(out["repo_mu"], repo_std)
        repo_raw = out["repo_mu"] if deterministic else repo_dist.sample()
        reposition = torch.tanh(repo_raw)  # (...,24,2)

        # --- dispatch: Categorical per request over 25 choices ---
        logits = out["dispatch_logits"].clone()  # (...,50,25)

        # mask unavailable vehicles (dispatch_mask is (24,))
        # idea: only allow choosing vehicle j if dispatch_mask[j]=1; always allow "no-action" bin
        if "dispatch_mask" in obs:
            veh_mask = obs["dispatch_mask"].to(logits.dtype)  # (...,24)
            # expand to (...,50,24)
            veh_mask = veh_mask.unsqueeze(-2).expand(*logits.shape[:-1], self.num_veh)
            # build full mask (...,50,25) where last index is no-action
            full_mask = torch.cat([veh_mask, torch.ones_like(veh_mask[..., :1])], dim=-1)
            logits = logits + (full_mask.log() - full_mask.log())  # no-op to keep shape
            logits = logits.masked_fill(full_mask <= 0.0, -1e9)

        dispatch_dist = Categorical(logits=logits)
        dispatch_cat = logits.argmax(dim=-1) if deterministic else dispatch_dist.sample()  # (...,50) in [0..24]

        # shift so "no-action" is -1 and vehicles are 0..23 (matches your start=-1 convention)
        dispatch = dispatch_cat - 1  # (...,50) in [-1..23]

        return {
            "prices": prices,
            "reposition": reposition,
            "dispatch": dispatch.to(torch.int64),
        }

    def log_prob_and_entropy(
        self,
        obs: dict[str, torch.Tensor],
        action: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For PPO update: returns (logp, entropy, value)
        logp/entropy are summed across action components.
        """
        out = self.forward(obs)

        # prices
        price_std = out["price_mu"].new_tensor(self.price_logstd).exp()
        price_dist = Normal(out["price_mu"], price_std)
        # action["prices"] is >=0 after softplus; strictly speaking this isn't the same RV.
        # Practical hack: treat it as pre-softplus or use Beta/LogNormal. For now: assume you store pre-softplus in rollout.
        # If you DON'T store pre-softplus, switch to LogNormal or Beta.
        price_logp = price_dist.log_prob(action["prices"]).sum(dim=-1)
        price_ent = price_dist.entropy().sum(dim=-1)

        # reposition
        repo_mu = out["repo_mu"]
        repo_std = repo_mu.new_tensor(self.repo_logstd).exp()
        repo_mu_flat = repo_mu.view(*repo_mu.shape[:-2], self.num_veh * 2)
        repo_dist = Normal(repo_mu_flat, repo_std)
        repo_act_flat = action["reposition"].view(*repo_mu_flat.shape)
        repo_logp = repo_dist.log_prob(repo_act_flat).sum(dim=-1)
        repo_ent = repo_dist.entropy().sum(dim=-1)

        # dispatch
        logits = out["dispatch_logits"]
        dispatch_dist = Categorical(logits=logits)
        dispatch_cat = action["dispatch"] + 1  # unshift back to [0..24]
        disp_logp = dispatch_dist.log_prob(dispatch_cat).sum(dim=-1)
        disp_ent = dispatch_dist.entropy().sum(dim=-1)

        logp = price_logp + repo_logp + disp_logp
        ent = price_ent + repo_ent + disp_ent
        value = out["value"]
        return logp, ent, value
