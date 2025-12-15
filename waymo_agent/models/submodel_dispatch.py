from __future__ import annotations

import typing as t

import torch
import torch.nn as nn
from torch.distributions import Categorical

from waymo_agent.models.submodel_base import SubActorHeadNN


class DispatchHead(SubActorHeadNN):
    """Categorical dispatch head.

    Produces one dispatch decision per pending request.

    Action encoding:
      - Model samples categories in {0, 1, ..., num_veh,} of size (num_veh + 1)
      - The last category (index = num_veh) corresponds to **NO-ACTION**.
      - Externally (env/action dict), we encode NO-ACTION as -1 and vehicles as 0..num_veh-1.

    Masking:
      - Uses obs['dispatch_mask'] (shape (num_veh,) or (B, num_veh)) to disallow assigning
        requests to non-idle vehicles.
      - If obs provides an additional per-request mask (e.g. 'dispatchable_mask'), we will
        also use it to exclude non-dispatchable requests from logp/entropy sums.

    Shapes:
      - logits: (B, max_pending, num_veh+1)
      - action: (B, max_pending) or (max_pending,)
    """

    def __init__(self, hidden: int, max_pending: int, num_veh: int, invalid_logit: float = -1e9):
        super().__init__()
        self.max_pending = int(max_pending)
        self.num_veh = int(num_veh)
        self.n_cat = self.num_veh + 1  # +1 for NO-ACTION
        self.invalid_logit = float(invalid_logit)

        self.logits_layer = nn.Linear(hidden, self.max_pending * self.n_cat)

    def _masked_logits(self, h: torch.Tensor, obs: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        # h: (hidden,) or (B, hidden)
        if h.ndim == 1:
            h = h.unsqueeze(0)

        B = h.shape[0]
        logits = self.logits_layer(h).view(B, self.max_pending, self.n_cat)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        # Vehicle availability mask: disallow non-idle vehicles
        if obs is not None and "dispatch_mask" in obs:
            vm = obs["dispatch_mask"]
            # vm expected 1 for idle/available, 0 for busy/unavailable
            if vm.ndim == 1:
                vm = vm.unsqueeze(0)  # (1, num_veh)
            vm = vm.to(device=logits.device, dtype=logits.dtype)

            # Build mask over categories (B, 1, n_cat)
            # Vehicles 0..num_veh-1 correspond to vm; NO-ACTION category always allowed.
            cat_mask = torch.ones((B, 1, self.n_cat), device=logits.device, dtype=logits.dtype)
            cat_mask[:, :, : self.num_veh] = vm.unsqueeze(1)  # (B,1,num_veh)

            logits = torch.where(cat_mask > 0.0, logits, logits.new_full((), self.invalid_logit))

        return logits

    def dist(self, h: torch.Tensor, obs: dict[str, torch.Tensor] | None = None):
        logits = self._masked_logits(h, obs)
        return Categorical(logits=logits)

    @staticmethod
    def _to_env_action(cat: torch.Tensor, num_veh: int) -> torch.Tensor:
        """Map categorical index -> env action.

        cat in [0..num_veh] where cat==num_veh means NO-ACTION.
        returns int64 in [-1..num_veh-1].
        """
        cat = cat.to(dtype=torch.int64)
        out = cat.clone()
        out[out == num_veh] = -1
        return out

    @staticmethod
    def _from_env_action(a: torch.Tensor, num_veh: int) -> torch.Tensor:
        """Map env action -> categorical index.

        env action in [-1..num_veh-1] where -1 means NO-ACTION.
        returns int64 in [0..num_veh].
        """
        a = a.to(dtype=torch.int64)
        cat = a.clone()
        cat[cat < 0] = num_veh
        return cat

    @torch.no_grad()
    def act(self, h: torch.Tensor, obs: dict[str, torch.Tensor] | None = None, deterministic: bool = False):
        d = self.dist(h, obs)
        if deterministic:
            # argmax over categories
            logits = t.cast(torch.Tensor, d.logits)  # (B, max_pending, n_cat)
            cat = logits.argmax(dim=-1)
        else:
            cat = d.sample()

        a = self._to_env_action(cat, self.num_veh)

        # return (max_pending,) if input had no batch
        return a.squeeze(0) if h.ndim == 1 else a

    def log_prob_and_entropy(self, h: torch.Tensor, action: torch.Tensor, obs: dict[str, torch.Tensor] | None = None):
        d = self.dist(h, obs)

        a = action
        if a.ndim == 1:
            a = a.unsqueeze(0)  # (1, max_pending)
        a = a.to(device=d.logits.device, dtype=torch.int64)

        cat = self._from_env_action(a, self.num_veh)  # (B, max_pending)

        # logp/entropy per request
        logp_elem = d.log_prob(cat)  # (B, max_pending)
        ent_elem = d.entropy()  # (B, max_pending)

        # Optional per-request mask: if present, only include dispatchable requests
        # (e.g. those in ACCEPTED state). This reduces gradient noise.
        if obs is not None:
            for key in ("dispatchable_mask", "dispatch_req_mask", "request_dispatch_mask"):
                if key in obs:
                    rm = obs[key]
                    if rm.ndim == 1:
                        rm = rm.unsqueeze(0)
                    rm = rm.to(device=logp_elem.device, dtype=logp_elem.dtype)
                    logp_elem = logp_elem * rm
                    ent_elem = ent_elem * rm
                    break

        logp = logp_elem.sum(dim=-1)  # (B,)
        ent = ent_elem.sum(dim=-1)  # (B,)

        return logp, ent
