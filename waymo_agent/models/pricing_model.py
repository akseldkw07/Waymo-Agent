from __future__ import annotations

import typing as t
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Categorical, LogNormal, Normal


from waymo_agent.constants import DEVICE_TORCH_STR
from waymo_agent.data_classes.space_dicts import ActionDict, ObservationDict
from waymo_agent.graph_env.ENV import RideShareEnv
from waymo_agent.models.sub_actor import SubActorNN

DEVICE = torch.device(DEVICE_TORCH_STR)


class PricingModel(SubActorNN):
    """
    Actor-Critic for:
      prices:     Box(0, inf, (50,))
    """

    def __init__(self, env: RideShareEnv, hidden: int = 256):
        super().__init__()
        self.max_pending = env.config.max_pending_requests
        self.num_veh = env.num_vehicles
        self.dispatch_n = self.num_veh + 1  # 25 (includes "no-action")
        self.obs_space = t.cast(ObservationDict, env.observation_space)
        self.action_space = t.cast(ActionDict, env.action_space)

    def forward(self, obs: dict[str, torch.Tensor]):
        """
        Take observation dict and return pricing
        """
        raise NotImplementedError("PricingModel.forward is not implemented yet.")

    @torch.no_grad()
    def act(self, obs: dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        """
        Take observation dict and return pricing
        """
        raise NotImplementedError("PricingModel.act is not implemented yet.")

    def log_prob_and_entropy(
        self, obs: dict[str, torch.Tensor], actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Take observation dict and actions and return log probs and entropy
        """
        raise NotImplementedError("PricingModel.log_prob_and_entropy is not implemented yet.")
