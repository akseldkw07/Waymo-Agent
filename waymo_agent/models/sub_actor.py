from __future__ import annotations

from abc import abstractmethod
from turtle import forward
import typing as t
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Categorical, LogNormal, Normal


from waymo_agent.constants import DEVICE_TORCH_STR
from waymo_agent.data_classes.space_dicts import ActionDict, ObservationDict
from waymo_agent.graph_env.ENV import RideShareEnv


DEVICE = torch.device(DEVICE_TORCH_STR)


class SubActorNN(nn.Module):
    """ """

    @abstractmethod
    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    @torch.no_grad()
    @abstractmethod
    def act(self, obs: dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        pass

    @abstractmethod
    def log_prob_and_entropy(
        self, obs: dict[str, torch.Tensor], actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass
