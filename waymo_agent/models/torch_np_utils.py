from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from waymo_agent.constants import DEVICE_TORCH_STR
from waymo_agent.data_classes.enriched_df_base import EnrichedDF
from waymo_agent.data_classes.space_dicts import ObservationDict

DEVICE = torch.device(DEVICE_TORCH_STR)


def save_weights(model: torch.nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_weights(model: torch.nn.Module, path: str | Path, device: str | torch.device = "mps") -> torch.nn.Module:
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


def stack_dict(buf: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack list of dict[tensor] into dict[tensor] with leading time/batch dim."""
    keys = buf[0].keys()
    return {k: torch.stack([b[k] for b in buf], dim=0) for k in keys}


def obs_pd_to_torch(obs: ObservationDict) -> dict[str, torch.Tensor]:
    """Convert env obs (numpy | pd.DataFrame | EnrichedDF) -> finite float32 tensors on DEVICE."""
    ret: dict[str, torch.Tensor] = {}

    for k, v in obs.items():
        # Preferred path: EnrichedDF knows how to produce a numeric training view
        if isinstance(v, EnrichedDF):
            arr = v.to_obs_numpy()

        # Raw pandas df (should be rare if prune_obs_dict_gymnasium is used)
        elif isinstance(v, pd.DataFrame):
            arr = v.to_numpy()
        else:
            # Already a numpy array from prune_obs_dict_gymnasium
            arr = v

        try:
            tns = torch.as_tensor(arr, device=DEVICE, dtype=torch.float32)
        except Exception as e:
            dtype_str = getattr(arr, "dtype", type(arr))
            raise ValueError(f"Failed to convert obs key '{k}' with dtype/type {dtype_str} to tensor.") from e

        # Critical: PPO must never see NaN/inf features
        tns = torch.nan_to_num(tns, nan=0.0, posinf=0.0, neginf=0.0)

        ret[k] = tns

    return ret
