from __future__ import annotations

import typing as t
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from waymo_agent.constants import DEVICE_TORCH_STR
from waymo_agent.data_classes.enriched_df_base import EnrichedDF
from waymo_agent.data_classes.space_dicts import ObservationDict
from waymo_agent.graph_env.ENV import RideShareEnv

DEVICE = torch.device(DEVICE_TORCH_STR)
if t.TYPE_CHECKING:
    from .ppo_model import RideShareActorCritic


@dataclass
class PPOTrainConfig:
    """Minimal PPO config tuned for your RideShareEnv shapes."""

    total_steps: int = 2_000

    gamma: float = 0.997
    gae_lambda: float = 0.95

    lr: float = 3e-4
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5

    update_epochs: int = 4
    minibatch_size: int = 256

    deterministic_eval: bool = True
    log_every_updates: int = 5
    eval_episodes: int = 2

    # Early stopping
    eval_every_updates: int = 5
    patience: int = 20
    min_delta: float = 1e-3
    save_best: bool = True


# =========================================================
# PPO training loop for RideShareEnv
# =========================================================


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


def action_torch_to_numpy(action: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Convert model action (torch) -> env action (numpy) with expected dtypes."""
    return {
        "prices": action["prices"].detach().cpu().numpy().astype(np.float64),
        "reposition": action["reposition"].detach().cpu().numpy().astype(np.float32),
        "dispatch": action["dispatch"].detach().cpu().numpy().astype(np.int64),
    }


def stack_dict(buf: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack list of dict[tensor] into dict[tensor] with leading time/batch dim."""
    keys = buf[0].keys()
    return {k: torch.stack([b[k] for b in buf], dim=0) for k in keys}


def compute_gae(
    rewards: torch.Tensor,  # (T,)
    values: torch.Tensor,  # (T+1,)
    dones: torch.Tensor,  # (T,) 1.0 if done else 0.0
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE-Lambda: returns (advantages (T,), returns (T,))."""
    T = rewards.shape[0]
    adv = torch.zeros(T, device=rewards.device, dtype=torch.float32)
    gae = torch.zeros((), device=rewards.device, dtype=torch.float32)

    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae

    returns = adv + values[:-1]
    return adv, returns


@torch.no_grad()
def evaluate_policy(
    env: RideShareEnv,
    model: RideShareActorCritic,
    episodes: int = 2,
    deterministic: bool = True,
) -> float:
    """Average episodic reward (undiscounted)."""
    model.eval()
    total = 0.0
    for _ in range(episodes):
        obs_np, _ = env.reset()
        done = False
        ep = 0.0
        while not done:
            obs_t = obs_pd_to_torch(obs_np)
            act_t = model.act(obs_t, deterministic=deterministic)
            obs_np, r, term, trunc, _info = env.step(action_torch_to_numpy(act_t))  # type: ignore[arg-type]
            ep += float(r)
            done = bool(term) or bool(trunc)
        total += ep
    return total / float(episodes)
