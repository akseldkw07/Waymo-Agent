from __future__ import annotations

from dataclasses import dataclass

import torch

from waymo_agent.constants import DEVICE_TORCH_STR
from waymo_agent.data_classes.space_dicts import action_torch_to_numpy
from waymo_agent.graph_env.ENV import RideShareEnv
from waymo_agent.models.torch_np_utils import obs_pd_to_torch

DEVICE = torch.device(DEVICE_TORCH_STR)

from .ppo_model import RideShareActorCritic
from tqdm.auto import tqdm


@dataclass
class PPOTrainConfig:
    """Minimal PPO config tuned for your RideShareEnv shapes."""

    total_steps: int = 20_000
    debug: bool = False

    gamma: float = 0.997
    gae_lambda: float = 0.95

    lr: float = 3e-4
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5

    update_epochs: int = 5
    minibatch_size: int = 256

    deterministic_eval: bool = True
    log_every_updates: int = 5
    eval_episodes: int = 2

    # Early stopping
    eval_every_updates: int = 5
    patience: int = 20
    min_delta: float = 1e-3
    save_best: bool = True


PPO_TRN_CFG_TS5k_Debug_EvalEp2 = PPOTrainConfig(total_steps=5_000, eval_every_updates=1, eval_episodes=2, debug=True)

# =========================================================
# PPO training loop for RideShareEnv
# =========================================================


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
    eval_bar = tqdm(total=episodes, desc="PPO evaluation episodes", leave=False, position=1, unit="episodes")
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
        eval_bar.update(1)
    eval_bar.close()
    return total / float(episodes)
