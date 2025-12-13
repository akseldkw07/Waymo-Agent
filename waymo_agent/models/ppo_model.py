from __future__ import annotations

import typing as t
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Categorical, LogNormal, Normal

from waymo_agent.constants import DEVICE_TORCH_STR
from waymo_agent.data_classes.enriched_df_base import EnrichedDF
from waymo_agent.data_classes.space_dicts import ActionDict, ObservationDict
from waymo_agent.graph_env.ENV import RideShareEnv

DEVICE = torch.device(DEVICE_TORCH_STR)


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
        obs = _to_device(obs)
        x = _flat_obs(obs)
        h = self.encoder(x)
        out: dict[str, torch.Tensor] = {}

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
        If env expects dispatch in [-1..23] (start=-1), we output that shift.
        """
        obs = _to_device(obs)

        out = self.forward(obs)

        # --- prices: LogNormal ensures positivity with correct log-prob ---
        eps_price = 1e-6

        price_mu = out["price_mu"].clamp(min=-10.0, max=10.0)
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


# =========================================================
# PPO training loop for RideShareEnv
# =========================================================


@dataclass
class PPOTrainConfig:
    """Minimal PPO config tuned for your RideShareEnv shapes."""

    total_steps: int = 200_000
    rollout_len: int = 512

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


def obs_pd_to_torch(obs: ObservationDict) -> dict[str, torch.Tensor]:
    """Convert env obs (pd.DataFrame | EnrichedDF) -> torch tensors on DEVICE."""
    ret = {}

    for k, v in obs.items():
        if isinstance(v, EnrichedDF):
            v = v.to_obs_numpy()

        elif isinstance(v, pd.DataFrame):
            v = v.to_numpy()

        try:
            ret[k] = torch.as_tensor(v, device=DEVICE, dtype=torch.float32)
        except Exception as e:
            raise ValueError(f"Failed to convert obs key '{k}' with type {(v.dtype)} to tensor.") from e

    return ret


def action_torch_to_numpy(action: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Convert model action (torch) -> env action (numpy) with expected dtypes."""
    return {
        "prices": action["prices"].detach().cpu().numpy().astype(np.float64),
        "reposition": action["reposition"].detach().cpu().numpy().astype(np.float32),
        "dispatch": action["dispatch"].detach().cpu().numpy().astype(np.int64),
    }


def _stack_dict(buf: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack list of dict[tensor] into dict[tensor] with leading time/batch dim."""
    keys = buf[0].keys()
    return {k: torch.stack([b[k] for b in buf], dim=0) for k in keys}


def _compute_gae(
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


def train_ppo(
    env: RideShareEnv,
    model: RideShareActorCritic | None = None,
    cfg: PPOTrainConfig | None = None,
) -> tuple[RideShareActorCritic, dict[str, list[float]]]:
    """
    Single-env PPO for your RideShareEnv using RideShareActorCritic.

    Stores rollout of length cfg.rollout_len:
      obs_t, act_t, old_logp_t, value_t, reward_t, done_t
    then performs PPO update with clipped surrogate + value loss + entropy bonus.

    Returns: (trained_model, logs)
    """
    cfg = cfg or PPOTrainConfig()
    model = model or RideShareActorCritic(env)
    model.to(DEVICE)

    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    logs: dict[str, list[float]] = {
        "loss": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clip_frac": [],
        "eval_return": [],
    }

    obs_np, _ = env.reset()
    steps_done = 0
    update_idx = 0

    while steps_done < cfg.total_steps:
        # --- rollout buffers ---
        obs_buf: list[dict[str, torch.Tensor]] = []
        act_buf: list[dict[str, torch.Tensor]] = []
        logp_buf: list[torch.Tensor] = []
        val_buf: list[torch.Tensor] = []
        rew_buf: list[float] = []
        done_buf: list[float] = []

        model.eval()
        for _ in range(cfg.rollout_len):
            obs_t = obs_pd_to_torch(obs_np)

            # sample action + compute logp/value on same obs
            act_t = model.act(obs_t, deterministic=False)
            logp_t, ent_t, v_t = model.log_prob_and_entropy(obs_t, act_t)

            obs_buf.append({k: v.detach() for k, v in obs_t.items()})
            act_buf.append({k: v.detach() for k, v in act_t.items()})
            logp_buf.append(logp_t.detach())
            val_buf.append(v_t.detach())

            obs_np, r, term, trunc, _info = env.step(action_torch_to_numpy(act_t))  # type: ignore[arg-type]
            done = bool(term) or bool(trunc)

            rew_buf.append(float(r))
            done_buf.append(1.0 if done else 0.0)

            steps_done += 1
            if steps_done >= cfg.total_steps:
                break

            if done:
                obs_np, _ = env.reset()

        # bootstrap value from final obs
        with torch.no_grad():
            obs_last_t = obs_pd_to_torch(obs_np)
            v_last = model.forward(obs_last_t)["value"].detach()

        T = len(rew_buf)
        rewards = torch.as_tensor(rew_buf, device=DEVICE, dtype=torch.float32)  # (T,)
        dones = torch.as_tensor(done_buf, device=DEVICE, dtype=torch.float32)  # (T,)
        values = torch.stack(val_buf + [v_last], dim=0).view(T + 1)  # (T+1,)
        old_logp = torch.stack(logp_buf, dim=0).view(T)  # (T,)

        adv, rets = _compute_gae(rewards, values, dones, cfg.gamma, cfg.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_batch = _stack_dict(obs_buf)
        act_batch = _stack_dict(act_buf)

        # --- PPO update ---
        model.train()
        idxs = torch.arange(T, device=DEVICE)

        loss_u = pl_u = vl_u = ent_u = kl_u = clip_u = 0.0
        n_mb = 0

        for _ep in range(cfg.update_epochs):
            perm = idxs[torch.randperm(T, device=DEVICE)]
            for start in range(0, T, cfg.minibatch_size):
                mb = perm[start : start + cfg.minibatch_size]
                if mb.numel() == 0:
                    continue

                mb_obs = {k: v[mb] for k, v in obs_batch.items()}
                mb_act = {k: v[mb] for k, v in act_batch.items()}
                mb_old_logp = old_logp[mb]
                mb_adv = adv[mb]
                mb_rets = rets[mb]

                logp, entropy, value = model.log_prob_and_entropy(mb_obs, mb_act)

                ratio = torch.exp(logp - mb_old_logp)
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv
                policy_loss = -torch.mean(torch.minimum(unclipped, clipped))

                value_loss = 0.5 * torch.mean((mb_rets - value) ** 2)
                entropy_mean = torch.mean(entropy)

                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_mean

                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optim.step()

                with torch.no_grad():
                    approx_kl = torch.mean(mb_old_logp - logp).item()
                    clip_frac = torch.mean((torch.abs(ratio - 1.0) > cfg.clip_eps).to(torch.float32)).item()

                loss_u += float(loss.item())
                pl_u += float(policy_loss.item())
                vl_u += float(value_loss.item())
                ent_u += float(entropy_mean.item())
                kl_u += float(approx_kl)
                clip_u += float(clip_frac)
                n_mb += 1

        if n_mb > 0:
            logs["loss"].append(loss_u / n_mb)
            logs["policy_loss"].append(pl_u / n_mb)
            logs["value_loss"].append(vl_u / n_mb)
            logs["entropy"].append(ent_u / n_mb)
            logs["approx_kl"].append(kl_u / n_mb)
            logs["clip_frac"].append(clip_u / n_mb)

        update_idx += 1
        if update_idx % cfg.log_every_updates == 0:
            ev = evaluate_policy(env, model, episodes=cfg.eval_episodes, deterministic=cfg.deterministic_eval)
            logs["eval_return"].append(ev)

    return model, logs
