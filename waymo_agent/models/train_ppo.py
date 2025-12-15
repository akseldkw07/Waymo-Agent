from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from waymo_agent.constants import DEVICE_TORCH_STR
from waymo_agent.data_classes.space_dicts import action_torch_to_numpy
from waymo_agent.graph_env.ENV import RideShareEnv
from waymo_agent.models.torch_np_utils import stack_dict

DEVICE = torch.device(DEVICE_TORCH_STR)

from .ppo_model import RideShareActorCritic
from .train_utils import *
from waymo_agent.models.train_debug import assert_finite_dict, assert_action_validity


def train_ppo(
    env: RideShareEnv,
    model: RideShareActorCritic | None = None,
    train_cfg: PPOTrainConfig | None = None,
    save_path: str | Path | None = None,
) -> tuple[RideShareActorCritic, dict[str, list[float]]]:
    """
    Single-env PPO for your RideShareEnv using RideShareActorCritic.

    Stores rollout of length cfg.rollout_len:
      obs_t, act_t, old_logp_t, value_t, reward_t, done_t
    then performs PPO update with clipped surrogate + value loss + entropy bonus.

    Returns: (trained_model, logs)
    """
    train_cfg = train_cfg or PPOTrainConfig()
    model = model or RideShareActorCritic(env)
    model.to(DEVICE)

    optim = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)

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

    best_eval_return = float("-inf")
    no_improve_checks = 0

    pbar = tqdm(total=train_cfg.total_steps, desc="PPO steps", unit="step", position=0)

    while steps_done < train_cfg.total_steps:
        # --- rollout buffers ---
        obs_buf: list[dict[str, torch.Tensor]] = []
        act_buf: list[dict[str, torch.Tensor]] = []
        logp_buf: list[torch.Tensor] = []
        val_buf: list[torch.Tensor] = []
        rew_buf: list[float] = []
        done_buf: list[float] = []

        model.eval()
        for _ in range(env.config.max_episode_steps):
            obs_t = obs_pd_to_torch(obs_np)

            # sample action + compute logp/value on same obs
            act_t = model.act(obs_t, deterministic=False)
            if train_cfg.debug:
                assert_finite_dict(obs_t, prefix="obs/")
                assert_finite_dict(act_t, prefix="act/")
                assert_action_validity(
                    act=act_t,
                    obs=obs_t,
                    max_pending=env.config.max_pending_requests,
                    num_veh=env.num_vehicles,
                    eps_price=1e-6,
                )
            logp_t, ent_t, v_t = model.log_prob_and_entropy(obs_t, act_t)
            assert torch.isfinite(logp_t).all(), f"logp_t bad: {logp_t}"
            assert torch.isfinite(v_t).all(), f"value bad: {v_t}"
            for k, v in act_t.items():
                assert torch.isfinite(v).all(), f"act[{k}] has non-finite"

            obs_buf.append({k: v.detach() for k, v in obs_t.items()})
            act_buf.append({k: v.detach() for k, v in act_t.items()})
            logp_buf.append(logp_t.detach())
            val_buf.append(v_t.detach())

            obs_np, r, term, trunc, _info = env.step(action_torch_to_numpy(act_t))  # type: ignore[arg-type]
            assert np.isfinite(r), f"reward non-finite: {r}"
            done = bool(term) or bool(trunc)

            rew_buf.append(float(r))
            done_buf.append(1.0 if done else 0.0)

            steps_done += 1
            pbar.update(1)
            if steps_done >= train_cfg.total_steps:
                break

            if done:
                obs_np, _ = env.reset()

        # bootstrap value from final obs
        with torch.no_grad():
            obs_last_t = obs_pd_to_torch(obs_np)
            h_last, v_last = model.forward(obs_last_t)

        T = len(rew_buf)
        rewards = torch.as_tensor(rew_buf, device=DEVICE, dtype=torch.float32)  # (T,)
        dones = torch.as_tensor(done_buf, device=DEVICE, dtype=torch.float32)  # (T,)
        values = torch.stack(val_buf + [v_last.detach()], dim=0).view(T + 1)  # (T+1,)
        old_logp = torch.stack(logp_buf, dim=0).view(T)  # (T,)

        adv, rets = compute_gae(rewards, values, dones, train_cfg.gamma, train_cfg.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_batch = stack_dict(obs_buf)
        act_batch = stack_dict(act_buf)

        # --- PPO update ---
        model.train()
        idxs = torch.arange(T, device=DEVICE)

        loss_u = pl_u = vl_u = ent_u = kl_u = clip_u = 0.0
        n_mb = 0

        eval_bar = tqdm(range(train_cfg.update_epochs), desc="PPO update epochs", leave=True, position=1)
        for _ep in eval_bar:
            perm = idxs[torch.randperm(T, device=DEVICE)]
            for start in range(0, T, train_cfg.minibatch_size):
                mb = perm[start : start + train_cfg.minibatch_size]
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
                clipped = torch.clamp(ratio, 1.0 - train_cfg.clip_eps, 1.0 + train_cfg.clip_eps) * mb_adv
                policy_loss = -torch.mean(torch.minimum(unclipped, clipped))

                value_loss = 0.5 * torch.mean((mb_rets - value) ** 2)
                entropy_mean = torch.mean(entropy)

                loss = policy_loss + train_cfg.vf_coef * value_loss - train_cfg.ent_coef * entropy_mean

                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
                optim.step()

                with torch.no_grad():
                    approx_kl = torch.mean(mb_old_logp - logp).item()
                    clip_frac = torch.mean((torch.abs(ratio - 1.0) > train_cfg.clip_eps).to(torch.float32)).item()

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

        if update_idx % train_cfg.log_every_updates == 0:
            eval_ret = evaluate_policy(
                env, model, episodes=train_cfg.eval_episodes, deterministic=train_cfg.deterministic_eval
            )
            logs["eval_return"].append(eval_ret)
            eval_bar.set_postfix({"update": update_idx, "eval_return": f"{eval_ret:.3f}"})
            eval_bar.refresh()
            pbar.set_postfix({"eval_return": f"{eval_ret:.3f}"})

            if eval_ret > best_eval_return + train_cfg.min_delta:
                best_eval_return = float(eval_ret)
                no_improve_checks = 0

                if train_cfg.save_best and save_path is not None:
                    sp = Path(save_path)
                    # If save_path is a directory, write best.pt inside it
                    if sp.suffix == "":
                        sp.mkdir(parents=True, exist_ok=True)
                        best_path = sp / "best.pt"
                    else:
                        sp.parent.mkdir(parents=True, exist_ok=True)
                        best_path = sp
                    torch.save(model.state_dict(), best_path)
            else:
                no_improve_checks += 1
                if train_cfg.patience > 0 and no_improve_checks >= train_cfg.patience:
                    pbar.set_postfix({"stopped": "early", "best_eval": f"{best_eval_return:.3f}"})
                    break
        update_idx += 1

    return model, logs
