import typing as t

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from waymo_agent import RideShareEnv
from waymo_agent.action_heuristic.heuristic_simple import DispatchAgent, PricingAgent, RepositionAgent
from waymo_agent.data_classes.space_dicts import ActionDict, ObservationDict
from waymo_agent.models.torch_np_utils import obs_pd_to_torch
from waymo_agent.data_classes.space_dicts import action_torch_to_numpy


def get_act_dict(agents: tuple[PricingAgent, DispatchAgent, RepositionAgent], obs: ObservationDict) -> ActionDict:
    price_agent, dispatch_agent, reposition_agent = agents
    prices = price_agent.price(obs)
    dispatch_actions = dispatch_agent.dispatch(obs)
    reposition_actions = reposition_agent.reposition(obs)

    action_agent: ActionDict = {
        "prices": prices,
        "dispatch": dispatch_actions,
        "reposition": reposition_actions,
    }
    return action_agent


def run_episode(
    env: RideShareEnv,
    policy_fn: t.Callable[[ObservationDict], ActionDict | dict[str, np.ndarray]],
    seed: int | None = None,
    max_steps: int | None = None,
    show_progress: bool = False,
    desc: str = "episode",
):
    obs, info = env.reset(seed=seed)
    done = False
    ep_return = 0.0
    steps = 0

    iterator = (
        tqdm(
            total=max_steps or env.config.max_episode_steps,
            desc=desc,
            leave=False,
            unit="step",
        )
        if show_progress
        else None
    )

    while not done:
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)  # type: ignore

        ep_return += float(reward)
        done = bool(terminated) or bool(truncated)
        steps += 1

        if iterator is not None:
            iterator.update(1)
            iterator.set_postfix(
                reward=f"{reward:.2f}",
                total=f"{ep_return:.2f}",
            )

        if max_steps is not None and steps >= max_steps:
            break

    if iterator is not None:
        iterator.close()

    return ep_return, info


def ppo_policy(model):
    def _fn(obs_np):
        obs_t = obs_pd_to_torch(obs_np)
        act_t = model.act(obs_t, deterministic=True)
        return action_torch_to_numpy(act_t)

    return _fn


def heuristic_policy(agents):  # or env_heuristic
    def _fn(obs_np):
        # call your heuristic action function here
        return get_act_dict(agents, obs_np)  # must be numpy action dict

    return _fn


def ppo_policy_stochastic(model):
    def _fn(obs_np):
        obs_t = obs_pd_to_torch(obs_np)
        act_t = model.act(obs_t, deterministic=False)  # <- important
        return action_torch_to_numpy(act_t)

    return _fn


def eval_paired_seeds(
    env_model: RideShareEnv,
    env_heuristic: RideShareEnv,
    ppo_policy_fn: t.Callable[[ObservationDict], ActionDict | dict[str, np.ndarray]],
    heur_policy_fn: t.Callable[[ObservationDict], ActionDict | dict[str, np.ndarray]],
    *,
    seeds: list[int],
    show_progress: bool = False,
):
    """
    Runs both policies on the *same* seeds.
    Returns a dict with arrays and summary stats.
    """
    ppo_returns = []
    heur_returns = []

    for s in tqdm(seeds, desc="Evaluating policies", unit="episode"):
        r1, _ = run_episode(env_model, ppo_policy_fn, seed=s, show_progress=show_progress, desc=f"PPO seed={s}")
        r2, _ = run_episode(env_heuristic, heur_policy_fn, seed=s, show_progress=show_progress, desc=f"Heur seed={s}")

        ppo_returns.append(r1)
        heur_returns.append(r2)

    ppo_returns = np.asarray(ppo_returns, dtype=np.float64)
    heur_returns = np.asarray(heur_returns, dtype=np.float64)
    diff = ppo_returns - heur_returns

    out = {
        "seeds": np.asarray(seeds),
        "ppo": ppo_returns,
        "heur": heur_returns,
        "diff": diff,
        "ppo_mean": float(ppo_returns.mean()),
        "ppo_std": float(ppo_returns.std(ddof=1)) if len(seeds) > 1 else float("nan"),
        "heur_mean": float(heur_returns.mean()),
        "heur_std": float(heur_returns.std(ddof=1)) if len(seeds) > 1 else float("nan"),
        "diff_mean": float(diff.mean()),
        "diff_std": float(diff.std(ddof=1)) if len(seeds) > 1 else float("nan"),
        "win_rate": float((diff > 0).mean()),
    }
    return out


# -----------------------------------PLOTTING UTILITIES-----------------------------------#
def plot_min_eval(results: dict):
    ppo = results["ppo"]
    heur = results["heur"]
    diff = results["diff"]

    # Paired scatter
    plt.figure()
    plt.scatter(heur, ppo)
    lo = float(min(ppo.min(), heur.min()))
    hi = float(max(ppo.max(), heur.max()))
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("Heuristic return")
    plt.ylabel("PPO return")
    plt.title(f"Paired seeds: win_rate={results['win_rate']:.2f}, diff_mean={results['diff_mean']:.2f}")
    plt.show()

    # Diff histogram
    plt.figure()
    plt.hist(diff, bins=12)
    plt.xlabel("PPO - Heuristic return")
    plt.ylabel("count")
    plt.title("Return differences across seeds")
    plt.show()


def plot_learning_curve_vs_baseline(logs: dict, heur_mean: float):
    y = np.asarray(logs["eval_return"], dtype=np.float64)
    x = np.arange(len(y))

    plt.figure()
    plt.plot(x, y, label="PPO eval return")
    plt.axhline(heur_mean, linestyle="--", label="Heuristic mean")
    plt.xlabel("Eval checkpoint")
    plt.ylabel("Return")
    plt.title("PPO learning curve vs heuristic")
    plt.legend()
    plt.show()
