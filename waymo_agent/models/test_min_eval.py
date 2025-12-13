import numpy as np
import torch

from waymo_agent.graph_env.ENV import RideShareEnv
from waymo_agent.models.ppo_model import (
    RideShareActorCritic,
    obs_pd_to_torch,
    action_torch_to_numpy,
)


def _make_env_and_model():
    env = RideShareEnv()
    model = RideShareActorCritic(env)
    model.eval()
    return env, model


def test_obs_to_torch_is_finite_and_on_device():
    env, _ = _make_env_and_model()
    obs_np, _ = env.reset(seed=0)

    obs_t = obs_pd_to_torch(obs_np)
    assert isinstance(obs_t, dict)
    assert len(obs_t) > 0

    for k, v in obs_t.items():
        assert isinstance(v, torch.Tensor), (k, type(v))
        assert v.dtype == torch.float32, (k, v.dtype)
        assert torch.isfinite(v).all().item(), f"Non-finite obs tensor for key={k}"


def test_model_action_shapes_and_dtypes():
    env, model = _make_env_and_model()
    obs_np, _ = env.reset(seed=0)
    obs_t = obs_pd_to_torch(obs_np)

    with torch.no_grad():
        act_t = model.act(obs_t, deterministic=False)

    assert set(act_t.keys()) == {"prices", "reposition", "dispatch"}

    # shapes
    assert tuple(act_t["prices"].shape) == (env.config.max_pending_requests,)
    assert tuple(act_t["reposition"].shape) == (env.num_vehicles, 2)
    assert tuple(act_t["dispatch"].shape) == (env.config.max_pending_requests,)

    # dtypes
    assert act_t["dispatch"].dtype in (torch.int64, torch.int32)
    assert act_t["prices"].dtype == torch.float32
    assert act_t["reposition"].dtype == torch.float32

    # ranges
    assert torch.isfinite(act_t["prices"]).all().item()
    assert (act_t["prices"] > 0).all().item(), "prices must be strictly positive"
    assert torch.isfinite(act_t["reposition"]).all().item()
    assert (act_t["reposition"] >= -1.0).all().item()
    assert (act_t["reposition"] <= 1.0).all().item()

    # dispatch should be in [-1..num_vehicles-1]
    d = act_t["dispatch"]
    assert (d >= -1).all().item()
    assert (d <= env.num_vehicles - 1).all().item()


def test_action_numpy_is_env_valid():
    env, model = _make_env_and_model()
    obs_np, _ = env.reset(seed=0)
    obs_t = obs_pd_to_torch(obs_np)

    with torch.no_grad():
        act_t = model.act(obs_t, deterministic=False)
    act_np = action_torch_to_numpy(act_t)

    # Ensure expected keys/types
    assert set(act_np.keys()) == {"prices", "reposition", "dispatch"}
    assert act_np["prices"].dtype == np.float64
    assert act_np["reposition"].dtype in (np.float32, np.float64)  # env Box usually float32 ok
    assert act_np["dispatch"].dtype == np.int64

    # Gym space containment check should pass
    assert env.action_space.contains(act_np), "Model action does not satisfy env.action_space"


def test_one_step_reward_is_finite():
    env, model = _make_env_and_model()
    obs_np, _ = env.reset(seed=0)
    obs_t = obs_pd_to_torch(obs_np)

    with torch.no_grad():
        act_t = model.act(obs_t, deterministic=False)
    act_np = action_torch_to_numpy(act_t)

    obs2, r, term, trunc, info = env.step(act_np)  # type: ignore[arg-type]
    assert np.isfinite(r), f"reward not finite: {r}"
    assert isinstance(info, dict)


def test_dispatch_not_always_no_action_under_stochastic_policy():
    # This catches the “dispatch always -1” degenerate behavior.
    env, model = _make_env_and_model()
    obs_np, _ = env.reset(seed=0)
    obs_t = obs_pd_to_torch(obs_np)

    with torch.no_grad():
        act_t = model.act(obs_t, deterministic=False)
    dispatch = act_t["dispatch"].cpu().numpy()

    frac_noop = (dispatch == -1).mean()
    # Allow it to be high, but not literally 1.0 unless mask forces it.
    # If this fails, your masking / indexing is probably wrong.
    assert frac_noop < 1.0, f"dispatch is ALL no-op (-1). frac_noop={frac_noop:.3f}"
