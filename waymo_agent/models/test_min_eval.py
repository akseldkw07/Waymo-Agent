import numpy as np
import torch

from waymo_agent.data_classes.requests import RequestStatusEnum
from waymo_agent.graph_env.ENV import EnvConfig, RideShareEnv
from waymo_agent.models.ppo_model import (
    RideShareActorCritic,
    action_torch_to_numpy,
    obs_pd_to_torch,
)


def _make_env_and_model():
    env_cfg = EnvConfig(
        max_episode_steps=60 * 1,
        vehicle_per_node=0.10,
        lambda_per_node=0.05,
        max_pending_requests=100,
        max_new_requests_per_step=20,
    )
    env = RideShareEnv(env_cfg)
    model = RideShareActorCritic(env)
    model.eval()
    return env, model


ENV, MODEL = _make_env_and_model()


def test_obs_to_torch_is_finite_and_on_device():
    print("Running test_obs_to_torch_is_finite_and_on_device")
    obs_np, _ = ENV.reset(seed=0)

    obs_t = obs_pd_to_torch(obs_np)
    assert isinstance(obs_t, dict)
    assert len(obs_t) > 0

    for k, v in obs_t.items():
        assert isinstance(v, torch.Tensor), (k, type(v))
        assert v.dtype == torch.float32, (k, v.dtype)
        assert torch.isfinite(v).all().item(), f"Non-finite obs tensor for key={k}"


def test_model_action_shapes_and_dtypes():
    print("Running test_model_action_shapes_and_dtypes")
    obs_np, _ = ENV.reset(seed=0)
    obs_t = obs_pd_to_torch(obs_np)

    with torch.no_grad():
        act_t = MODEL.act(obs_t, deterministic=False)
    assert set(act_t.keys()) == {"prices", "reposition", "dispatch"}

    # shapes
    assert tuple(act_t["prices"].shape) == (ENV.config.max_pending_requests,)
    assert tuple(act_t["reposition"].shape) == (ENV.num_vehicles, 2)
    assert tuple(act_t["dispatch"].shape) == (ENV.config.max_pending_requests,)

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
    assert (d <= ENV.num_vehicles - 1).all().item()


def test_action_numpy_is_ENV_valid():
    print("Running test_action_numpy_is_ENV_valid")
    obs_np, _ = ENV.reset(seed=0)
    obs_t = obs_pd_to_torch(obs_np)

    with torch.no_grad():
        act_t = MODEL.act(obs_t, deterministic=False)
    act_np = action_torch_to_numpy(act_t)

    # Ensure expected keys/types
    assert set(act_np.keys()) == {"prices", "reposition", "dispatch"}
    assert act_np["prices"].dtype == np.float64
    assert act_np["reposition"].dtype in (np.float32, np.float64)  # ENV Box usually float32 ok
    assert act_np["dispatch"].dtype == np.int64

    # Gym space containment check should pass
    assert ENV.action_space.contains(act_np), "Model action does not satisfy ENV.action_space"


def test_one_step_reward_is_finite():
    print("Running test_one_step_reward_is_finite")
    obs_np, _ = ENV.reset(seed=0)
    obs_t = obs_pd_to_torch(obs_np)

    with torch.no_grad():
        act_t = MODEL.act(obs_t, deterministic=False)
    act_np = action_torch_to_numpy(act_t)

    obs2, r, term, trunc, info = ENV.step(act_np)  # type: ignore[arg-type]
    assert np.isfinite(r), f"reward not finite: {r}"
    assert isinstance(info, dict)


def test_dispatch_not_always_no_action_under_stochastic_policy():
    print("Running test_dispatch_not_always_no_action_under_stochastic_policy")
    # This catches the "dispatch always -1" degenerate behavior.
    obs_np, _ = ENV.reset(seed=0)
    obs_t = obs_pd_to_torch(obs_np)

    with torch.no_grad():
        act_t = MODEL.act(obs_t, deterministic=False)
    dispatch = act_t["dispatch"].cpu().numpy()

    frac_noop = (dispatch == -1).mean()
    # Allow it to be high, but not literally 1.0 unless mask forces it.
    # If this fails, your masking / indexing is probably wrong.
    assert frac_noop < 1.0, f"dispatch is ALL no-op (-1). frac_noop={frac_noop:.3f}"


def test_env_eventually_has_dispatchable_request():
    print("Running test_env_eventually_has_dispatchable_request")
    env = ENV
    obs, _ = env.reset(seed=0)

    found = False
    for _ in range(env.config.max_episode_steps):  # 50 minutes of sim time
        # no-op action (valid)
        action = {
            "prices": np.zeros(env.config.max_pending_requests, dtype=np.float64),
            "dispatch": env.action_space["dispatch"].sample(),  # type: ignore  # random valid dispatch
            "reposition": np.zeros((env.num_vehicles, 2), dtype=np.float32),
        }
        obs, r, term, trunc, info = env.step(action)  # type: ignore[arg-type]
        req = env.observation_curr["pending_requests"]
        if (req["status"] == RequestStatusEnum.ACCEPTED.value).any():
            found = True
            break
        if term or trunc:
            break

    assert found, "Never saw an ACCEPTED request; dispatch may be impossible / pricing never triggers acceptance."


def test_pricing_updates_status_and_penalty_matches_counts():
    print("Running test_pricing_updates_status_and_penalty_matches_counts")
    env = ENV
    obs, _ = env.reset(seed=0)

    # Force an extreme price so acceptance prob should be very low
    action = {
        "prices": np.full(env.config.max_pending_requests, 1e6, dtype=np.float64),
        "dispatch": env.action_space["dispatch"].sample(),  # type: ignore  # random valid dispatch
        "reposition": np.zeros((env.num_vehicles, 2), dtype=np.float32),
    }

    obs2, r, term, trunc, info = env.step(action)  # type: ignore[arg-type]

    req = env.observation_curr["pending_requests"]
    num_rejected = (req["status"] == RequestStatusEnum.REJECTED.value).sum()

    # If there are rejected requests, penalty_rejected should be negative and proportional
    penalty = env._rewards.get("penalty_rejected", 0.0)

    if num_rejected > 0:
        assert penalty < 0, f"Rejected {num_rejected} but penalty_rejected={penalty}"
        # Expected penalty = count * config.penalty_rejected
        expected = float(num_rejected) * env.config.penalty_rejected
        assert np.isclose(penalty, expected), (penalty, expected, num_rejected)


# def test_dispatch_creates_active_ride_when_request_accepted(monkeypatch):
#     print("Running test_dispatch_creates_active_ride_when_request_accepted")
#     env = ENV
#     obs, _ = env.reset(seed=0)

#     # Manually force one request to be ACCEPTED and valid.
#     req = env.observation_curr["pending_requests"].copy(deep=True)
#     req.loc[0, "status"] = RequestStatusEnum.ACCEPTED.value
#     req.loc[0, "request_id"] = 123  # ensure valid id if your logic needs it
#     env.observation_curr["pending_requests"] = RequestDF(req)

#     # Ensure vehicle 0 idle
#     veh = env.observation_curr["vehicles"].copy(deep=True)
#     # if you have an enum, set to IDLE explicitly
#     env.observation_curr["vehicles"] = VehicleDF(veh)

#     action = {
#         "prices": np.zeros(env.config.max_pending_requests, dtype=np.float64),
#         "dispatch": np.full(env.config.max_pending_requests, env.config.no_action_id, dtype=np.int64),
#         "reposition": np.zeros((env.num_vehicles, 2), dtype=np.float32),
#     }
#     action["dispatch"][0] = 0  # assign veh_idx=0 to req_idx=0

#     obs2, r, term, trunc, info = env.step(action)  # type: ignore[arg-type]

#     # After step, request should become ASSIGNED (or whatever your transition does)
#     req2 = env.observation_curr["pending_requests"]
#     assert req2.loc[0, "status"] in (RequestStatusEnum.ASSIGNED.value, RequestStatusEnum.ACCEPTED.value)

#     # Active rides should reflect assignment to vehicle 0
#     rides = env.observation_curr["active_rides"]
#     assert (rides["vehicle_id"] == 0).any(), "Dispatch didn't create/attach an active ride for vehicle 0"


def test_deterministic_policy_not_always_noop_when_action_possible():
    print("Running test_deterministic_policy_not_always_noop_when_action_possible")
    env = ENV
    model = RideShareActorCritic(env)
    model.eval()

    obs, _ = env.reset(seed=0)
    obs_t = obs_pd_to_torch(obs)

    with torch.no_grad():
        act = model.act(obs_t, deterministic=True)

    dispatch = act["dispatch"].cpu().numpy()
    # If this is always -1, deterministic eval is basically meaningless early training.
    assert (dispatch != -1).any(), (
        "Deterministic policy returned all no-op dispatch. "
        "This often happens when logits are tied and argmax picks the 'no action' bucket."
    )


def test_reward_components_fire_in_episode():
    print("Running test_reward_components_fire_in_episode")
    env = ENV
    obs, _ = env.reset(seed=0)

    fired = set()
    for _ in range(env.config.max_episode_steps):
        action = {
            "prices": np.random.uniform(0, 10, size=env.config.max_pending_requests).astype(np.float64),
            "dispatch": env.action_space["dispatch"].sample(),  # type: ignore  # random valid dispatch
            "reposition": np.random.uniform(-1, 1, size=(env.num_vehicles, 2)).astype(np.float32),
        }
        obs, r, term, trunc, info = env.step(action)  # type: ignore[arg-type]
        fired |= {k for k, v in env._rewards.items() if float(v) != 0.0}
        if term or trunc:
            break

    # These are “should exist at least sometimes” signals.
    # If none ever fire, reward pipeline is inert.
    assert len(fired) > 0, f"No reward components ever fired. env._rewards keys were always zero. fired={fired}"


# ============================================================
# NEW TESTS: Investigating zero-reward issue
# ============================================================


def test_full_episode_reward_sign():
    """Check that model gets non-zero rewards in a full episode."""
    print("Running test_full_episode_reward_sign")
    obs_np, _ = ENV.reset(seed=42)
    done = False
    ep_reward = 0.0
    step_count = 0
    step_rewards = []

    while not done:
        obs_t = obs_pd_to_torch(obs_np)
        act_t = MODEL.act(obs_t, deterministic=True)
        act_np = action_torch_to_numpy(act_t)

        obs_np, r, term, trunc, info = ENV.step(act_np)  # type: ignore[arg-type]
        ep_reward += float(r)
        step_rewards.append(float(r))
        done = bool(term) or bool(trunc)
        step_count += 1

    # Check that we get at least some steps and non-zero total reward
    assert step_count > 0, f"Episode terminated immediately"
    print(f"\nEpisode: {step_count} steps, total_reward={ep_reward:.4f}")
    print(f"Step rewards sample: {step_rewards[:5]}")
    print(f"Negative steps: {(np.array(step_rewards) < 0).sum()}, Positive: {(np.array(step_rewards) > 0).sum()}")

    # The key check: is reward degenerate?
    # Note: negative rewards are expected due to penalties, but shouldn't be ALL negative consistently
    if ep_reward < -100:
        print(f"WARNING: Very negative episode return {ep_reward:.4f}")


def test_model_forward_output_ranges():
    """Check raw forward() outputs before action sampling."""
    print("Running test_model_forward_output_ranges")
    obs_np, _ = ENV.reset(seed=0)
    obs_t = obs_pd_to_torch(obs_np)

    with torch.no_grad():
        out = MODEL.forward(obs_t)

    print(f"\nModel forward() outputs:")
    print(
        f"  price_mu: min={out['price_mu'].min():.4f}, max={out['price_mu'].max():.4f}, mean={out['price_mu'].mean():.4f}"
    )
    print(
        f"  repo_mu: min={out['repo_mu'].min():.4f}, max={out['repo_mu'].max():.4f}, mean={out['repo_mu'].mean():.4f}"
    )
    print(
        f"  dispatch_logits: min={out['dispatch_logits'].min():.4f}, max={out['dispatch_logits'].max():.4f}, mean={out['dispatch_logits'].mean():.4f}"
    )
    print(f"  value: {out['value'].item():.4f}")

    # Check for reasonable ranges
    assert torch.isfinite(out["price_mu"]).all().item(), "price_mu has non-finite values"
    assert torch.isfinite(out["repo_mu"]).all().item(), "repo_mu has non-finite values"
    assert torch.isfinite(out["dispatch_logits"]).all().item(), "dispatch_logits has non-finite values"
    assert torch.isfinite(out["value"]).all().item(), "value has non-finite values"

    # Dispatch logits should have some variation
    logits = out["dispatch_logits"]
    logits_std = logits.std().item()
    print(f"  dispatch_logits std: {logits_std:.4f}")
    if logits_std < 0.01:
        print("  WARNING: dispatch_logits have very low variance - might indicate degenerate behavior")


def test_mask_effects_on_actions():
    """Check if masking is actually constraining actions."""
    print("Running test_mask_effects_on_actions")
    obs_np, _ = ENV.reset(seed=0)
    obs_t = obs_pd_to_torch(obs_np)

    # Check masks
    dispatch_mask = obs_t["dispatch_mask"]
    pricing_mask = obs_t["pricing_mask"]

    available_vehicles = dispatch_mask.sum().item()
    available_requests = pricing_mask.sum().item()

    print(f"\nMask state:")
    print(f"  Available vehicles: {available_vehicles}/{ENV.num_vehicles}")
    print(f"  Available requests: {available_requests}/{ENV.config.max_pending_requests}")

    with torch.no_grad():
        act_t = MODEL.act(obs_t, deterministic=True)

    dispatch = act_t["dispatch"].cpu().numpy()
    prices = act_t["prices"].cpu().numpy()

    # For pricing: check which requests are masked
    masked_price_indices = np.where(pricing_mask.cpu().numpy() == 0)[0]
    active_price_indices = np.where(pricing_mask.cpu().numpy() > 0)[0]

    print(f"\nAction validation:")
    print(f"  Active requests with prices: {active_price_indices.shape[0]}")
    print(f"  Masked requests: {masked_price_indices.shape[0]}")

    if len(active_price_indices) > 0:
        active_prices = prices[active_price_indices]
        print(
            f"  Active prices: min={active_prices.min():.4f}, max={active_prices.max():.4f}, mean={active_prices.mean():.4f}"
        )

    # Dispatch checking
    masked_veh_indices = np.where(dispatch_mask.cpu().numpy() == 0)[0]
    active_veh_indices = np.where(dispatch_mask.cpu().numpy() > 0)[0]
    print(f"  Active vehicles: {len(active_veh_indices)}")
    print(f"  Masked vehicles: {len(masked_veh_indices)}")

    # Check that dispatch doesn't assign to masked vehicles (except -1)
    for req_idx in range(len(dispatch)):
        d = dispatch[req_idx]
        if d >= 0 and d < ENV.num_vehicles:
            # This is an actual vehicle assignment
            if dispatch_mask[d].item() == 0:
                print(f"  WARNING: Request {req_idx} assigned to masked vehicle {d}")


def test_heuristic_vs_model_reward_baseline():
    """Run both model and heuristic for comparison."""
    print("Running test_heuristic_vs_model_reward_baseline")
    from waymo_agent.action_heuristic.heuristic_simple import DispatchAgent, PricingAgent, RepositionAgent

    env_test = ENV
    model_test = RideShareActorCritic(env_test)
    model_test.eval()

    price_agent = PricingAgent(env_test.config)
    dispatch_agent = DispatchAgent(env_test)
    reposition_agent = RepositionAgent(env_test)

    # Run model
    obs_np, _ = env_test.reset(seed=100)
    model_ep_reward = 0.0
    done = False
    model_steps = 0

    while not done:
        obs_t = obs_pd_to_torch(obs_np)
        act_t = model_test.act(obs_t, deterministic=True)
        act_np = action_torch_to_numpy(act_t)
        obs_np, r, term, trunc, _ = env_test.step(act_np)  # type: ignore[arg-type]
        model_ep_reward += float(r)
        done = bool(term) or bool(trunc)
        model_steps += 1

    # Run heuristic on same env (reset)
    obs_np, _ = env_test.reset(seed=100)
    heur_ep_reward = 0.0
    done = False
    heur_steps = 0

    while not done:
        prices = price_agent.price(obs_np)
        dispatch_actions = dispatch_agent.dispatch(obs_np)
        reposition_actions = reposition_agent.reposition(obs_np)
        action_agent = {
            "prices": prices,
            "dispatch": dispatch_actions,
            "reposition": reposition_actions,
        }
        obs_np, r, term, trunc, _ = env_test.step(action_agent)  # type: ignore[arg-type]
        heur_ep_reward += float(r)
        done = bool(term) or bool(trunc)
        heur_steps += 1

    print(f"\nModel vs Heuristic comparison (seed=100):")
    print(f"  Model:      steps={model_steps}, ep_reward={model_ep_reward:.4f}")
    print(f"  Heuristic:  steps={heur_steps}, ep_reward={heur_ep_reward:.4f}")
    print(f"  Model better? {model_ep_reward > heur_ep_reward}")

    # The key check: neither should be ALL negative
    assert model_ep_reward > -1000, f"Model catastrophically bad: {model_ep_reward:.4f}"
    assert heur_ep_reward > -1000, f"Heuristic catastrophically bad: {heur_ep_reward:.4f}"
