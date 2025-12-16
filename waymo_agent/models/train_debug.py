import torch


def assert_finite_dict(d: dict[str, torch.Tensor], prefix: str = "") -> None:
    for k, v in d.items():
        if not torch.isfinite(v).all():
            bad = v[~torch.isfinite(v)]
            raise AssertionError(f"{prefix}{k} has non-finite values, example={bad.flatten()[:5]}")


def assert_action_validity(
    *,
    act: dict[str, torch.Tensor],
    obs: dict[str, torch.Tensor],
    max_pending: int,
    num_veh: int,
    eps_price: float = 1e-6,
) -> None:
    prices = act["prices"]
    dispatch = act["dispatch"]
    reposition = act["reposition"]

    assert tuple(prices.shape) == (max_pending,)
    assert tuple(dispatch.shape) == (max_pending,)
    assert tuple(reposition.shape) == (num_veh, 2)

    # prices >= eps
    if not (prices >= eps_price).all():
        raise AssertionError("prices contains non-positive entries")

    # reposition bounds
    if not ((reposition >= -1.0).all() and (reposition <= 1.0).all()):
        raise AssertionError("reposition out of bounds")

    # dispatch bounds
    if not ((dispatch >= -1).all() and (dispatch < num_veh).all()):
        raise AssertionError("dispatch out of range [-1, num_veh-1]")

    # mask constraint: assigned vehicles must be idle
    dm = obs["dispatch_mask"].detach().cpu().numpy().astype(bool)  # (num_veh,)
    assigned = dispatch.detach().cpu().numpy()
    assigned = assigned[assigned >= 0].astype(int)
    if assigned.size > 0 and not dm[assigned].all():
        raise AssertionError(f"dispatch assigned masked vehicle(s): {assigned[~dm[assigned]]}")
