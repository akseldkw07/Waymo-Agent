import typing as t

import numpy as np

from waymo_agent.data_classes import ActiveRideDF, RequestDF, VehicleDF


class ObservationDict(t.TypedDict):
    """
    TypedDict for the observation space of the RideShare environment.
    """

    globals: np.ndarray
    supply_demand_ratio: np.ndarray
    vehicles: VehicleDF
    pending_requests: RequestDF
    active_rides: ActiveRideDF
    dispatch_mask: np.ndarray
    pricing_mask: np.ndarray


def prune_obs_dict_gymnasium(obs: ObservationDict) -> ObservationDict:
    """
    Prune the observation dictionary to be compatible with Gymnasium.

    Args:
        obs (ObservationDict): The original observation dictionary.

    Returns:
        ObservationDict: The pruned observation dictionary.
    """
    obs_gymnasium: ObservationDict = {
        "globals": obs["globals"],
        "supply_demand_ratio": obs["supply_demand_ratio"],
        "vehicles": obs["vehicles"].to_training_df(),
        "pending_requests": obs["pending_requests"].to_training_df(),
        "active_rides": obs["active_rides"].to_training_df(),
        "dispatch_mask": obs["dispatch_mask"],
        "pricing_mask": obs["pricing_mask"],
    }
    return obs_gymnasium


class ActionDict(t.TypedDict):
    """
    TypedDict for the action space of the RideShare environment.
    """

    prices: np.ndarray
    reposition: np.ndarray
    dispatch: np.ndarray


T = t.TypeVar("T", bound=t.Any | dict)


def validate_keys(dict_type: t.Type[T], dict: t.Mapping) -> None:
    """
    Assert that the keys of a TypedDict match the expected keys.

    Args:
        dict_type (t.Type[t.TypedDict]): The TypedDict type to check.
        dict (t.Mapping): The dictionary to validate.
    """
    expected_keys = set(dict_type.__annotations__.keys())
    keys = set(dict.keys())
    assert expected_keys == keys, f"Expected keys {keys}, but got {expected_keys}"
