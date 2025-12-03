import typing as t

import numpy as np


class ObservationDict(t.TypedDict):
    """
    TypedDict for the observation space of the RideShare environment.
    """

    globals: np.ndarray
    supply_demand_ratio: np.ndarray
    vehicles: np.ndarray
    pending_requests: np.ndarray
    active_rides: np.ndarray
    dispatch_mask: np.ndarray
    pricing_mask: np.ndarray


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
