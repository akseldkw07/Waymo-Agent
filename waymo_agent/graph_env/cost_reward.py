from __future__ import annotations

import numpy as np

from waymo_agent.data_classes import EnvConfig
from waymo_agent.data_classes.active_rides import ActiveRideDF


def compute_operating_cost(distance_meters: np.ndarray, config: EnvConfig) -> np.ndarray:
    """
    Compute the operating cost for a vehicle given the distance traveled.
    """
    return distance_meters * config.operating_cost_per_meter


def compute_amortized_reward(
    curr: ActiveRideDF, prev: ActiveRideDF, config: EnvConfig, mask: np.ndarray | None = None
) -> np.ndarray:
    """
    Compute the amortized reward for rides based on distance traveled and completion.
    """
    if mask is None:
        mask = np.ones(len(curr), dtype=bool)
    total_dist = np.maximum(curr.total_trip_distance_meters, 1e-6)
    prev_rem = prev.trip_distance_remaining_meters
    curr_rem = curr.trip_distance_remaining_meters

    delta_rem = np.maximum(prev_rem - curr_rem, 0.0)
    progress_frac_step = delta_rem / total_dist

    amortized_reward = 0.85 * curr.price * progress_frac_step
    is_completed = (curr.complete) & (~prev.complete)
    completion_bonus = 0.15 * curr.price * is_completed.astype(float)

    step_reward = amortized_reward + completion_bonus
    step_cost = delta_rem * config.operating_cost_per_meter

    return (step_reward - step_cost) * mask
