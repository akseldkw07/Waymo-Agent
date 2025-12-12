from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import MANHATTAN_ENRICHED_GRAPH, MAP_DIR


@dataclass
class RewardShapingConfig:
    """
    Reward shaping parameters

    The idea here is to provide dense rewards to the agent to help learning
    instead of sparse rewards only at ride completion. In theory, these
    should be "learnable" and removable, but are helpful for initial training.
    """

    penalty_multiple_dispatch_assignment: float = -0.1  # penalty for assigning multiple vehicles to the same request
    penalty_assign_to_unavailable_vehicle: float = -0.2  # penalty for assigning a vehicle that is not available
    distance_penalty_xy_normed: float = -0.05  # penalty per unit normalized distance traveled without a passenger

    def reward_df_empty(self, len: int = 1) -> pd.DataFrame:
        df_cols = self.__dataclass_fields__.keys()
        ret = pd.DataFrame({col: [0.0] * len for col in df_cols})
        return ret


@dataclass
class EnvConfig:
    # Map and environment parameters
    map_name: str = MANHATTAN_ENRICHED_GRAPH.format(782)  # "manhattan-sparse-782-nodes-enriched.graphml"
    map_dir: Path = MAP_DIR
    vehicle_per_node: float = 0.03  # must be greater than 0.0
    max_vehicles: int = 25
    time_step_delta: dt.timedelta = dt.timedelta(minutes=1)
    max_episode_steps: int = 60 * 24  # 1 day
    day_per_week: int = 7
    hours_per_day: int = 24

    # Demand parameters
    lambda_per_node: float = vehicle_per_node / 10
    # lambda_per_node: float = vehicle_per_node
    lambda_variation_coef: float = 2.0
    max_new_requests_per_step: int | None = None  # if None, no limit

    # Request parameters
    max_pending_requests: int = 50
    max_wait_time_minutes: int = 15  # in minutes

    # Pricing
    max_price: float = np.inf  # maximum allowable price for a ride

    # Dispatching
    no_action_id = -1
    invalid_id: int = -1

    # Customer acceptance model parameters
    acceptance_margin_weight: float = -2.0
    acceptance_supply_demand_weight: float = 3.0

    # Battery and charging parameters
    battery_consumption_per_km: float = 0.01
    charge_rate_per_minute: float = 0.12
    min_battery_for_assignment: float = 0.2
    chargable_node_ratio: float = 0.01
    max_chargers_per_node: int = 1

    # Rewards & Penalties
    penalty_rejected: float = -1.0
    penalty_expire: float = -2.0
    penalty_overflow: float = -1.5
    reward_frac_ride_completion: float = 15.0  # fraction of ride price rewarded upon ride completion
    operating_cost_per_km: float = 1.0

    @property
    def operating_cost_per_meter(self) -> float:
        return self.operating_cost_per_km / 1000.0

    # Verbose Rewards
    verbose_rewards: bool = True
    shaped_reward_config: RewardShapingConfig = field(default_factory=RewardShapingConfig)
