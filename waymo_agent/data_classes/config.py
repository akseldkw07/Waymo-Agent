from __future__ import annotations

import datetime as dt
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import MANHATTAN_ENRICHED_GRAPH, MAP_DIR

if t.TYPE_CHECKING:
    from ..osmnx.osmnx_constants import Plot_graph_TypedDict, Plot_route_TypedDict


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

    def pen_df_empty(self, len: int = 1) -> pd.DataFrame:
        df_cols = self.__dataclass_fields__.keys()
        ret = pd.DataFrame({col: [0.0] * len for col in df_cols})
        return ret


@dataclass
class EnvConfig:
    # Map and environment parameters
    map_name: str = MANHATTAN_ENRICHED_GRAPH.format(782)  # "manhattan-sparse-782-nodes-enriched.graphml"
    map_dir: Path = MAP_DIR
    vehicle_per_node: float = 0.1  # must be greater than 0.0
    max_vehicles: int = 25
    time_step_delta: dt.timedelta = dt.timedelta(minutes=1)
    max_episode_steps: int = 60 * 24  # 1 day
    day_per_week: int = 7
    hours_per_day: int = 24

    # Demand parameters
    lambda_per_node: float = 1 / 200  # average number of requests per node per minute
    lambda_variation_coef: float = 2.0

    # Request parameters
    max_pending_requests: int = 25
    max_wait_time: float = 0.25  # in hours

    # Pricing
    distance_fare: float = 1.8  # per meter
    base_fare: float = 2.5
    time_fare_per_hour: float = 15.0
    max_price: float = np.inf  # maximum allowable price for a ride

    # Dispatching
    no_action_id = -1
    invalid_id: int = -1

    # Customer acceptance model parameters
    acceptance_margin_weight: float = -0.20
    acceptance_supply_demand_weight: float = 0.6

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

    # Graphing parameters
    ox_plot_requests: Plot_route_TypedDict = field(
        default_factory=lambda: {"route_color": "cyan", "route_linewidth": 2.0}
    )
    ox_plot_active_rides: Plot_route_TypedDict = field(
        default_factory=lambda: {"route_color": "limegreen", "route_linewidth": 3.0}
    )

    ox_plot_default: Plot_graph_TypedDict = field(
        default_factory=lambda: {
            "node_size": 5,
            "node_color": "white",
            "node_alpha": 0.85,
            "figsize": (10, 10),
            "bgcolor": "black",
            "show": False,
            "edge_linewidth": 1.0,
        }
    )

    # Verbose Rewards
    verbose_rewards: bool = True
    shaped_reward_config: RewardShapingConfig = field(default_factory=RewardShapingConfig)
