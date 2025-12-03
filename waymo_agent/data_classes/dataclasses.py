from __future__ import annotations

import enum
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..constants import MANHATTAN_ENRICHED_GRAPH, MAP_DIR

if t.TYPE_CHECKING:
    from ..osmnx.osmnx_constants import Plot_graph_TypedDict, Plot_route_TypedDict

# import waymo_agent.osmnx.osmnx_constants as osmnx


class WeatherEnum(enum.IntEnum):
    CLEAR = 0
    RAINY = 1
    SNOWY = 2


class VehicleStatusEnum(enum.IntEnum):
    IDLE = 0  # idle means vehicle is not assigned to any request, can be repositioned
    TO_PICKUP = 1
    WITH_PASSENGER = 2
    CHARGING = 3  # vehicle is at a charging station, recharging. NOTE not currently applicable


class RequestStatusEnum(enum.IntEnum):
    AWAITING_PRICE = 0
    ACCEPTED = 1
    ASSIGNED = 2
    CANCELLED = 3
    COMPLETED = 4


@dataclass
class EnvConfig:
    # Map and environment parameters
    map_name: str = MANHATTAN_ENRICHED_GRAPH.format(774)  # "manhattan-sparse-774-nodes-enriched.graphml"
    map_dir: Path = MAP_DIR
    vehicle_per_node: float = 0.1
    minutes_per_step: int = 1
    max_episode_steps: int = 60 * 12  # 12 hours
    day_per_week: int = 7
    hours_per_day: int = 24

    # Demand parameters
    lambda_per_node: float = 0.01
    lambda_variation_coef: float = 2.0

    # Request parameters
    max_pending_requests: int = 25
    max_wait_time: float = 0.25  # in hours

    # Pricing
    distance_fare: float = 1.8  # per meter
    base_fare: float = 2.5
    time_fare_per_hour: float = 15.0
    max_price: float = 1_000.0  # maximum allowable price for a ride

    # Customer acceptance model parameters
    acceptance_base: float = 0.2
    acceptance_price_weight: float = -2.0
    acceptance_distance_weight: float = -0.05
    acceptance_wait_weight: float = -0.1
    acceptance_supply_demand_weight: float = 0.5
    customer_bias_std: float = 0.5
    operating_cost_per_km: float = 0.4

    # Battery and charging parameters
    battery_consumption_per_km: float = 0.01
    charge_rate_per_minute: float = 0.12
    min_battery_for_assignment: float = 0.2
    chargable_node_ratio: float = 0.01
    max_chargers_per_node: int = 1

    """
    # Penalties
    penalty_rejected: float = -1.0
    penalty_cancelled: float = -2.0
    penalty_overflow: float = -1.5
    """

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
            "node_alpha": 0.5,
            "figsize": (10, 10),
            "bgcolor": "black",
            "show": True,
            "edge_linewidth": 1.0,
        }
    )


@dataclass
class VehicleState:
    vehicle_id: int
    loc_norm: tuple[float, float]  # normalized (x, y) location in [0, 1]x[0, 1]
    battery: float  # 0.0 to 1.0 NOTE not used in current version
    status: VehicleStatusEnum = VehicleStatusEnum.IDLE
    ride_id: int | None = None

    def to_numpy(self) -> np.ndarray:

        status_one_hot = np.zeros((4,), dtype=np.float32)
        status_one_hot[int(self.status)] = 1.0
        return np.array([*self.loc_norm, self.battery, *status_one_hot], dtype=np.float32)

    @staticmethod
    def space_config(config: EnvConfig):
        shape = (7,)
        low = np.zeros(shape, dtype=np.float32)
        high = np.ones(shape, dtype=np.float32)
        return {"shape": shape, "low": low, "high": high}


@dataclass
class PathPosition:
    # TODO is this necessary?
    route: list[int]  # node ids along path
    edge_idx: int  # which edge on the route we're on (between route[edge_idx] -> route[edge_idx+1])
    dist_on_edge: float  # meters already traveled along this current edge


@dataclass
class RequestState:
    request_id: int
    rider_id: int
    request_time: tuple[int, float]  # timestep when request was created. (day_of_week, time_of_day in hours)

    pickup_node_id: int
    pickup_loc_norm: tuple[float, float]
    dropoff_node_id: int
    dropoff_loc_norm: tuple[float, float]
    route: PathPosition

    distance_raw: float  # in meters
    distance_norm: float  # normalized distance for easier NN training, [0,1]
    base_price: float  # this is price of distance * distance_fare + base fee + time charge

    wait_time: float = 0.0
    status: RequestStatusEnum = RequestStatusEnum.AWAITING_PRICE

    def to_numpy(self) -> np.ndarray:
        return np.array(
            [
                *self.pickup_loc_norm,
                *self.dropoff_loc_norm,
                self.distance_norm,
                self.base_price,
                self.wait_time,
                1.0 if self.status == RequestStatusEnum.ACCEPTED else 0.0,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def space_config(config: EnvConfig):
        shape = (8,)
        low = np.zeros(shape, dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, config.max_price, config.max_wait_time, 1.0], dtype=np.float32)
        return {"shape": shape, "low": low, "high": high}


@dataclass
class ActiveRide:
    ride_id: int
    vehicle_id: int

    pickup_node: int
    pickup_loc_norm: tuple[float, float]
    dropoff_node: int
    dropoff_loc_norm: tuple[float, float]

    price: float
    total_trip_distance_raw: float  # in meters
    total_trip_distance_norm: float  # normalized distance for easier NN training, [0,1]
    trip_distance_remaining_norm: float
    pickup_distance_remaining_norm: float

    route: PathPosition

    def to_numpy(self) -> np.ndarray:
        ret = np.array(
            [
                *self.pickup_loc_norm,
                *self.dropoff_loc_norm,
                self.price,
                self.total_trip_distance_norm,
                self.trip_distance_remaining_norm,
                self.pickup_distance_remaining_norm,
            ],
            dtype=np.float32,
        )
        return ret

    @staticmethod
    def space_config(config: EnvConfig):
        shape = (8,)
        low = np.zeros(shape, dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0, config.max_price, 1.0, 1.0, 1.0], dtype=np.float32)

        return {"shape": shape, "low": low, "high": high}
