from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path


class VehicleStatus(enum.IntEnum):
    IDLE = 0
    TO_PICKUP = 1
    WITH_PASSENGER = 2
    REPOSITIONING = 3
    CHARGING = 4


class RequestStatus(enum.IntEnum):
    AWAITING_PRICE = 0
    ACCEPTED = 1
    ASSIGNED = 2
    CANCELLED = 3
    COMPLETED = 4


@dataclass
class EnvConfig:
    map_name: str | None = "manhattan"
    map_dir: str | Path = "maps"
    num_nodes: int = 8
    num_vehicles: int = 4
    max_pending_requests: int = 12
    demand_rate: float = 2.5
    minutes_per_step: int = 1
    max_episode_steps: int = 12 * 60
    base_fare: float = 4.0
    distance_fare: float = 1.8
    price_action_scale: float = 0.25
    min_price_multiplier: float = 0.6
    max_price_multiplier: float = 2.0
    acceptance_base: float = 0.2
    acceptance_price_weight: float = -2.0
    acceptance_distance_weight: float = -0.05
    acceptance_wait_weight: float = -0.1
    acceptance_supply_demand_weight: float = 0.5
    customer_bias_std: float = 0.5
    operating_cost_per_km: float = 0.4
    battery_consumption_per_km: float = 0.01
    charge_rate_per_minute: float = 0.12
    min_battery_for_assignment: float = 0.2
    speed_km_per_min: float = 1.0
    penalty_rejected: float = -1.0
    penalty_cancelled: float = -2.0
    penalty_overflow: float = -1.5
    wait_time_normalizer: float = 10.0
    supply_demand_normalizer: float = 2.5
    surge_reference: float = 1.0


@dataclass
class VehicleState:
    node: int
    battery: float
    status: VehicleStatus = VehicleStatus.IDLE
    target_node: int | None = None
    remaining_distance: float = 0.0
    ride_id: int | None = None
    origin_node: int | None = None
    travel_distance_total: float = 0.0


@dataclass
class RequestState:
    request_id: int
    pickup_node: int
    dropoff_node: int
    distance: float
    base_price: float
    price: float
    customer_bias: float
    wait_time: float = 0.0
    status: RequestStatus = RequestStatus.AWAITING_PRICE


@dataclass
class ActiveRide:
    ride_id: int
    vehicle_id: int
    pickup_node: int
    dropoff_node: int
    price: float
    pickup_distance_remaining: float
    trip_distance_remaining: float
    total_distance: float
