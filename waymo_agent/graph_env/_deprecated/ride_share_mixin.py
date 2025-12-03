from __future__ import annotations

import numpy as np
from gymnasium import spaces

from ...data_classes.dataclasses import RequestStatusEnum, VehicleState, VehicleStatusEnum
from ..mixin.interface import GraphMixinInterface


class RideShareObservationMixin(GraphMixinInterface):
    """
    Mixin responsible for constructing the observation space and generating observations.

    This mixin handles:
    - Initializing vehicle states.
    - Defining the observation and action spaces.
    - Constructing the observation dictionary from the current environment state.
    - Calculating metrics like supply-demand ratio.
    """

    # ------------------------------------------------------------------ #
    # Initialization helpers
    # ------------------------------------------------------------------ #
    def _initialize_vehicles(self):
        """
        Initialize vehicles based on configuration.
        """
        # self.num_vehicles is already set in __init__

        self.vehicles = []
        for i in range(self.num_vehicles):
            # Randomly assign a starting node
            start_node_idx = self.np_random.integers(0, self.num_nodes)

            vehicle = VehicleState(
                node_id=start_node_idx, battery=1.0, status=VehicleStatusEnum.IDLE  # Start with full battery
            )
            self.vehicles.append(vehicle)

    def _validate_action(self, action: dict):
        if not self.action_space.contains(action):
            raise ValueError("Action is outside the defined action space.")

    def _build_observation_space(self):
        num_r = self.config.max_pending_requests
        num_v = self.num_vehicles
        self.observation_space = spaces.Dict(
            {
                "global": spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
                "supply_demand_ratio": spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
                "vehicles": spaces.Box(low=0.0, high=1.0, shape=(num_v, 6), dtype=np.float32),
                "pending_requests": spaces.Box(low=0.0, high=1.0, shape=(num_r, 6), dtype=np.float32),
                "dispatch_mask": spaces.Box(low=0.0, high=1.0, shape=(num_r, num_v + 1), dtype=np.float32),
                "pricing_mask": spaces.Box(low=0.0, high=1.0, shape=(num_r,), dtype=np.float32),
            }
        )

    def _build_action_space(self):
        """
        TODO rethink the action space
        What kinds of decisions can we replace with a very good heuristic? such as how to reposition a car
        that's already in use
        """
        num_v = self.num_vehicles
        num_r = self.config.max_pending_requests
        self.action_space = spaces.Dict(
            {
                "price_adjustments": spaces.Box(low=-1.0, high=1.0, shape=(num_r,), dtype=np.float32),
                "dispatch": spaces.MultiDiscrete(np.full(num_r, num_v + 1, dtype=np.int32)),
                "reposition": spaces.MultiDiscrete(np.full(num_v, self.num_nodes, dtype=np.int32)),
                "toggle_charging": spaces.MultiBinary(num_v),
            }
        )

    def _get_observation(self):
        global_features = np.array(
            [
                self.time_of_day / 24.0,
                self.day_of_week / 6.0,
                self.weather_idx / max(1, len(self.WEATHER_STATES) - 1),
            ],
            dtype=np.float32,
        )
        supply_demand = np.array([self._supply_demand_ratio()], dtype=np.float32)
        vehicle_obs = np.zeros((self.num_vehicles, 6), dtype=np.float32)
        for idx, vehicle in enumerate(self.vehicles):
            # TODO: Implement _normalize_node properly or ensure it exists in interface
            # For now assuming simple normalization by num_nodes
            vehicle_obs[idx, 0] = float(vehicle.node_id) / max(1, self.num_nodes)
            vehicle_obs[idx, 1] = np.float32(np.clip(vehicle.battery, 0.0, 1.0))
            vehicle_obs[idx, 2] = 1.0 if vehicle.status == VehicleStatusEnum.IDLE else 0.0
            vehicle_obs[idx, 3] = 1.0 if vehicle.status == VehicleStatusEnum.TO_PICKUP else 0.0
            vehicle_obs[idx, 4] = 1.0 if vehicle.status == VehicleStatusEnum.WITH_PASSENGER else 0.0
            # TODO: Implement _normalize_distance properly
            # For now assuming simple normalization
            vehicle_obs[idx, 5] = np.float32(
                vehicle.remaining_distance / 10000.0
            )  # Assuming max distance ~10km for normalization
        request_obs = np.zeros((self.config.max_pending_requests, 6), dtype=np.float32)
        dispatch_mask = np.zeros((self.config.max_pending_requests, self.num_vehicles + 1), dtype=np.float32)
        pricing_mask = np.zeros((self.config.max_pending_requests,), dtype=np.float32)
        for idx, request in enumerate(self.pending_requests):
            if request is None:
                dispatch_mask[idx, 0] = 1.0
                continue
            request_obs[idx, 0] = float(request.pickup_node_id) / max(1, self.num_nodes)
            request_obs[idx, 1] = float(request.dropoff_node) / max(1, self.num_nodes)
            request_obs[idx, 2] = np.float32(
                np.clip(request.wait_time / 10.0, 0.0, 1.0)
            )  # Hardcoded normalizer for now
            request_obs[idx, 3] = np.float32(request.distance / 10000.0)
            request_obs[idx, 4] = np.float32(
                np.clip(request.price / max(request.base_price, 1e-6), 0.0, self.config.max_price_multiplier)
            )
            request_obs[idx, 5] = 1.0 if request.status == RequestStatusEnum.ACCEPTED else 0.0
            dispatch_mask[idx, 0] = 1.0
            if request.status == RequestStatusEnum.AWAITING_PRICE:
                pricing_mask[idx] = 1.0
            if request.status == RequestStatusEnum.ACCEPTED:
                for v_idx, vehicle in enumerate(self.vehicles):
                    feasible = (
                        vehicle.status == VehicleStatusEnum.IDLE
                        and vehicle.battery >= self.config.min_battery_for_assignment
                    )
                    dispatch_mask[idx, v_idx + 1] = 1.0 if feasible else 0.0
        observation = {
            "global": global_features,
            "supply_demand_ratio": supply_demand,
            "vehicles": vehicle_obs,
            "pending_requests": request_obs,
            "dispatch_mask": dispatch_mask,
            "pricing_mask": pricing_mask,
        }
        return observation

    def _get_info(self):
        info = {
            "step": self.current_step,
            "metrics": dict(self.metrics),
            "active_rides": len(self.active_rides),
            "pending_requests": sum(1 for r in self.pending_requests if r is not None),
            "map_name": self.map_name,
            "node_ids": list(self.node_ids),
        }
        return info

    def _supply_demand_ratio(self) -> float:
        idle = sum(1 for v in self.vehicles if v.status == VehicleStatusEnum.IDLE)
        outstanding = sum(1 for r in self.pending_requests if r is not None and r.status != RequestStatusEnum.CANCELLED)
        outstanding += len(self.active_rides)
        outstanding = max(outstanding, 1)
        ratio = idle / outstanding
        # normalized = ratio / self.config.supply_demand_normalizer
        normalized = ratio / 2.5  # Hardcoded normalizer
        return float(np.clip(normalized, 0.0, 10.0))
