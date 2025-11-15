from __future__ import annotations

import numpy as np
from gymnasium import spaces

from ..dataclasses import RequestStatus, VehicleStatus
from .interface import GraphMixinInterface


class RideShareObservationMixin(GraphMixinInterface):
    """Observation and info helpers."""

    def _build_observation_space(self):
        num_v = self.config.num_vehicles
        num_r = self.config.max_pending_requests
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
        num_v = self.config.num_vehicles
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
        vehicle_obs = np.zeros((self.config.num_vehicles, 6), dtype=np.float32)
        for idx, vehicle in enumerate(self.vehicles):
            vehicle_obs[idx, 0] = self._normalize_node(vehicle.node)
            vehicle_obs[idx, 1] = np.float32(np.clip(vehicle.battery, 0.0, 1.0))
            vehicle_obs[idx, 2] = 1.0 if vehicle.status == VehicleStatus.IDLE else 0.0
            vehicle_obs[idx, 3] = 1.0 if vehicle.status == VehicleStatus.TO_PICKUP else 0.0
            vehicle_obs[idx, 4] = 1.0 if vehicle.status == VehicleStatus.WITH_PASSENGER else 0.0
            vehicle_obs[idx, 5] = np.float32(self._normalize_distance(vehicle.remaining_distance))
        request_obs = np.zeros((self.config.max_pending_requests, 6), dtype=np.float32)
        dispatch_mask = np.zeros((self.config.max_pending_requests, self.config.num_vehicles + 1), dtype=np.float32)
        pricing_mask = np.zeros((self.config.max_pending_requests,), dtype=np.float32)
        for idx, request in enumerate(self.pending_requests):
            if request is None:
                dispatch_mask[idx, 0] = 1.0
                continue
            request_obs[idx, 0] = self._normalize_node(request.pickup_node)
            request_obs[idx, 1] = self._normalize_node(request.dropoff_node)
            request_obs[idx, 2] = np.float32(np.clip(request.wait_time / self.config.wait_time_normalizer, 0.0, 1.0))
            request_obs[idx, 3] = np.float32(self._normalize_distance(request.distance))
            request_obs[idx, 4] = np.float32(
                np.clip(request.price / max(request.base_price, 1e-6), 0.0, self.config.max_price_multiplier)
            )
            request_obs[idx, 5] = 1.0 if request.status == RequestStatus.ACCEPTED else 0.0
            dispatch_mask[idx, 0] = 1.0
            if request.status == RequestStatus.AWAITING_PRICE:
                pricing_mask[idx] = 1.0
            if request.status == RequestStatus.ACCEPTED:
                for v_idx, vehicle in enumerate(self.vehicles):
                    feasible = (
                        vehicle.status == VehicleStatus.IDLE
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
        idle = sum(1 for v in self.vehicles if v.status == VehicleStatus.IDLE)
        outstanding = sum(1 for r in self.pending_requests if r is not None and r.status != RequestStatus.CANCELLED)
        outstanding += len(self.active_rides)
        outstanding = max(outstanding, 1)
        ratio = idle / outstanding
        normalized = ratio / self.config.supply_demand_normalizer
        return float(np.clip(normalized, 0.0, 10.0))
