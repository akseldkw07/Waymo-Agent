from __future__ import annotations

import numpy as np
from gymnasium import spaces

from waymo_agent.data_classes.space_dicts import ObservationDict, validate_keys
from waymo_agent.simulation.generate_vehicle import init_vehicles

from ...data_classes.dataclasses import ActiveRide, RequestState, VehicleState, VehicleStatusEnum
from .interface import GraphMixinInterface


class ObservationSpaceMixin(GraphMixinInterface):
    """
    Mixin responsible for constructing the observation space and generating observations.

    This mixin handles:
    - Initializing vehicle states.
    - Defining the observation space
    """

    @property
    def global_space_config(self):
        """ """
        shape = (7,)
        low = np.array([-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0])
        high = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        return {"shape": shape, "low": low, "high": high}

    def build_observation_space(self):
        """
        Define the observation space for the environment.
            - Global features: day of week, time of day, weather_one_hot(3) (7,) #TODO stretch goal: add demand per node, next step speed limits
            - Supply-demand ratio: (1,)
            - Vehicles: (num_vehicles, 7) -> [loc_x_norm, loc_y_norm, battery, status_one_hot(4)]
            - Pending requests: (max_pending_requests, 8) -> [pickup_x_norm, pickup_y_norm, dropoff_x_norm, dropoff_y_norm, wait_time_norm, distance_norm, price_norm, status_one_hot(2)]
            - Active rides: (num_vehicles, 8) -> [pickup_x_norm, pickup_y_norm, dropoff_x_norm, dropoff_y_norm, price, total_trip_distance, trip_distance_remaining, pickup_distance_remaining]
            - Vehicle availability mask: (num_vehicles) -> 1 if vehicle can be dispatched to request, else 0
            - Pricing mask: (max_pending_requests) -> 1 if request needs pricing decision, else 0
        """
        num_r = self.config.max_pending_requests
        num_v = self.num_vehicles

        observation_space = spaces.Dict(
            {
                "globals": spaces.Box(**self.global_space_config),
                "supply_demand_ratio": spaces.Box(low=0.0, high=10.0, shape=(1,)),
                "vehicles": spaces.Box(**VehicleState.space_config(self.config)),
                "pending_requests": spaces.Box(**RequestState.space_config(self.config)),
                "active_rides": spaces.Box(**ActiveRide.space_config(self.config)),
                "dispatch_mask": spaces.Box(low=0.0, high=1.0, shape=(num_v,)),
                "pricing_mask": spaces.Box(low=0.0, high=1.0, shape=(num_r,)),
            }
        )
        validate_keys(ObservationDict, observation_space)
        self.observation_space = observation_space

    def reset_globals(self):
        """
        Reset step, time & weather
        """
        self.current_step = 0
        self.day_of_week = int(self.np_random.integers(self.config.day_per_week))
        self.time_of_day = float(self.np_random.uniform(0.0, self.config.hours_per_day))
        self.weather_idx = int(self.np_random.integers(len(VehicleStatusEnum)))
        self.calc_time_normed()

    def initialize_vehicles(self):
        """
        Initialize vehicles based on configuration.
        """
        self.vehicles = init_vehicles(self.graph, self.num_vehicles, self.np_random)

    @property
    def info(self) -> dict:
        return {"step": self.TimeVerbose, "metrics": self.metrics, "map_name": self.map_name, "error": self.error_msg}
