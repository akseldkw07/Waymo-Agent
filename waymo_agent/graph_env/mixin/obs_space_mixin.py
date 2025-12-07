from __future__ import annotations

import typing as t

import numpy as np
import pandas as pd
from gymnasium import spaces

from waymo_agent.constants import DATA_DIR
from waymo_agent.data_classes import *
from waymo_agent.data_classes import ActiveRideDF, VehicleDF
from waymo_agent.simulation import init_vehicle_df, random_dt
from waymo_agent.simulation.generate_obs_state import get_sd_ratio, init_active_ride_df

from .interface import GymEnvInterface


class ObservationSpaceMixin(GymEnvInterface):
    """
    Mixin responsible for constructing the observation space and generating observations.

    This mixin handles:
    - Initializing vehicle states.
    - Defining the observation space
    """

    def load_aux_data(self):
        """
        Load auxiliary data
            - Customer dataframe
        """
        self.cust_df = t.cast(CustomerDF, pd.read_parquet(DATA_DIR / CustomerDFGenator.filename))

    @property
    def global_space_config(self):
        """ """
        shape = (7,)
        low = np.array([-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0])
        high = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        return {"shape": shape, "low": low, "high": high}

    def define_observation_space(self):
        """
        Define the observation space for the environment.
            - globals: day of week (sin, cos), time of day (sin, cos), weather_one_hot(3) (7,) #TODO stretch goal: add demand per node, next step speed limits
            - supply_demand_ratio: current supply-demand ratio, vehicle-per-node, lambda-per-node (3,)
            - vehicles: (num_vehicles, 7) -> [loc_x_norm, loc_y_norm, battery, status_one_hot(4)]
            - pending_requests: (max_pending_requests, 9) -> [pickup_x_norm, pickup_y_norm, dropoff_x_norm, dropoff_y_norm, distance_meters, est_cost, max_wait_time, wait_time, status_one_hot(1)] # TODO add max wait time before
            - active_rides: (num_vehicles, 9) -> [pickup_x_norm, pickup_y_norm, dropoff_x_norm, dropoff_y_norm, price, est_cost, total_trip_distance, trip_distance_remaining, pickup_distance_remaining]
            - dispatch_mask: (num_vehicles) -> 1 if vehicle can be dispatched to request, else 0
            - pricing_mask: (max_pending_requests) -> 1 if request needs pricing decision, else 0
        """
        num_v = self.num_vehicles

        observation_space = spaces.Dict(
            {
                "globals": spaces.Box(**self.global_space_config),
                "supply_demand_ratio": spaces.Box(low=0.0, high=10.0, shape=(3,)),
                "vehicles": spaces.Box(**VehicleDF.space_config(self.config, num_v)),
                "pending_requests": spaces.Box(**RequestDF.space_config(self.config)),
                "active_rides": spaces.Box(**ActiveRideDF.space_config(self.config, num_v)),
                "dispatch_mask": spaces.Box(low=0.0, high=1.0, shape=(num_v,)),
                "pricing_mask": spaces.Box(low=0.0, high=1.0, shape=(self.config.max_pending_requests,)),
            }
        )
        validate_keys(ObservationDict, observation_space)
        self.observation_space = observation_space

    def reset_observation(self):
        """
        Build observation space following env.reset() or env.__init__()

        Ie, there's no action to pass or previous observation to reference.
        """
        self.bc_row, self._breadcrumbs = {}, []
        self.bc_row.update({"step": self.current_step, "timestamp": self.time_dt})
        real_requests = RequestDF.spawn_requests(self)
        filler_requests = RequestDF.generate_empty(num_rows=self.config.max_pending_requests - len(real_requests))
        requests = RequestDF(pd.concat([real_requests, filler_requests], ignore_index=True).reset_index(drop=True))
        vehicles = init_vehicle_df(self)
        rides = init_active_ride_df(self, vehicles)

        sd_ratio = get_sd_ratio(self.config, requests, vehicles)
        self.bc_row.update({"supply_demand_ratio": sd_ratio[0]})
        observation: ObservationDict = {
            "globals": self.MetaState,
            "supply_demand_ratio": sd_ratio,
            "vehicles": vehicles,
            "pending_requests": requests,
            "active_rides": rides,
            "dispatch_mask": vehicles.f_available.astype(np.float32),
            "pricing_mask": requests.f_awaiting_price.astype(np.float32),
        }
        validate_keys(ObservationDict, observation)
        self.observation_curr = observation
        self.observation_prev = observation  # TODO should I be doing this?
        self.observation_space.contains(observation)  # This will fail due to dataframes instead of numpy arrays
        # TODOD should I .to_numpy() here?
        self.bc_row.update({"error_msg": self.error_msg, "earned_revenue": 0})
        self.append_breadcrumbs()
        return observation

    def reset_globals(self):
        """
        Reset step, time, weather, error message, metrics, and breadcrumbs

        Gets called on env.reset() and __init__()
        """
        self.current_step = 0
        self.time_dt = random_dt()
        self.weather_idx = int(self.np_random.integers(len(WeatherEnum)))
        self.error_msg = ""
        self.metrics = {  # TODO think harder about metrics to track
            "completed_rides": 0.0,
            "rejected_requests": 0.0,
            "cancelled_requests": 0.0,
            "overflow_requests": 0.0,
            "earned_revenue": 0.0,
            "energy_spent": 0.0,
            "distance_travelled": 0.0,
        }
        self.calc_time_normed()

    @property
    def info(self) -> EnvInfoTypedDict:
        return {"step": self.TimeVerbose, "metrics": self.metrics, "map_name": self.map_name, "error": self.error_msg}
