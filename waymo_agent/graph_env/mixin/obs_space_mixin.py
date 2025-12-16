from __future__ import annotations

import typing as t
import warnings

import numpy as np
import pandas as pd
from gymnasium import spaces

from waymo_agent.constants import DATA_DIR
from waymo_agent.data_classes import *
from waymo_agent.data_classes import ActiveRideDF, VehicleDF
from waymo_agent.data_classes.supply_demand_df import SupplyDemandDF
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
        shape = (5,)
        low = np.array([-1.0, -1.0, -1.0, -1.0, 0.0]).astype(np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0, len(WeatherEnum) - 1]).astype(np.float32)
        return {"shape": shape, "low": low, "high": high}

    def define_observation_space(self):
        """
        TODO no need to breakup enums, let the user handle

        Define the observation space for the environment.
            - globals: day of week (sin, cos), time of day (sin, cos), weather_idx (5,) #TODO stretch goal: add demand per node, next step speed limits
            - supply_demand_ratio: current supply-demand ratio, vehicle-per-node, lambda-per-node (3,)
            - vehicles: (num_vehicles, 7) -> [loc_x_norm, loc_y_norm, battery, status(enum)]
            - pending_requests: (max_pending_requests, 9) -> [pickup_x_norm, pickup_y_norm, dropoff_x_norm, dropoff_y_norm, distance_meters, est_cost, max_wait_time, wait_time, status_bool] # TODO add max wait time before
            - active_rides: (num_vehicles, 9) -> [pickup_x_norm, pickup_y_norm, dropoff_x_norm, dropoff_y_norm, price, est_cost, total_trip_distance, trip_distance_remaining, pickup_distance_remaining]
            - dispatch_mask: (num_vehicles) -> 1 if vehicle can be dispatched to request, else 0
            - pricing_mask: (max_pending_requests) -> 1 if request needs pricing decision, else 0
        """
        num_v = self.num_vehicles

        observation_space = spaces.Dict(
            {
                "globals": spaces.Box(**self.global_space_config),
                "supply_demand_ratio": spaces.Box(**SupplyDemandDF.space_config(self.config)),
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
        self._remove_requests = []

        self.bc_row.update({"step": self.current_step, "timestamp": self.time_dt})
        real_requests = RequestDF.spawn_requests(self, self.config.max_new_requests_per_step)
        filler_requests = RequestDF.generate_empty(
            num_rows=self.config.max_pending_requests - len(real_requests), dt=self.time_dt
        )
        # Requests, Vehicles, Rides
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            requests = RequestDF(pd.concat([real_requests, filler_requests], ignore_index=True).reset_index(drop=True))
        if len(requests) > self.config.max_pending_requests:
            f_prune = requests.index >= self.config.max_pending_requests
            requests.drop(index=requests.index[f_prune], inplace=True)

        vehicles = init_vehicle_df(self)
        rides = init_active_ride_df(self, vehicles)

        assert (
            len(requests) == self.config.max_pending_requests
        ), f"Requests len {len(requests)} != max_pending_requests {self.config.max_pending_requests}"
        assert (
            len(vehicles) == self.num_vehicles == vehicles.f_idle.sum()
        ), f"VehicleDF len {len(vehicles)} != num_vehicles {self.num_vehicles} != f_idle sum {vehicles.f_idle.sum()}"
        assert len(rides) == len(vehicles), f"Rides len {len(rides)} != Vehicles len {len(vehicles)}"
        assert rides.f_valid.sum() == 0, f"Rides f_valid sum {rides.f_valid.sum()} != 0 on reset"

        sd_ratio = get_sd_ratio(self.config, requests, vehicles)
        assert (
            sd_ratio.shape == SupplyDemandDF.space_config(self.config)["shape"]
        ), f"SupplyDemandDF ratio shape {sd_ratio.shape} != (3,)"
        self.bc_row.update({"supply_demand_ratio": sd_ratio[0]})

        # Observation
        observation_full: ObservationDict = {
            "globals": self.MetaState,
            "supply_demand_ratio": sd_ratio,
            "vehicles": VehicleDF(vehicles.astype(VehicleDF.target_dtypes)),
            "pending_requests": RequestDF(requests.astype(RequestDF.target_dtypes)),
            "active_rides": ActiveRideDF(rides.astype(ActiveRideDF.target_dtypes)),
            "dispatch_mask": vehicles.f_idle.astype(np.int8),
            "pricing_mask": requests.f_awaiting_price.astype(np.int8),
        }
        validate_keys(ObservationDict, observation_full)
        self.observation_curr = observation_full
        self.observation_prev = observation_full  # TODO should I be doing this?
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.observation_space.contains(observation_full)

        # Breadcrumbs
        self.bc_row.update({"error_msg": self.error_msg, "rewards": 0})
        self.append_breadcrumbs()
        return observation_full

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
            "rewards": 0.0,
            "energy_spent": 0.0,
            "distance_travelled": 0.0,
        }
        self.calc_time_normed()

    @property
    def info(self) -> EnvInfoTypedDict:
        return {"step": self.TimeVerbose, "metrics": self.metrics, "map_name": self.map_name, "error": self.error_msg}
