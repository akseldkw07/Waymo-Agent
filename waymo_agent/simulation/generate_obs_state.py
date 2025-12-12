import typing as t

import numpy as np

from waymo_agent.data_classes import ActiveRideDF, EnvConfig, RequestDF, VehicleDF, SupplyDemandDF

if t.TYPE_CHECKING:
    from waymo_agent.graph_env.mixin import ObservationSpaceMixin, TransitionMixin


def init_vehicle_df(env: "ObservationSpaceMixin") -> VehicleDF:
    ret = VehicleDF.generate_empty(env)

    return ret


def init_active_ride_df(env: "ObservationSpaceMixin | TransitionMixin", vehicles: VehicleDF):
    ret = ActiveRideDF.generate_empty(num_rows=env.num_vehicles)
    ret.vehicle_id = vehicles.vehicle_id.copy()

    return ret


def get_sd_ratio(config: EnvConfig, requests: RequestDF, vehicles: VehicleDF) -> np.ndarray:

    ratio = SupplyDemandDF.get_sd_ratio(config, requests, vehicles).to_obs_numpy()[0]
    return ratio
