import typing as t

import numpy as np

from waymo_agent.data_classes import ActiveRideDF, EnvConfig, RequestDF, VehicleDF

if t.TYPE_CHECKING:
    from waymo_agent.graph_env.mixin import ObservationSpaceMixin, TransitionMixin


def init_vehicle_df(env: "ObservationSpaceMixin") -> VehicleDF:
    ret = VehicleDF.generate_empty(env)
    return ret


def init_active_ride_df(env: "ObservationSpaceMixin | TransitionMixin", vehicles: VehicleDF) -> ActiveRideDF:
    ret = ActiveRideDF.generate_empty(num_rows=env.num_vehicles)
    ret.vehicle_id = vehicles.vehicle_id.copy()

    assert isinstance(ret, ActiveRideDF)
    return ret


def get_sd_ratio(config: EnvConfig, requests: RequestDF, vehicles: VehicleDF) -> np.ndarray:
    # TODO somehow make this more robust - better naming schema or something like that

    ratio = np.array(
        [
            vehicles.f_available.sum() / requests.f_awaiting_price.sum().clip(min=1),
            config.vehicle_per_node,
            config.lambda_per_node,
        ],
        dtype=np.float32,
    )
    return ratio
