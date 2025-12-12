from __future__ import annotations

import enum
import typing as t

import numpy as np
import pandas as pd

from waymo_agent.data_classes.config import EnvConfig
from waymo_agent.data_classes.enriched_df_base import EnrichedDF, validate_typed_df_keys

if t.TYPE_CHECKING:
    from . import RequestDF


class ActiveRideDF(EnrichedDF):
    ride_id: pd.Series
    vehicle_id: pd.Series

    pickup_node: pd.Series
    pickup_x_norm: pd.Series
    pickup_y_norm: pd.Series

    dropoff_node: pd.Series
    dropoff_x_norm: pd.Series
    dropoff_y_norm: pd.Series

    price: pd.Series
    est_cost: pd.Series

    total_trip_distance_meters: pd.Series
    trip_distance_remaining_meters: pd.Series
    pickup_distance_remaining_meters: pd.Series

    route_nodes: pd.Series  # list[int] per row
    curr_start_node: pd.Series
    curr_end_node: pd.Series
    route_dist_on_edge: pd.Series  # how much distance has been traveled on the CURRENT edge

    is_complete: pd.Series

    @property
    def f_valid(self):
        return (self.ride_id != EnvConfig().invalid_id).to_numpy(dtype=bool)

    @property
    def f_has_route(self) -> np.ndarray:
        return np.array([len(rn) >= 2 for rn in self.route_nodes], dtype=bool)

    # Training view (define_observation_space):
    # [pickup_x_norm, pickup_y_norm, dropoff_x_norm, dropoff_y_norm,
    #  price, total_trip_distance_meters, trip_distance_remaining_meters,
    #  pickup_distance_remaining_meters] -> 8 dims, no enums.
    cols_to_keep: t.ClassVar[list[str]] = [
        "pickup_x_norm",
        "pickup_y_norm",
        "dropoff_x_norm",
        "dropoff_y_norm",
        "price",
        "est_cost",
        "total_trip_distance_meters",
        "trip_distance_remaining_meters",
        "pickup_distance_remaining_meters",
    ]
    enum_fields: t.ClassVar[dict[str, type[enum.IntEnum]]] = {}
    target_dtypes = {
        "ride_id": np.int64,
        "vehicle_id": np.int64,
        "pickup_node": np.int64,
        "pickup_x_norm": np.float32,
        "pickup_y_norm": np.float32,
        "dropoff_node": np.int64,
        "dropoff_x_norm": np.float32,
        "dropoff_y_norm": np.float32,
        "price": np.float32,
        "est_cost": np.float32,
        "total_trip_distance_meters": np.float32,
        "trip_distance_remaining_meters": np.float32,
        "pickup_distance_remaining_meters": np.float32,
        "route_nodes": np.object_,
        "curr_start_node": np.int64,
        "curr_end_node": np.int64,
        "route_dist_on_edge": np.float32,
        "is_complete": np.bool_,
    }

    @classmethod
    def space_config(cls, config: EnvConfig, num_rides: int):
        width = cls.calc_width()
        shape = (num_rides, width)
        row_high = np.ones((width,), dtype=np.float32)
        low = np.zeros(shape, dtype=np.float32)
        high = np.tile(row_high, (num_rides, 1))
        return {"shape": shape, "low": low, "high": high}

    @classmethod
    def from_requests(cls, requests: RequestDF, vehicle_ids: np.ndarray | pd.Series) -> ActiveRideDF:
        df = pd.DataFrame(
            {
                "ride_id": requests["request_id"].to_numpy(),
                "vehicle_id": vehicle_ids,
                "pickup_node": requests["pickup_node_id"].to_numpy(),
                "pickup_x_norm": requests["pickup_x_norm"].to_numpy(),
                "pickup_y_norm": requests["pickup_y_norm"].to_numpy(),
                "dropoff_node": requests["dropoff_node_id"].to_numpy(),
                "dropoff_x_norm": requests["dropoff_x_norm"].to_numpy(),
                "dropoff_y_norm": requests["dropoff_y_norm"].to_numpy(),
                "price": requests["price"].to_numpy(),  # NOTE this field is added dynamically post-init
                "est_cost": requests["est_cost"].to_numpy(),
                "total_trip_distance_meters": requests["distance_meters"].to_numpy(),
                "trip_distance_remaining_meters": requests["distance_meters"].to_numpy(),
                "pickup_distance_remaining_meters": requests["distance_meters"].to_numpy(),
                "route_nodes": requests["route_nodes"].to_numpy(),
                "curr_start_node": requests["route_nodes"].apply(lambda x: x[0]),
                "curr_end_node": requests["route_nodes"].apply(lambda x: x[1] if len(x) > 1 else x[0]),
                "route_dist_on_edge": np.zeros(len(requests), dtype=np.float32),
                "is_complete": np.full(len(requests), False, dtype=bool),
            }
        ).reset_index(drop=True)
        df = df.astype(cls.target_dtypes)
        ret = cls(df)
        validate_typed_df_keys(ret, ActiveRideDF)
        ret.validate_dtypes()
        ret.sort_values("vehicle_id", inplace=True)
        return ret
