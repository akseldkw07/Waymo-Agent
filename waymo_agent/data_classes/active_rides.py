from __future__ import annotations

import enum
import typing as t

import numpy as np
import pandas as pd

from waymo_agent.data_classes.config import EnvConfig
from waymo_agent.data_classes.enriched_df_base import EnrichedDF
from waymo_agent.graph_env.df_utils import validate_typed_df_keys

if t.TYPE_CHECKING:
    from . import RequestDF


class ActiveRideDF(EnrichedDF):
    ride_id: np.ndarray
    vehicle_id: np.ndarray

    pickup_node: np.ndarray
    pickup_x_norm: np.ndarray
    pickup_y_norm: np.ndarray

    dropoff_node: np.ndarray
    dropoff_x_norm: np.ndarray
    dropoff_y_norm: np.ndarray

    price: np.ndarray
    est_cost: np.ndarray

    total_trip_distance_meters: np.ndarray
    trip_distance_remaining_meters: np.ndarray
    pickup_distance_remaining_meters: np.ndarray

    route_nodes: np.ndarray  # list[int] per row
    curr_start_node: np.ndarray
    curr_end_node: np.ndarray
    route_dist_on_edge: np.ndarray  # how much distance has been traveled on the CURRENT edge

    complete: np.ndarray

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
    target_dtypes: t.ClassVar[dict[str, type]] = {
        "ride_id": np.int64,
        "vehicle_id": np.int64,
        "pickup_node": np.int64,
        "pickup_x_norm": np.float32,
        "pickup_y_norm": np.float32,
        "dropoff_node": np.int64,
        "dropoff_x_norm": np.float32,
        "dropoff_y_norm": np.float32,
        "price": np.float32,
        "total_trip_distance_meters": np.float32,
        "trip_distance_remaining_meters": np.float32,
        "pickup_distance_remaining_meters": np.float32,
        "route_nodes": np.object_,
        "curr_start_node": np.int64,
        "curr_end_node": np.int64,
        "route_dist_on_edge": np.float32,
        "complete": np.bool_,
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
                "curr_start_node": np.zeros(len(requests), dtype=np.int64),
                "curr_end_node": np.zeros(len(requests), dtype=np.int64),
                "route_dist_on_edge": np.zeros(len(requests), dtype=np.float32),
                "complete": np.full(len(requests), False, dtype=bool),
            }
        ).reset_index(drop=True)
        df = df.astype(cls.target_dtypes)
        ret = cls(df)
        validate_typed_df_keys(ret, ActiveRideDF)
        ret.validate_dtypes()
        ret.sort_values("vehicle_id", inplace=True)
        return ret
