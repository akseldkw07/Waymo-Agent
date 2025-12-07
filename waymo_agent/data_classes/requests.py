from __future__ import annotations

import enum
import typing as t

import numpy as np
import pandas as pd

from waymo_agent.data_classes.config import EnvConfig
from waymo_agent.data_classes.enriched_df_base import EnrichedDF
from waymo_agent.graph_env.cost_reward import compute_operating_cost
from waymo_agent.graph_env.df_utils import validate_typed_df_keys

if t.TYPE_CHECKING:
    from waymo_agent.graph_env.mixin import ObservationSpaceMixin, TransitionMixin

CURR_ID: int = 0


class RequestStatusEnum(enum.IntEnum):
    CANCEL_EXCEED_WAIT_TIME = -2
    REJECTED = -1

    AWAITING_PRICE = 0

    ACCEPTED = 1
    ASSIGNED = 3
    COMPLETED = 5


class RequestDF(EnrichedDF):
    request_id: np.ndarray
    request_dt: np.datetime64  # pd.Timestamp

    pickup_node_id: np.ndarray
    pickup_x_norm: np.ndarray
    pickup_y_norm: np.ndarray

    cust_id: np.ndarray
    cust_bias: np.ndarray
    cust_temperature: np.ndarray

    dropoff_node_id: np.ndarray
    dropoff_x_norm: np.ndarray
    dropoff_y_norm: np.ndarray

    route_nodes: np.ndarray  # list[int] per row → dtype=object
    curr_start_node: np.ndarray  # current start node in route_nodes
    curr_end_node: np.ndarray  # current end node in route_nodes
    route_dist_on_edge: np.ndarray  # how much distance has been traveled on the CURRENT edge
    distance_meters: np.ndarray

    est_cost: np.ndarray
    price: np.ndarray

    max_wait_time: np.ndarray
    wait_time: np.ndarray
    status: np.ndarray  # RequestStatusEnum value
    target_dtypes: t.ClassVar[dict[str, type]] = {
        "request_id": np.int64,
        "request_dt": np.datetime64,  # datetime64[ns]
        "pickup_node_id": np.int64,
        "pickup_x_norm": np.float64,
        "pickup_y_norm": np.float64,
        "cust_id": np.int64,
        "cust_bias": np.float64,
        "cust_temperature": np.float64,
        "dropoff_node_id": np.int64,
        "dropoff_x_norm": np.float64,
        "dropoff_y_norm": np.float64,
        "route_nodes": np.object_,  # list[int] per row → dtype=object
        "curr_start_node": np.int64,
        "curr_end_node": np.int64,
        "route_dist_on_edge": np.float64,
        "distance_meters": np.float64,
        "est_cost": np.float64,
        "price": np.float64,
        "max_wait_time": np.float64,
        "wait_time": np.float64,
        "status": np.int64,
    }

    @property
    def f_real_requests(self) -> np.ndarray:
        return self.request_id != EnvConfig().invalid_id

    @property
    def f_awaiting_price(self) -> np.ndarray:
        raw = self.status == RequestStatusEnum.AWAITING_PRICE
        ret = raw & self.f_real_requests
        return ret

    @property
    def f_need_dispatch(self) -> np.ndarray:
        raw = self.status == RequestStatusEnum.ACCEPTED
        ret = raw & self.f_real_requests
        return ret

    @property
    def f_reject_expired(self) -> np.ndarray:
        raw = (self.status == RequestStatusEnum.CANCEL_EXCEED_WAIT_TIME) | (self.status == RequestStatusEnum.REJECTED)
        ret = raw & self.f_real_requests
        return ret

    @property
    def f_en_route(self) -> np.ndarray:
        """
        Requests associated with vehicle
        """
        raw = self.status == RequestStatusEnum.ASSIGNED
        ret = raw & self.f_real_requests
        return ret

    @property
    def f_completed(self) -> np.ndarray:
        raw = self.status == RequestStatusEnum.COMPLETED
        ret = raw & self.f_real_requests
        return ret

    @property
    def f_inactive(self) -> np.ndarray:
        """
        Requests that are no longer active (cancelled, rejected, completed)
        """
        raw = self.f_reject_expired | self.f_completed
        ret = raw & self.f_real_requests
        return ret

    # Training view (based on define_observation_space docstring):
    # [pickup_x_norm, pickup_y_norm, dropoff_x_norm, dropoff_y_norm,
    #  distance_meters, est_cost, max_wait_time, wait_time] + status_one_hot
    cols_to_keep: t.ClassVar[list[str]] = [
        "pickup_x_norm",
        "pickup_y_norm",
        "dropoff_x_norm",
        "dropoff_y_norm",
        "distance_meters",
        "est_cost",
        "max_wait_time",
        "wait_time",
    ]
    enum_fields: t.ClassVar[dict[str, type[enum.IntEnum]]] = {"status": RequestStatusEnum}

    @staticmethod
    def column_order() -> list[str]:
        return [
            "request_id",
            "request_dt",
            "pickup_node_id",
            "pickup_x_norm",
            "pickup_y_norm",
            "cust_id",
            "cust_bias",
            "cust_temperature",
            "dropoff_node_id",
            "dropoff_x_norm",
            "dropoff_y_norm",
            "route_nodes",
            "curr_start_node",
            "curr_end_node",
            "route_dist_on_edge",
            "distance_meters",
            "est_cost",
            "price",
            "max_wait_time",
            "wait_time",
            "status",
        ]

    @staticmethod
    def space_config(config: EnvConfig):
        cls = RequestDF
        width = cls.calc_width()
        shape = (config.max_pending_requests, width)
        row_high = np.ones((width,), dtype=np.float32)
        low = np.zeros(shape, dtype=np.float32)
        high = np.tile(row_high, (config.max_pending_requests, 1))
        return {"shape": shape, "low": low, "high": high}

    @classmethod
    def spawn_requests(cls, env: ObservationSpaceMixin | TransitionMixin) -> RequestDF:
        """
        Generate new ride requests. Each request is associated with a customer and a pickup/dropoff node.

        The process is to sample from all nodes and create requests based on the lambda rate per node.

        Stochastic
        """
        config = env.config

        f_requests = np.random.poisson(lam=env.node_df["lambda"]).astype(bool)

        global CURR_ID
        new_ids = np.arange(CURR_ID, CURR_ID + f_requests.sum())
        CURR_ID += f_requests.sum()

        cust_samples = env.cust_df.sample(f_requests.sum(), replace=True)
        cust_samples.columns = ["cust_id", "cust_bias", "cust_temperature"]
        dropoff_df = env.node_df.sample(f_requests.sum(), replace=True, ignore_index=True)[
            ["node_id", "x_norm", "y_norm"]
        ]
        dropoff_df.columns = ["dropoff_node_id", "dropoff_x_norm", "dropoff_y_norm"]

        request_df = env.node_df[f_requests][["node_id", "x_norm", "y_norm"]].reset_index(drop=True)
        request_df.columns = ["pickup_node_id", "pickup_x_norm", "pickup_y_norm"]
        request_df["request_id"] = new_ids

        df = pd.concat([request_df, cust_samples.reset_index(drop=True), dropoff_df.reset_index(drop=True)], axis=1)
        request_df = RequestDF(df)

        request_df["route_dist_on_edge"] = 0.0
        path = env.calc_shortest_paths(request_df["pickup_node_id"].to_list(), request_df["dropoff_node_id"].to_list())
        request_df[["route_nodes", "distance_meters"]] = path
        request_df["curr_start_node"] = request_df["pickup_node_id"]
        request_df["curr_end_node"] = [nodes[1] if len(nodes) > 1 else nodes[0] for nodes in request_df["route_nodes"]]
        request_df["est_cost"] = compute_operating_cost(request_df.distance_meters, config)
        request_df["price"] = np.nan
        request_df["max_wait_time"] = config.max_wait_time
        request_df["wait_time"] = 0.0
        request_df["status"] = RequestStatusEnum.AWAITING_PRICE
        request_df["request_dt"] = pd.to_datetime(env.time_dt)

        request_df = RequestDF(request_df[RequestDF.column_order()])

        validate_typed_df_keys(request_df, RequestDF)

        return request_df

    @classmethod
    def generate_empty(cls, num_rows: int = 0) -> RequestDF:
        cfg = EnvConfig()
        df = super().generate_empty(num_rows=num_rows)
        df["request_id"] = cfg.invalid_id
        ret = cls(df)
        validate_typed_df_keys(ret, RequestDF)
        return ret
