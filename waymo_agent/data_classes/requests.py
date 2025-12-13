from __future__ import annotations

import datetime as dt
import enum
import typing as t

import numpy as np
import pandas as pd

from waymo_agent.data_classes.config import EnvConfig
from waymo_agent.data_classes.enriched_df_base import EnrichedDF, validate_typed_df_keys
from waymo_agent.graph_env.cost_reward import compute_operating_cost

if t.TYPE_CHECKING:
    from waymo_agent.data_classes.config_plot import PlotConfig
    from waymo_agent.graph_env.mixin import ObservationSpaceMixin, TransitionMixin

CURR_REQ_ID: int = 0
CFG = EnvConfig()


class RequestStatusEnum(enum.IntEnum):
    INVALID = -10

    CANCEL_EXCEED_WAIT_TIME = -2
    REJECTED = -1

    AWAITING_PRICE = 0

    ACCEPTED = 1
    ASSIGNED = 3
    COMPLETED = 5


class RequestDF(EnrichedDF):
    request_id: pd.Series
    request_dt: np.datetime64  # pd.Timestamp

    pickup_node_id: pd.Series
    pickup_x_norm: pd.Series
    pickup_y_norm: pd.Series

    cust_id: pd.Series
    cust_bias: pd.Series
    cust_temperature: pd.Series

    dropoff_node_id: pd.Series
    dropoff_x_norm: pd.Series
    dropoff_y_norm: pd.Series

    route_nodes: pd.Series  # list[int] per row → dtype=object
    curr_start_node: pd.Series  # current start node in route_nodes
    curr_end_node: pd.Series  # current end node in route_nodes
    route_dist_on_edge: pd.Series  # how much distance has been traveled on the CURRENT edge
    distance_meters: pd.Series

    est_cost: pd.Series
    price: pd.Series

    max_wait_time: np.timedelta64
    wait_time: pd.Series
    status: pd.Series  # RequestStatusEnum value
    target_dtypes = {
        "request_id": np.int64,
        "request_dt": np.datetime64,  # datetime64[ns]
        "cust_id": np.int64,
        "cust_bias": np.float64,
        "cust_temperature": np.float64,
        "est_cost": np.float64,
        "price": np.float64,
        "max_wait_time": np.timedelta64,
        "wait_time": np.timedelta64,
        "status": np.int64,
        "pickup_node_id": np.int64,
        "pickup_x_norm": np.float64,
        "pickup_y_norm": np.float64,
        "dropoff_node_id": np.int64,
        "dropoff_x_norm": np.float64,
        "dropoff_y_norm": np.float64,
        "route_nodes": np.object_,  # list[int] per row → dtype=object
        "curr_start_node": np.int64,
        "curr_end_node": np.int64,
        "route_dist_on_edge": np.float64,
        "distance_meters": np.float64,
    }
    default_vals: t.ClassVar[dict[str, t.Any]] = {
        "max_wait_time": (CFG.max_wait_time_minutes),
        "wait_time": 0,
        "price": -1.0,
        "est_cost": -1.0,
        "status": RequestStatusEnum.INVALID,
    }

    # Training view (based on define_observation_space docstring):
    # [pickup_x_norm, pickup_y_norm, dropoff_x_norm, dropoff_y_norm,
    #  distance_meters, est_cost, max_wait_time, wait_time] + status_one_hot
    cols_to_pass_to_model: t.ClassVar[list[str]] = [
        "pickup_x_norm",
        "pickup_y_norm",
        "dropoff_x_norm",
        "dropoff_y_norm",
        "distance_meters",
        "cust_bias",
        "cust_temperature",
        "est_cost",
        "max_wait_time",
        "wait_time",
        "status",
    ]

    @property
    def f_valid(self):
        try:
            ret = (self.request_id != CFG.invalid_id).to_numpy(dtype=bool)
        except Exception:
            ret = (self.status != RequestStatusEnum.INVALID).to_numpy(dtype=bool)
        return ret

    @property
    def f_awaiting_price(self):
        raw = self.status == RequestStatusEnum.AWAITING_PRICE
        ret = raw & self.f_valid
        return ret.to_numpy(dtype=bool)

    @property
    def f_need_dispatch(self):
        raw: pd.Series = self.status == RequestStatusEnum.ACCEPTED
        ret = raw & self.f_valid

        return ret.to_numpy(dtype=bool)

    @property
    def f_reject_expired(self) -> np.ndarray:
        raw = (self.status == RequestStatusEnum.CANCEL_EXCEED_WAIT_TIME) | (self.status == RequestStatusEnum.REJECTED)
        ret = raw & self.f_valid
        return ret.to_numpy(dtype=bool)

    @property
    def f_en_route(self) -> np.ndarray:
        """
        Requests associated with vehicle
        """
        raw = self.status == RequestStatusEnum.ASSIGNED
        ret = raw & self.f_valid
        return ret.to_numpy(dtype=bool)

    @property
    def f_completed(self) -> np.ndarray:
        raw = self.status == RequestStatusEnum.COMPLETED
        ret = raw & self.f_valid
        return ret.to_numpy(dtype=bool)

    @property
    def f_inactive(self) -> np.ndarray:
        """
        Requests that are no longer active (cancelled, rejected, completed)
        """
        raw = self.f_reject_expired | self.f_completed
        ret = np.array(raw & self.f_valid)
        return ret

    def f_plot_route(self, plt_cfg: PlotConfig) -> np.ndarray:
        valid_types_asint = [e.value for e in plt_cfg.request_status_colors.keys() if isinstance(e, RequestStatusEnum)]
        return np.isin(self.status, valid_types_asint)

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
    def generate_empty(cls, num_rows: int, dt: dt.datetime) -> RequestDF:
        if num_rows <= 0:
            return cls(pd.DataFrame(columns=cls.column_order()))
        df = super().generate_empty(num_rows=num_rows)
        df["request_dt"] = pd.to_datetime(np.nan)
        return df

    @classmethod
    def spawn_requests(cls, env: ObservationSpaceMixin | TransitionMixin, max_req: int | None = None) -> RequestDF:
        """
        Generate new ride requests. Each request is associated with a customer and a pickup/dropoff node.

        The process is to sample from all nodes and create requests based on the lambda rate per node.

        If max_req is not None, limits the number of spawned requests to max_req.

        Stochastic
        """
        global CURR_REQ_ID
        config = env.config

        f_requests = np.random.poisson(lam=env.node_df["lambda"]).astype(bool)

        # CURR_REQ_ID
        new_ids = np.arange(CURR_REQ_ID, CURR_REQ_ID + (num_req := f_requests.sum()))
        CURR_REQ_ID += num_req

        cust_samples = env.cust_df.sample(num_req, replace=True)
        cust_samples.columns = ["cust_id", "cust_bias", "cust_temperature"]
        dropoff_df = env.node_df.sample(num_req, replace=True, ignore_index=True)[["node_id", "x_norm", "y_norm"]]
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
        request_df["max_wait_time"] = pd.Timedelta(minutes=config.max_wait_time_minutes)
        request_df["wait_time"] = pd.Timedelta(0)
        request_df["status"] = RequestStatusEnum.AWAITING_PRICE
        request_df["request_dt"] = pd.to_datetime(env.time_dt)

        if max_req is not None and len(request_df) > max_req:
            request_df = request_df.sample(max_req)

        request_df = RequestDF(request_df[RequestDF.column_order()])
        validate_typed_df_keys(request_df, RequestDF)

        return request_df
