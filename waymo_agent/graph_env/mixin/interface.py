from __future__ import annotations

import datetime as dt
import logging
import typing as t
from abc import ABC

import gymnasium as gym
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from waymo_agent.data_classes import ActionDict, CustomerDF, EnvConfig, EnvInfoTypedDict, ObservationDict
from waymo_agent.data_classes.config_plot import PlotConfig
from waymo_agent.data_classes.enriched_df_base import validate_typed_df_keys
from waymo_agent.data_classes.metrics import TimeSeriesMetricsDF
from waymo_agent.data_classes.requests import RequestDF
from waymo_agent.simulation.dt_utils import embed_datetime_to_circle

AxesLike = Axes
FigureLike = Figure
Line2DType = type[Line2D]

"""Gymnasium environment that mirrors the proposal architecture for the Waymo RL project."""


class GymEnvInterface(gym.Env, ABC):
    """
    Base interface for RideShare environment mixins that provide graph and geometry functionality.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 6}
    config: EnvConfig
    plt_cfg: PlotConfig
    render_mode: str | t.Literal["human", "ansi"]
    map_name: str
    num_vehicles: int

    # Graph
    graph: nx.MultiDiGraph
    node_ids: list[int]
    node_index: dict[int, int]
    node_coords: np.ndarray
    l2_recovery: dict[str, float]

    # Dataframes
    node_df: pd.DataFrame
    edge_df: pd.DataFrame
    cust_df: CustomerDF

    # Observation and Action History
    observation_prev: ObservationDict
    observation_curr: ObservationDict
    # action_prev: ActionDict
    action_curr: ActionDict

    # Time
    current_step: int
    time_dt: dt.datetime
    day_of_week_norm: tuple[float, float]
    time_of_day_norm: tuple[float, float]
    weather_idx: int

    # Metrics and Debugging
    _breadcrumbs: list[dict[str, float | int | str | dt.datetime]]
    _remove_requests: list[RequestDF]
    bc_row: dict[str, float | int | str | dt.datetime]
    metrics: dict[str, float]  # This is like "current" snapshot of accumulated metrics
    error_msg: str

    @property
    def size(self) -> tuple[int, int]: ...

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(self.__class__.__name__)

    @property
    def info(self) -> EnvInfoTypedDict: ...

    @property
    def MetaState(self):
        meta_state = np.array([*self.day_of_week_norm, *self.time_of_day_norm, self.weather_idx])
        return meta_state

    @property
    def TimeVerbose(self):
        return {
            "step": self.current_step,
            "datetime": self.time_dt,
            "day_of_week": self.day_of_week_norm,
            "time_of_day": self.time_of_day_norm,
            "weather_idx": self.weather_idx,
        }

    @property
    def breadcrumbs(self) -> TimeSeriesMetricsDF:
        """Get breadcrumbs time series as a DataFrame."""
        return TimeSeriesMetricsDF.from_breadcrumbs(self._breadcrumbs)

    @property
    def DiscardedRequests(self) -> RequestDF:
        """Get discarded requests as a DataFrame."""
        return RequestDF(pd.concat(self._remove_requests, ignore_index=True))

    def append_breadcrumbs(self):
        """Append current metrics to breadcrumbs time series."""

        validate_typed_df_keys(self.bc_row, TimeSeriesMetricsDF)
        self._breadcrumbs.append(self.bc_row.copy())
        self.bc_row = {}

    def calc_time_normed(self):
        # Cyclical time features

        self.day_of_week_norm = embed_datetime_to_circle(self.time_dt, "dow")
        self.time_of_day_norm = embed_datetime_to_circle(self.time_dt, "time")

    def calc_shortest_paths(self, *args, **kwargs) -> pd.DataFrame: ...

    def nearest_node_id(self, *args, **kwargs) -> t.Sequence[int]: ...
