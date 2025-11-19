from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces

if t.TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    AxesLike = Axes
    FigureLike = Figure
    Line2DType = type[Line2D]
else:
    AxesLike = FigureLike = t.Any  # type: ignore
    Line2DType = t.Any  # type: ignore

"""Gymnasium environment that mirrors the proposal architecture for the Waymo RL project."""

import logging

from ..dataclasses import ActiveRide, EnvConfig, RequestState, VehicleState


class GraphMixinInterface(gym.Env, ABC):
    """
    Base interface for RideShare environment mixins that provide graph and geometry functionality.
    TODO remove | None from type hints
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 6}
    WEATHER_STATES: tuple[str, ...] = ("clear", "rain", "snow")
    render_mode: str | t.Literal["human", "ansi"]
    map_name: str | None
    config: EnvConfig
    vehicles: list[VehicleState]
    pending_requests: list[RequestState | None]
    active_rides: dict[int, ActiveRide]
    observation_space: spaces.Space
    action_space: spaces.Space
    num_vehicles: int

    graph: nx.MultiDiGraph
    node_ids: list[str]
    node_index: dict[str, int]
    node_metadata: list[dict[str, t.Any]]
    node_coords: np.ndarray
    node_lambdas: np.ndarray
    node_lambda_weights: np.ndarray
    graph_edges: list[tuple[int, int]]
    edge_segments: list[t.Any]
    np_rand: np.random.Generator
    next_request_id: int
    current_step: int
    day_of_week: int
    time_of_day: float
    weather_idx: int
    metrics: dict[str, float]

    @property
    def num_nodes(self) -> int: ...

    @property
    def charging_nodes(self) -> dict[str, int]: ...

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(self.__class__.__name__)

    # TODO remove these
    @abstractmethod
    def _distance(self, *args, **kwargs) -> float: ...

    @abstractmethod
    def nearest_charging_station(self, *args, **kwargs) -> int: ...

    # @abstractmethod
    # def _normalize_distance(self, *args, **kwargs) -> float: ...

    # @abstractmethod
    # def _supply_demand_ratio(self, *args, **kwargs) -> float: ...
