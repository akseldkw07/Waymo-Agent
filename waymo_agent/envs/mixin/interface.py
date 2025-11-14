from __future__ import annotations

import typing as t
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

"""Gymnasium environment that mirrors the proposal architecture for the Waymo RL project."""

from ..dataclasses import ActiveRide, EnvConfig, RequestState, VehicleState


class GraphMixinInterface(gym.Env, ABC):
    """
    Base interface for RideShare environment mixins that provide graph and geometry functionality.
    TODO remove | None from type hints
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 6}
    WEATHER_STATES: tuple[str, ...] = ("clear", "rain", "snow")
    render_mode: str | t.Literal["human", "ansi"]
    map_dir: Path
    map_name: str | None
    config: EnvConfig
    vehicles: list[VehicleState]
    pending_requests: list[RequestState | None]
    active_rides: dict[int, ActiveRide]
    observation_space: spaces.Space
    action_space: spaces.Space

    graph: nx.DiGraph
    node_ids: list[str]
    node_index: dict[str, int]
    node_metadata: list[dict[str, Any]]
    node_coords: np.ndarray
    distance_matrix: np.ndarray | None
    speed_matrix: np.ndarray | None
    graph_edges: list[tuple[int, int]]
    charging_nodes: list[int]
    num_nodes: int
    np_rand: np.random.Generator
    next_request_id: int
    current_step: int
    day_of_week: int
    time_of_day: float
    weather_idx: int
    metrics: dict[str, float]
    _mpl: t.Any | None
    _figure = Figure | None
    _axes = Axes | None
    _line_cls = Line2D | None

    # TODO
    @abstractmethod
    def _distance(self, *args, **kwargs) -> float: ...

    @abstractmethod
    def _nearest_charging_station(self, *args, **kwargs) -> int: ...

    @abstractmethod
    def _normalize_distance(self, *args, **kwargs) -> float: ...

    @abstractmethod
    def _supply_demand_ratio(self, *args, **kwargs) -> float: ...

    @abstractmethod
    def _normalize_node(self, *args, **kwargs) -> float: ...
