from __future__ import annotations

import typing as t
from abc import ABC

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from waymo_agent.data_classes.space_dicts import ActionDict, ObservationDict

AxesLike = Axes
FigureLike = Figure
Line2DType = type[Line2D]

"""Gymnasium environment that mirrors the proposal architecture for the Waymo RL project."""

import logging

from ...data_classes.dataclasses import ActiveRide, EnvConfig, RequestState, VehicleState


class GraphMixinInterface(gym.Env, ABC):
    """
    Base interface for RideShare environment mixins that provide graph and geometry functionality.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 6}
    config: EnvConfig
    render_mode: str | t.Literal["human", "ansi"]
    map_name: str

    # Graph
    graph: nx.MultiDiGraph
    node_ids: list[int]
    node_index: dict[int, int]
    node_coords: np.ndarray

    # Vehicles
    vehicles: list[VehicleState]
    num_vehicles: int

    # Observation and Action Spaces
    observation_space: spaces.Dict  # type: ignore
    observation_curr: ObservationDict
    action_space: spaces.Dict  # type: ignore
    action_curr: ActionDict

    # Requests and Rides
    pending_requests: list[RequestState]
    active_rides: dict[int, ActiveRide]

    # Time and Metrics
    current_step: int
    day_of_week: int
    day_of_week_norm: tuple[float, float]
    time_of_day: float
    time_of_day_norm: tuple[float, float]
    weather_idx: int
    metrics: dict[str, float]
    error_msg: str

    @property
    def size(self) -> tuple[int, int]: ...

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(self.__class__.__name__)

    @property
    def info(self) -> dict[str, t.Any]: ...

    @property
    def TimeVerbose(self):
        return {
            "step": self.current_step,
            "date": {"raw": self.day_of_week, "normed": self.day_of_week_norm},
            "time": {"raw": self.time_of_day, "normed": self.time_of_day_norm},
            "weather_idx": self.weather_idx,
        }

    def calc_time_normed(self):
        # Cyclical time features

        precision = 4
        day_sin = np.sin(2 * np.pi * self.day_of_week / self.config.day_per_week).round(precision)
        day_cos = np.cos(2 * np.pi * self.day_of_week / self.config.day_per_week).round(precision)
        time_sin = np.sin(2 * np.pi * self.time_of_day / self.config.hours_per_day).round(precision)
        time_cos = np.cos(2 * np.pi * self.time_of_day / self.config.hours_per_day).round(precision)

        self.day_of_week_norm = (day_sin, day_cos)
        self.time_of_day_norm = (time_sin, time_cos)
