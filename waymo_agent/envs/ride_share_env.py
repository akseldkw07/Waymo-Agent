from __future__ import annotations

import typing as t
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from .dataclasses import ActiveRide, EnvConfig, RequestState, VehicleState
from .mixin import RideShareDynamicsMixin, RideShareGraphMixin, RideShareObservationMixin, RideShareRenderingMixin


class RideShareEnv(RideShareRenderingMixin, RideShareObservationMixin, RideShareDynamicsMixin, RideShareGraphMixin):
    """Gymnasium environment that mirrors the proposal architecture for the Waymo RL project."""

    def __init__(self, config: EnvConfig | None = None, render_mode: t.Literal["human", "ansi"] = "human"):
        super().__init__()
        self.config = config or EnvConfig()
        self.render_mode = render_mode
        self.map_dir = Path(self.config.map_dir).expanduser()
        self.map_name = self.config.map_name
        self.graph: nx.DiGraph = nx.DiGraph()
        self.node_ids: list[str] = []
        self.node_index: dict[str, int] = {}
        self.node_metadata: list[dict[str, Any]] = []
        self.node_coords: np.ndarray = np.zeros((0, 2), dtype=np.float64)
        self.distance_matrix: np.ndarray | None = None
        self.speed_matrix: np.ndarray | None = None
        self.graph_edges: list[tuple[int, int]] = []
        self.charging_nodes: list[int] = []
        self._build_graph()
        self.num_nodes = int(self.node_coords.shape[0])
        if self.num_nodes == 0:
            raise ValueError("Map must contain at least one node.")
        self.config.num_nodes = self.num_nodes
        self._build_observation_space()
        self._build_action_space()
        self.np_random: np.random.Generator = np.random.default_rng()
        self.vehicles: list[VehicleState] = []
        self.pending_requests: list[RequestState | None] = []
        self.active_rides: dict[int, ActiveRide] = {}
        self.next_request_id: int = 0
        self.current_step: int = 0
        self.day_of_week: int = 0
        self.time_of_day: float = 0.0
        self.weather_idx: int = 0
        self.metrics: dict[str, float] = {}
        self._mpl = None
        self._figure = None
        self._axes = None
        self._line_cls = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.current_step = 0
        self.day_of_week = int(self.np_random.integers(7))
        self.time_of_day = float(self.np_random.uniform(6.0, 9.0))
        self.weather_idx = int(self.np_random.integers(len(self.WEATHER_STATES)))
        self.metrics = {
            "completed_rides": 0.0,
            "rejected_requests": 0.0,
            "cancelled_requests": 0.0,
            "overflow_requests": 0.0,
            "earned_revenue": 0.0,
            "energy_spent": 0.0,
            "distance_travelled": 0.0,
        }
        self._initialize_vehicles()
        self.pending_requests = [None] * self.config.max_pending_requests
        self.active_rides.clear()
        self.next_request_id = 0
        self._spawn_requests(initial=True)
        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self, action: dict):
        self._validate_action(action)
        reward = 0.0
        rejected = self._apply_pricing_decisions(action)
        reward += rejected * self.config.penalty_rejected
        assigned = self._apply_dispatch_decisions(action)
        reward += assigned * 0.0
        repositioned = self._apply_repositioning(action)
        reward += repositioned * 0.0
        reward += self._apply_charging(action)
        ride_reward = self._advance_vehicle_tasks()
        reward += ride_reward
        cancellations = self._advance_waiting_requests()
        reward += cancellations * self.config.penalty_cancelled
        overflow = self._spawn_requests()
        reward += overflow * self.config.penalty_overflow
        self._advance_clock()
        terminated = self.current_step >= self.config.max_episode_steps
        truncated = False
        observation = self._get_observation()
        info = self._get_info()
        return observation, float(reward), bool(terminated), bool(truncated), info
