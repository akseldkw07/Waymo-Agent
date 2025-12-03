from __future__ import annotations

import typing as t
from pathlib import Path


from ..data_classes.dataclasses import ActiveRide, EnvConfig, RequestState, VehicleState
from .mixin import ActionMixin, ObservationSpaceMixin, OSMnxWrapperMixin, RenderingMixin, TransitionMixin


class RideShareEnv(RenderingMixin, ObservationSpaceMixin, ActionMixin, TransitionMixin, OSMnxWrapperMixin):
    """Gymnasium environment for simulating ride-share dispatch on OSMnx road networks."""

    def __init__(self, config: EnvConfig | None = None, render_mode: t.Literal["human", "ansi"] = "human"):
        super().__init__()
        self.config = config or EnvConfig()
        self.render_mode = render_mode
        self.map_dir = Path(self.config.map_dir).expanduser()
        self.map_name = self.config.map_name

        self.build_graph()

        # Initialize num_vehicles based on graph size
        self.num_vehicles = int(self.num_nodes * self.config.vehicle_per_node)
        self.num_vehicles = max(1, self.num_vehicles)

        self.build_observation_space()
        self.build_action_space()
        self.vehicles: list[VehicleState] = []
        self.pending_requests: list[RequestState] = []
        self.active_rides: dict[int, ActiveRide] = {}
        self.next_request_id: int = 0
        self.current_step: int = 0
        self.day_of_week: int = 0
        self.time_of_day: float = 0.0
        self.weather_idx: int = 0
        self.metrics: dict[str, float] = {}
        self.error_msg: str = ""

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """
        Initialize the environment to start a new episode.
        """

        super().reset(seed=seed, options=options)
        self.reset_globals()
        self.metrics = {
            "completed_rides": 0.0,
            "rejected_requests": 0.0,
            "cancelled_requests": 0.0,
            "overflow_requests": 0.0,
            "earned_revenue": 0.0,
            "energy_spent": 0.0,
            "distance_travelled": 0.0,
        }
        self.initialize_vehicles()
        # self.pending_requests = [None] * self.config.max_pending_requests
        self.active_rides.clear()
        self.next_request_id = 0
        self._spawn_requests()
        observation = self.get_observation()
        return observation, self.info

    def step(self, action: dict):
        try:
            self._validate_action(action)
        except Exception as e:
            self.error_msg = str(e)
            terminated = True
            return self.observation_curr, 0.0, terminated, False, self.info
        reward = 0.0

        # TODO implement transition logic

        self._advance_clock()
        terminated = False
        truncated = self.current_step >= self.config.max_episode_steps
        self.observation_curr = self.get_observation()
        return self.observation_curr, float(reward), terminated, bool(truncated), self.info
