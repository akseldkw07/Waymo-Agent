from __future__ import annotations

import typing as t
from math import ceil
from pathlib import Path

from waymo_agent.data_classes import ActionDict, EnvConfig

from .mixin import ActionMixin, ObservationSpaceMixin, OSMnxWrapperMixin, RenderingMixin, TransitionMixin


class RideShareEnv(RenderingMixin, OSMnxWrapperMixin, ObservationSpaceMixin, ActionMixin, TransitionMixin):
    """Gymnasium environment for simulating ride-share dispatch on OSMnx road networks."""

    def __init__(self, config: EnvConfig | None = None, render_mode: t.Literal["human", "ansi"] = "human"):
        super().__init__()
        self.config = config or EnvConfig()
        self.render_mode = render_mode
        self.map_dir = Path(self.config.map_dir).expanduser()
        self.map_name = self.config.map_name

        self.build_graph()
        self.load_aux_data()

        # Initialize num_vehicles based on graph size
        self.num_vehicles = ceil(self.num_nodes * self.config.vehicle_per_node)
        self.num_vehicles = min(self.config.max_vehicles, self.num_vehicles)

        self.define_observation_space()
        self.define_action_space()
        self.reset_globals()  # Initialize time and other global variables before resetting observation
        self.reset_observation()

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # type: ignore
        """
        Initialize the environment to start a new episode.
        """

        super().reset(seed=seed, options=options)
        self.reset_globals()

        observation = self.reset_observation()
        return observation, self.info

    def step(self, action: ActionDict):  # type: ignore
        try:
            self._validate_action(action)
        except Exception as e:
            self.error_msg = str(e)
            terminated = True
            return self.observation_curr, 0.0, terminated, False, self.info
        # TODO implement transition logic
        reward = 0.0

        self._advance_clock()
        terminated = False
        truncated = self.current_step >= self.config.max_episode_steps
        self.observation_curr = self.get_observation(action)
        return self.observation_curr, float(reward), terminated, bool(truncated), self.info
