from __future__ import annotations

import typing as t
from math import ceil
from pathlib import Path

import numpy as np

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
        self.reset_debug_and_interim()

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

        terminated = False
        truncated = self.current_step >= self.config.max_episode_steps

        observation, reward = self.get_observation(action)
        obs_gymnasium: dict[str, np.ndarray] = {
            "globals": observation["globals"],
            "supply_demand_ratio": observation["supply_demand_ratio"],
            "vehicles": observation["vehicles"].to_obs_numpy(),
            "pending_requests": observation["pending_requests"].to_obs_numpy(),
            "active_rides": observation["active_rides"].to_obs_numpy(),
            "dispatch_mask": observation["dispatch_mask"],
            "pricing_mask": observation["pricing_mask"],
        }
        try:
            self.observation_space.contains(obs_gymnasium)
        except Exception as e:
            self.error_msg = str(e)
            terminated = True
        return obs_gymnasium, reward, terminated, bool(truncated), self.info
