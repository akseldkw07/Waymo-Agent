from __future__ import annotations

import typing as t
import warnings
from math import ceil
from pathlib import Path

from waymo_agent.data_classes.space_dicts import prune_obs_dict_gymnasium

from ..data_classes.config import EnvConfig
from ..data_classes.config_plot import PlotConfig
from .mixin import ActionMixin, ObservationSpaceMixin, OSMnxWrapperMixin, RenderingMixin, TransitionMixin

if t.TYPE_CHECKING:
    from waymo_agent.data_classes import ActionDict


class RideShareEnv(RenderingMixin, OSMnxWrapperMixin, ObservationSpaceMixin, ActionMixin, TransitionMixin):
    """Gymnasium environment for simulating ride-share dispatch on OSMnx road networks."""

    def __init__(
        self,
        config: EnvConfig | None = None,
        plot_config: PlotConfig | None = None,
        render_mode: t.Literal["human", "ansi"] = "human",
    ):
        super().__init__()
        self.config = config or EnvConfig()
        self.plt_cfg = plot_config or PlotConfig()
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
        # self.reset_debug_and_interim()

        observation_full = self.reset_observation()
        obs_pruned = prune_obs_dict_gymnasium(observation_full)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.observation_space.contains(obs_pruned)
        return obs_pruned, self.info

    def step(self, action: ActionDict):  # type: ignore
        try:
            self._validate_action(action)
        except Exception as e:
            self.error_msg = str(e)
            terminated = True
            return self.observation_curr, 0.0, terminated, False, self.info
        reward = 0.0

        terminated = False
        truncated = self.current_step >= self.config.max_episode_steps

        observation_full, reward = self.get_observation(action)
        obs_pruned = prune_obs_dict_gymnasium(observation_full)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.observation_space.contains(obs_pruned)
        except Exception as e:
            self.error_msg = str(e)
            terminated = True
        return obs_pruned, reward, terminated, bool(truncated), self.info
