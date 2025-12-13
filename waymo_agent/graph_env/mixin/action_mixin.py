"""
Define action space and apply actions to the environment.
"""

from __future__ import annotations


import numpy as np
from gymnasium import spaces

from waymo_agent.data_classes import ActionDict, validate_keys

from .interface import GymEnvInterface


class ActionMixin(GymEnvInterface):
    """
    Mixing responsible for defining the action space and applying actions to the environment.

    This mixin handles:
    - Defining the action space.
    - Validating actions.
    """

    def define_action_space(self):
        """
        Define the action space for the agent.
            - Prices: (max_pending_requests,) -> continuous [0.0, max_price] (max_price set to np.inf)
            - Reposition: (num_vehicles, 2) -> continuous [-1.0, 1.0] for x and y direction
            - Dispatch: (max_pending_requests, num_vehicles) -> one-hot encoding for vehicle assignment
        """
        action_space = spaces.Dict(
            {
                "prices": spaces.Box(
                    low=0.0, high=self.config.max_price, shape=(self.config.max_pending_requests,), dtype=np.float64
                ),
                "reposition": spaces.Box(low=-1.0, high=1.0, shape=(self.num_vehicles, 2)),
                "dispatch": spaces.MultiDiscrete(
                    [self.num_vehicles + 1] * self.config.max_pending_requests,
                    start=[self.config.no_action_id] * self.config.max_pending_requests,
                ),
            }
        )
        validate_keys(ActionDict, action_space)
        self.action_space = action_space

    def _validate_action(self, action: ActionDict):
        validate_keys(ActionDict, action)
        if not self.action_space.contains(action):
            raise ValueError("Action is outside the defined action space.")
