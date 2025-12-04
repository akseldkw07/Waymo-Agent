"""
Define action space and apply actions to the environment.
"""

from __future__ import annotations

from gymnasium import spaces

from waymo_agent.data_classes.space_dicts import ActionDict, validate_keys

from .interface import GraphMixinInterface


class ActionMixin(GraphMixinInterface):
    """
    Mixing responsible for defining the action space and applying actions to the environment.

    This mixin handles:
    - Defining the action space.
    - Validating actions.
    """

    def build_action_space(self):
        """
        Define the action space for the agent.
            - Prices: (max_pending_requests,) -> continuous [0.0, 2.0]
            - Reposition: (num_vehicles, 2) -> continuous [-1.0, 1.0] for x and y direction
            - Dispatch: (max_pending_requests, num_vehicles) -> one-hot encoding for vehicle assignment
        """
        action_space = spaces.Dict(
            {
                "prices": spaces.Box(
                    low=0.0, high=2.0, shape=(self.config.max_pending_requests,)
                ),  # TODO fix max price
                "reposition": spaces.Box(low=-1.0, high=1.0, shape=(self.num_vehicles, 2)),
                "dispatch": spaces.MultiBinary((self.config.max_pending_requests, self.num_vehicles)),
            }
        )
        validate_keys(ActionDict, action_space)
        self.action_space = action_space

    def _validate_action(self, action: dict):
        validate_keys(ActionDict, action)
        if not self.action_space.contains(action):
            raise ValueError("Action is outside the defined action space.")
