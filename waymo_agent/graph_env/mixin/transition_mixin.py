from __future__ import annotations


from waymo_agent.data_classes.space_dicts import ObservationDict

from ...data_classes.dataclasses import ActiveRide
from .interface import GraphMixinInterface


class RewardMixin(GraphMixinInterface):
    """
    Mixin responsible for calculating rewards based on environment transitions.

    This includes:
        - Amortized rewards for completed rides + pricing decisions.
        - Penalties for rejected/cancelled requests.
        - Costs associated with vehicle operations (e.g., repositioning, idling).
    """

    def _ride_reward_schedule(self, active_ride_curr: ActiveRide, active_ride_prev: ActiveRide) -> float:
        """
        Calculate the reward for a active ride.

        Uses an amortized reward structure:
            - Reward is given based on the fraction of the trip completed in the current step.
            - Encourages efficient and timely ride completions.

        Deterministic
        """
        ...


class TransitionMixin(RewardMixin):
    """
    Mixin responsible for handling environment transitions.

    This includes:
    - Applying pricing and dispatch actions.
    - Progressing vehicle tasks (rides, repositioning, charging).
    - Managing request lifecycles (spawning, waiting, cancellations).
    - Advancing the simulation clock.

    NOTE there's a mix of deterministic and stochastic transitions. They will be explicitly noted.
    """

    def _spawn_requests(self):
        """
        Generate new ride requests. Each request is associated with a customer and a pickup/dropoff node.

        The process is to sample from all nodes and create requests based on the lambda rate per node.

        Stochastic
        """

    def _advance_clock(self):
        """
        Advance the simulation clock by one time step.

        Deterministic
        """
        self.current_step += 1
        self.time_of_day += self.config.minutes_per_step / 60.0
        if self.time_of_day >= 24.0:
            self.time_of_day -= 24.0
            self.day_of_week = (self.day_of_week + 1) % 7

        self.calc_time_normed()

    def _update_battery(self):
        """
        Charge / drain vehicle batteries based on charging status and movement.

        Deterministic

        NOTE battery life not yet implemented
        """

    def _move_vehicles(self):
        """
        Move vehicles along their paths based on their current tasks (rides, repositioning).

        Deterministic
        """

    def _update_vehicle_states(self):
        """
        Update vehicle states based on their current tasks and positions.
            - if vehicle arrives at charging station, transition to CHARGING (not implemented)
            - if vehicle battery full, transition to IDLE
            - if vehicle arrives at pickup, transition to WITH_PASSENGER
            - if vehicle arrives at dropoff, transition to IDLE
            - if vehicle arrives at reposition target, transition to IDLE
            - if vehicle dispatched to pickup, transition to TO_PICKUP

        Deterministic
        """

    def _update_requests(self):
        """
        Update the status of all pending requests, including wait times and cancellations.
            - when price is offered, customer accepts or rejects
            - if price accepted in previous step, vehicle is dispatched (if possible)
            - if vehicle arrives at pickup, ride starts
            - if wait time exceeds threshold, request is cancelled

        Deterministic + Stochastic (acceptance)
        """

    def _update_metrics(self):
        """
        Update environment metrics based on completed rides, cancellations, rejections, etc.

        Deterministic
        """

    def _update_routes(self):
        """
        Recalculate routes for vehicles if necessary (e.g., due to dynamic traffic conditions).

        Deterministic (for now, as traffic not implemented)
        """

    def _assert_state_consistency(self):
        """
        Assert the internal consistency of the environment state.
            - No two vehicles occupy the same node.
            - Active rides correspond to vehicles that are currently occupied.
            - Pending requests do not exceed maximum allowed.
            - Vehicle states are valid.
            - self.active_rides aligns with self.observation_curr["active_rides"]

        Deterministic
        """

    def get_observation(self) -> ObservationDict:
        """
        Construct the observation dictionary from the current environment state. The returned observaion
        is post-transition, i.e., after all updates have been applied for the current step.

        Deterministic
        """
        ...
