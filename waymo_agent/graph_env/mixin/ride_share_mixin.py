from __future__ import annotations

import numpy as np

from ..dataclasses import ActiveRide, RequestState, RequestStatus, VehicleState
from .interface import GraphMixinInterface


class FleetActionMixin(GraphMixinInterface):
    """Action handling and environment dynamics grounded on NetworkX distances."""

    # ------------------------------------------------------------------ #
    # Pricing / dispatch decisions
    # ------------------------------------------------------------------ #
    def _apply_pricing_decisions(self, action: dict) -> int:
        price_adjustments = np.asarray(action["price_adjustments"], dtype=np.float32)
        rejected = 0
        for idx, request in enumerate(self.pending_requests):
            if request is None or request.status != RequestStatus.AWAITING_PRICE:
                continue
            adj = float(price_adjustments[idx])
            multiplier = 1.0 + self.config.price_action_scale * float(adj)
            multiplier = float(np.clip(multiplier, self.config.min_price_multiplier, self.config.max_price_multiplier))
            request.price = request.base_price * multiplier
            accept_prob = self._request_acceptance_probability(request)
            if self.np_random.random() < accept_prob:
                request.status = RequestStatus.ACCEPTED
            else:
                request.status = RequestStatus.CANCELLED
                rejected += 1
                self.metrics["rejected_requests"] += 1.0
        return rejected

    def _apply_dispatch_decisions(self, action: dict) -> int:
        """
        TODO
        Dispatch vehicles to accepted requests
        Follow the pdf template
        """

    def _apply_repositioning(self, action: dict) -> int:
        """
        TODO
        Reposition idle vehicles based on action inputs
        Implement in a networkx grounded manner
        """

    def _apply_charging(self, action: dict) -> float:
        """
        TODO
        Charge vehicles based on placement at charging nodes
        """

    # ------------------------------------------------------------------ #
    # Simulation progression
    # ------------------------------------------------------------------ #
    def _advance_vehicle_tasks(self) -> float:
        """
        TODO
        Progress all vehicles based on their current tasks (rides, repositioning, charging)
        Return total reward from ride completions
        """
        return total_reward

    def _progress_ride(self, vehicle: VehicleState, ride: ActiveRide) -> float:
        """
        TODO
        Progress a vehicle along its current ride
        Return reward if ride is completed, else 0.0
        """
        return reward

    def _progress_reposition(self, vehicle: VehicleState):
        """
        TODO
        Move the vehicle towards its repositioning target
        Update battery and distance travelled metrics
        """
        ...

    def _charge_vehicle(self, vehicle: VehicleState):
        """
        TODO
        Charge the vehicle's battery based on charge rate and time step
        Update vehicle status if fully charged
        """
        ...

    # ------------------------------------------------------------------ #
    # Request lifecycle helpers
    # ------------------------------------------------------------------ #
    def _advance_waiting_requests(self) -> int:
        """
        TODO
        Advance wait times for all pending requests
        Cancel requests that have waited too long
        Return number of cancellations
        """
        cancellations = 0
        for idx, request in enumerate(self.pending_requests):
            if request is None:
                continue
            if request.status in (RequestStatus.CANCELLED, RequestStatus.COMPLETED):
                self.pending_requests[idx] = None
                continue
            request.wait_time += self.config.minutes_per_step
            if request.status == RequestStatus.ACCEPTED and request.wait_time > 30.0:
                request.status = RequestStatus.CANCELLED
                cancellations += 1
                self.metrics["cancelled_requests"] += 1.0
                self.pending_requests[idx] = None
        return cancellations

    def _spawn_requests(self, initial: bool = False) -> int:
        """
        TODO
        Spawn new ride requests based on lambda rates
        """

    def _sample_dropoff(self, pickup_idx: int) -> int:
        if self.num_nodes <= 1:
            return pickup_idx
        weights = np.array(self.node_lambda_weights, copy=True)
        weights[pickup_idx] = 0.0
        total = weights.sum()
        if total <= 0.0:
            uniform = np.ones(self.num_nodes, dtype=np.float64)
            uniform[pickup_idx] = 0.0
            weights = uniform / uniform.sum()
        else:
            weights /= total
        return int(self.np_random.choice(self.num_nodes, p=weights))

    def _create_request(self, slot: int, pickup: int, dropoff: int):
        if pickup == dropoff:
            dropoff = (dropoff + 1) % self.num_nodes
        distance = self._distance(pickup, dropoff)
        base_price = self.config.base_fare + distance * self.config.distance_fare
        customer_bias = float(self.np_random.normal(0.0, self.config.customer_bias_std))
        request = RequestState(
            request_id=self.next_request_id,
            pickup_node=pickup,
            dropoff_node=dropoff,
            distance=distance,
            base_price=base_price,
            price=base_price,
            customer_bias=customer_bias,
        )
        self.pending_requests[slot] = request
        self.next_request_id += 1

    # ------------------------------------------------------------------ #
    # Clock + bookkeeping
    # ------------------------------------------------------------------ #
    def _advance_clock(self):
        self.current_step += 1
        self.time_of_day += self.config.minutes_per_step / 60.0
        if self.time_of_day >= 24.0:
            self.time_of_day -= 24.0
            self.day_of_week = (self.day_of_week + 1) % 7
