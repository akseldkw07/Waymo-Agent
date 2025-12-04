from __future__ import annotations

import numpy as np

from ...data_classes.dataclasses import RequestStatusEnum, VehicleState, VehicleStatusEnum
from ..mixin.interface import GraphMixinInterface


class FleetActionMixin(GraphMixinInterface):
    """
    Mixin responsible for handling fleet actions and environment dynamics.

    This includes:
    - Pricing decisions: Adjusting prices for requests.
    - Dispatch decisions: Assigning vehicles to requests.
    - Repositioning: Moving idle vehicles to new locations.
    - Charging: Managing vehicle charging states.
    - Simulation progression: Moving vehicles, updating battery, and managing request lifecycles.
    """

    # ------------------------------------------------------------------ #
    # Pricing / dispatch decisions
    # ------------------------------------------------------------------ #
    def _apply_pricing_decisions(self, action: dict) -> int:
        """
        Apply pricing adjustments to pending requests.

        Args:
            action (dict): The action dictionary containing "price_adjustments".

        Returns:
            int: The number of requests rejected by customers due to price/probability.
        """
        price_adjustments = np.asarray(action["price_adjustments"], dtype=np.float32)
        rejected = 0
        for idx, request in enumerate(self.pending_requests):
            if request is None or request.status != RequestStatusEnum.AWAITING_PRICE:
                continue
            adj = float(price_adjustments[idx])
            # multiplier = 1.0 + self.config.price_action_scale * float(adj)
            multiplier = 1.0 + 0.25 * float(adj)  # Hardcoded scale
            # multiplier = float(np.clip(multiplier, self.config.min_price_multiplier, self.config.max_price_multiplier))
            multiplier = float(np.clip(multiplier, 0.6, 2.0))  # Hardcoded limits
            request.price = request.est_cost * multiplier
            accept_prob = self._request_acceptance_probability(request)
            if self.np_random.random() < accept_prob:
                request.status = RequestStatusEnum.ACCEPTED
            else:
                request.status = RequestStatusEnum.CANCELLED
                rejected += 1
                self.metrics["rejected_requests"] += 1.0
        return rejected

    def _request_acceptance_probability(self, request: RequestState) -> float:
        # Simple acceptance model based on price
        # acceptance_base: float = 0.2
        # acceptance_price_weight: float = -2.0
        base_prob = 0.8  # Higher base acceptance
        price_ratio = request.price / request.base_price
        prob = base_prob - 0.5 * (price_ratio - 1.0)
        return float(np.clip(prob, 0.05, 1.0))

    def _apply_dispatch_decisions(self, action: dict) -> int:
        """
        Dispatch vehicles to accepted requests
        """
        dispatch_actions = action["dispatch"]
        assigned_count = 0

        # Iterate through requests and their dispatch actions
        for req_idx, vehicle_idx_plus_1 in enumerate(dispatch_actions):
            request = self.pending_requests[req_idx]

            # Check if request is valid and accepted
            if request is None or request.status != RequestStatusEnum.ACCEPTED:
                continue

            # vehicle_idx_plus_1 = 0 means no dispatch (or wait)
            if vehicle_idx_plus_1 == 0:
                continue

            vehicle_idx = vehicle_idx_plus_1 - 1
            if vehicle_idx >= len(self.vehicles):
                continue

            vehicle: VehicleState = self.vehicles[vehicle_idx]

            # Check if vehicle is available
            if vehicle.status != VehicleStatusEnum.IDLE:
                continue

            # Assign vehicle
            vehicle.status = VehicleStatusEnum.TO_PICKUP
            vehicle.target_node = request.pickup_node_id
            vehicle.ride_id = request.request_id

            # Calculate distance to pickup
            dist_to_pickup = self.distance(vehicle.node_id, request.pickup_node_id)
            if dist_to_pickup == float("inf"):
                # Cannot reach pickup
                vehicle.status = VehicleStatusEnum.IDLE
                vehicle.target_node = None
                vehicle.ride_id = None
                continue

            vehicle.remaining_distance = dist_to_pickup

            request.status = RequestStatusEnum.ASSIGNED
            assigned_count += 1

        return assigned_count

    def _apply_repositioning(self, action: dict) -> int:
        """
        Reposition idle vehicles based on action inputs
        """
        reposition_actions = action["reposition"]
        repositioned_count = 0

        for v_idx, target_node in enumerate(reposition_actions):
            if v_idx >= len(self.vehicles):
                break

            vehicle = self.vehicles[v_idx]

            # Only reposition idle vehicles
            if vehicle.status != VehicleStatusEnum.IDLE:
                continue

            # If target is current node, do nothing
            if target_node == vehicle.node_id:
                continue

            vehicle.status = VehicleStatusEnum.IDLE
            vehicle.target_node = target_node

            dist = self.distance(vehicle.node_id, target_node)
            if dist == float("inf"):
                vehicle.status = VehicleStatusEnum.IDLE
                vehicle.target_node = None
                continue

            vehicle.remaining_distance = dist

            repositioned_count += 1

        return repositioned_count

    def _apply_charging(self, action: dict) -> float:
        """
        Charge vehicles based on placement at charging nodes
        """
        toggle_charging = action["toggle_charging"]
        reward_cost = 0.0

        for v_idx, toggle in enumerate(toggle_charging):
            if v_idx >= len(self.vehicles):
                break

            vehicle = self.vehicles[v_idx]

            # Start charging
            if toggle == 1 and vehicle.status == VehicleStatusEnum.IDLE:
                # Check if at charging station
                if self.node_ids[vehicle.node_id] in self.charging_nodes:
                    vehicle.status = VehicleStatusEnum.CHARGING

            # Stop charging
            elif toggle == 1 and vehicle.status == VehicleStatusEnum.CHARGING:
                vehicle.status = VehicleStatusEnum.IDLE

        return reward_cost

    # ------------------------------------------------------------------ #
    # Simulation progression
    # ------------------------------------------------------------------ #
    def _advance_vehicle_tasks(self) -> float:
        """
        Progress all vehicles based on their current tasks (rides, repositioning, charging)
        Return total reward from ride completions
        """
        total_reward = 0.0

        # Assume constant speed for simplicity or derived from map
        # speed_mps = 10.0 # meters per second ~ 36 km/h
        # distance_step = speed_mps * self.config.minutes_per_step * 60
        distance_step = 500.0  # 500 meters per minute

        for vehicle in self.vehicles:
            if vehicle.status == VehicleStatusEnum.IDLE:
                continue

            elif vehicle.status == VehicleStatusEnum.CHARGING:
                self._charge_vehicle(vehicle)

            elif vehicle.status == VehicleStatusEnum.IDLE:
                self._progress_reposition(vehicle, distance_step)

            elif vehicle.status == VehicleStatusEnum.TO_PICKUP:
                self._progress_to_pickup(vehicle, distance_step)

            elif vehicle.status == VehicleStatusEnum.WITH_PASSENGER:
                reward = self._progress_ride(vehicle, distance_step)
                total_reward += reward

        return total_reward

    def _progress_to_pickup(self, vehicle: VehicleState, dist_step: float):
        if vehicle.remaining_distance <= dist_step:
            # Arrived at pickup
            vehicle.travel_distance_total += vehicle.remaining_distance
            vehicle.battery -= vehicle.remaining_distance * self.config.battery_consumption_per_km / 1000.0
            vehicle.remaining_distance = 0.0
            vehicle.node_id = vehicle.target_node

            # Transition to WITH_PASSENGER
            # Find the request
            req_idx = -1
            for idx, r in enumerate(self.pending_requests):
                if r is not None and r.request_id == vehicle.ride_id:
                    req_idx = idx
                    break

            if req_idx != -1:
                request = self.pending_requests[req_idx]
                if request is None:
                    # Should not happen given the search above, but for type safety
                    vehicle.status = VehicleStatusEnum.IDLE
                    vehicle.target_node = None
                    return

                vehicle.status = VehicleStatusEnum.WITH_PASSENGER
                vehicle.target_node = request.dropoff_node
                vehicle.remaining_distance = request.distance

                # Create ActiveRide
                active_ride = ActiveRide(
                    ride_id=request.request_id,
                    vehicle_id=self.vehicles.index(vehicle),
                    pickup_node=request.pickup_node_id,
                    dropoff_node=request.dropoff_node,
                    price=request.price,
                    pickup_distance_remaining=0.0,
                    trip_distance_remaining=request.distance,
                    total_distance=request.distance,
                )
                self.active_rides[request.request_id] = active_ride

                # Clear request from pending
                self.pending_requests[req_idx] = None
            else:
                # Request disappeared? Go back to IDLE
                vehicle.status = VehicleStatusEnum.IDLE
                vehicle.target_node = None

        else:
            # Move towards pickup
            vehicle.remaining_distance -= dist_step
            vehicle.travel_distance_total += dist_step
            vehicle.battery -= dist_step * self.config.battery_consumption_per_km / 1000.0
            # We don't update vehicle.node strictly here to avoid map matching cost every step
            # Ideally we would project along the path

    def _progress_ride(self, vehicle: VehicleState, dist_step: float) -> float:
        """
        Progress a vehicle along its current ride
        Return reward if ride is completed, else 0.0
        """
        reward = 0.0
        assert vehicle.target_node is not None

        if vehicle.remaining_distance <= dist_step:
            # Arrived at dropoff
            vehicle.travel_distance_total += vehicle.remaining_distance
            vehicle.battery -= vehicle.remaining_distance * self.config.battery_consumption_per_km / 1000.0
            vehicle.remaining_distance = 0.0
            vehicle.node_id = vehicle.target_node
            vehicle.status = VehicleStatusEnum.IDLE
            vehicle.target_node = None

            # Complete ride
            if vehicle.ride_id in self.active_rides:
                ride = self.active_rides.pop(vehicle.ride_id)
                reward = ride.price
                self.metrics["completed_rides"] += 1.0
                self.metrics["earned_revenue"] += ride.price
                self.metrics["distance_travelled"] += ride.total_distance

            vehicle.ride_id = None

        else:
            # Move towards dropoff
            vehicle.remaining_distance -= dist_step
            vehicle.travel_distance_total += dist_step
            vehicle.battery -= dist_step * self.config.battery_consumption_per_km / 1000.0

            if vehicle.ride_id in self.active_rides:
                self.active_rides[vehicle.ride_id].trip_distance_remaining -= dist_step

        return reward

    def _progress_reposition(self, vehicle: VehicleState, dist_step: float):
        """
        Move the vehicle towards its repositioning target
        """
        if vehicle.remaining_distance <= dist_step:
            vehicle.travel_distance_total += vehicle.remaining_distance
            vehicle.battery -= vehicle.remaining_distance * self.config.battery_consumption_per_km / 1000.0
            vehicle.remaining_distance = 0.0
            vehicle.node_id = vehicle.target_node
            vehicle.status = VehicleStatusEnum.IDLE
            vehicle.target_node = None
        else:
            vehicle.remaining_distance -= dist_step
            vehicle.travel_distance_total += dist_step
            vehicle.battery -= dist_step * self.config.battery_consumption_per_km / 1000.0

    def _charge_vehicle(self, vehicle: VehicleState):
        """
        Charge the vehicle's battery based on charge rate and time step
        """
        charge_amount = self.config.charge_rate_per_minute * self.config.minutes_per_step
        vehicle.battery = min(1.0, vehicle.battery + charge_amount)
        if vehicle.battery >= 0.95:
            vehicle.status = VehicleStatusEnum.IDLE

    # ------------------------------------------------------------------ #
    # Request lifecycle helpers
    # ------------------------------------------------------------------ #
    def _advance_waiting_requests(self) -> int:
        """
        Advance wait times for all pending requests
        Cancel requests that have waited too long
        Return number of cancellations
        """
        cancellations = 0
        for idx, request in enumerate(self.pending_requests):
            if request is None:
                continue
            if request.status in (RequestStatusEnum.CANCELLED, RequestStatusEnum.COMPLETED):
                self.pending_requests[idx] = None
                continue
            request.wait_time += self.config.minutes_per_step
            if request.status == RequestStatusEnum.ACCEPTED and request.wait_time > 30.0:
                request.status = RequestStatusEnum.CANCELLED
                cancellations += 1
                self.metrics["cancelled_requests"] += 1.0
                self.pending_requests[idx] = None
        return cancellations

    def _spawn_requests(self) -> int:
        """
        Spawn new ride requests based on lambda rates
        """
        # Simple Poisson process
        # Total lambda for the system
        total_lambda = self.config.lambda_per_node * self.num_nodes

        # Expected number of new requests
        expected_new = total_lambda * self.config.minutes_per_step
        num_new = self.np_random.poisson(expected_new)

        overflow = 0
        for _ in range(num_new):
            # Find empty slot
            slot = -1
            for i in range(len(self.pending_requests)):
                if self.pending_requests[i] is None:
                    slot = i
                    break

            if slot == -1:
                overflow += 1
                self.metrics["overflow_requests"] += 1.0
                continue

            # Sample pickup and dropoff
            pickup = self.np_random.integers(0, self.num_nodes).astype(int)
            dropoff = self._sample_dropoff(pickup)

            self._create_request(slot, pickup, dropoff)

        return overflow

    def _sample_dropoff(self, pickup_idx: int) -> int:
        if self.num_nodes <= 1:
            return pickup_idx
        # Simplified uniform sampling for now if weights are missing
        dropoff = self.np_random.integers(0, self.num_nodes)
        while dropoff == pickup_idx:
            dropoff = self.np_random.integers(0, self.num_nodes)
        return dropoff.astype(int)

    def _create_request(self, slot: int, pickup: int, dropoff: int):
        if pickup == dropoff:
            dropoff = (dropoff + 1) % self.num_nodes
        distance = self.distance(pickup, dropoff)
        # base_price = self.config.base_fare + distance * self.config.distance_fare
        base_price = 2.5 + (distance / 1000.0) * 1.5  # Hardcoded fare model
        # customer_bias = float(self.np_random.normal(0.0, self.config.customer_bias_std))
        customer_bias = float(self.np_random.normal(0.0, 0.5))
        request = RequestState(
            request_id=self.next_request_id,
            pickup_node=pickup,
            dropoff_node=dropoff,
            distance=distance,
            base_price=base_price,
            price=base_price,
            customer_bias=customer_bias,
            status=RequestStatusEnum.AWAITING_PRICE,
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
