from __future__ import annotations

import numpy as np

from ..dataclasses import ActiveRide, RequestState, RequestStatus, VehicleState, VehicleStatus
from .interface import GraphMixinInterface


class RideShareDynamicsMixin(GraphMixinInterface):
    """Action handling and dynamics for RideShare environments."""

    def _initialize_vehicles(self):
        self.vehicles = []
        for _ in range(self.config.num_vehicles):
            node = int(self.np_random.integers(self.num_nodes))
            battery = float(self.np_random.uniform(0.6, 1.0))
            self.vehicles.append(VehicleState(node=node, battery=battery, origin_node=node))

    def _validate_action(self, action: dict):
        if not self.action_space.contains(action):
            raise ValueError("Action is outside the defined action space.")

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
            accepted = self.np_random.random() < accept_prob
            if accepted:
                request.status = RequestStatus.ACCEPTED
            else:
                request.status = RequestStatus.CANCELLED
                rejected += 1
                self.metrics["rejected_requests"] += 1.0
        return rejected

    def _apply_dispatch_decisions(self, action: dict) -> int:
        dispatch = np.asarray(action["dispatch"], dtype=np.int32)
        assigned = 0
        for idx, request in enumerate(self.pending_requests):
            if request is None or request.status != RequestStatus.ACCEPTED:
                continue
            vehicle_index = int(dispatch[idx]) - 1
            if vehicle_index < 0 or vehicle_index >= len(self.vehicles):
                continue
            vehicle = self.vehicles[vehicle_index]
            if vehicle.status != VehicleStatus.IDLE:
                continue
            if vehicle.battery < self.config.min_battery_for_assignment:
                continue
            pickup_dist = self._distance(vehicle.node, request.pickup_node)
            trip_dist = self._distance(request.pickup_node, request.dropoff_node)
            ride_id = request.request_id
            ride = ActiveRide(
                ride_id=ride_id,
                vehicle_id=vehicle_index,
                pickup_node=request.pickup_node,
                dropoff_node=request.dropoff_node,
                price=request.price,
                pickup_distance_remaining=pickup_dist,
                trip_distance_remaining=trip_dist,
                total_distance=pickup_dist + trip_dist,
            )
            self.active_rides[ride_id] = ride
            vehicle.status = VehicleStatus.TO_PICKUP
            vehicle.target_node = request.pickup_node
            vehicle.ride_id = ride_id
            vehicle.remaining_distance = pickup_dist
            vehicle.travel_distance_total = pickup_dist
            vehicle.origin_node = vehicle.node
            request.status = RequestStatus.ASSIGNED
            self.pending_requests[idx] = None
            assigned += 1
        return assigned

    def _apply_repositioning(self, action: dict) -> int:
        reposition = np.asarray(action["reposition"], dtype=np.int32)
        moves = 0
        for idx, vehicle in enumerate(self.vehicles):
            target = int(reposition[idx])
            if vehicle.status != VehicleStatus.IDLE:
                continue
            if target == vehicle.node:
                continue
            distance = self._distance(vehicle.node, target)
            if distance <= 0.0:
                continue
            vehicle.status = VehicleStatus.REPOSITIONING
            vehicle.target_node = target
            vehicle.remaining_distance = distance
            vehicle.travel_distance_total = distance
            vehicle.origin_node = vehicle.node
            vehicle.ride_id = None
            moves += 1
        return moves

    def _apply_charging(self, action: dict) -> float:
        toggles = np.asarray(action["toggle_charging"], dtype=np.int32)
        reward = 0.0
        for idx, vehicle in enumerate(self.vehicles):
            wants_charge = bool(toggles[idx])
            at_station = vehicle.node in self.charging_nodes
            if vehicle.status == VehicleStatus.CHARGING:
                if not wants_charge and vehicle.battery >= 0.95:
                    vehicle.status = VehicleStatus.IDLE
                else:
                    self._charge_vehicle(vehicle)
                continue
            if wants_charge:
                if at_station and vehicle.status == VehicleStatus.IDLE:
                    vehicle.status = VehicleStatus.CHARGING
                    self._charge_vehicle(vehicle)
                elif vehicle.status == VehicleStatus.IDLE:
                    station = self._nearest_charging_station(vehicle.node)
                    vehicle.status = VehicleStatus.REPOSITIONING
                    vehicle.target_node = station
                    vehicle.remaining_distance = self._distance(vehicle.node, station)
                    vehicle.travel_distance_total = vehicle.remaining_distance
                    vehicle.origin_node = vehicle.node
        return reward

    def _advance_vehicle_tasks(self) -> float:
        total_reward = 0.0
        for idx, vehicle in enumerate(self.vehicles):
            if vehicle.status in (VehicleStatus.TO_PICKUP, VehicleStatus.WITH_PASSENGER):
                ride = self.active_rides.get(vehicle.ride_id or -1)
                if ride is None:
                    vehicle.status = VehicleStatus.IDLE
                    vehicle.ride_id = None
                    vehicle.remaining_distance = 0.0
                    continue
                total_reward += self._progress_ride(vehicle, ride)
            elif vehicle.status == VehicleStatus.REPOSITIONING:
                self._progress_reposition(vehicle)
            elif vehicle.status == VehicleStatus.CHARGING:
                self._charge_vehicle(vehicle)
        return total_reward

    def _progress_ride(self, vehicle: VehicleState, ride: ActiveRide) -> float:
        speed = self.config.speed_km_per_min
        distance_travelled = min(speed * self.config.minutes_per_step, vehicle.remaining_distance)
        vehicle.remaining_distance -= distance_travelled
        vehicle.battery = max(0.0, vehicle.battery - distance_travelled * self.config.battery_consumption_per_km)
        self.metrics["distance_travelled"] += distance_travelled
        self.metrics["energy_spent"] += distance_travelled * self.config.battery_consumption_per_km
        if vehicle.status == VehicleStatus.TO_PICKUP:
            ride.pickup_distance_remaining = max(0.0, ride.pickup_distance_remaining - distance_travelled)
            if ride.pickup_distance_remaining <= 1e-6:
                vehicle.status = VehicleStatus.WITH_PASSENGER
                vehicle.target_node = ride.dropoff_node
                vehicle.remaining_distance = ride.trip_distance_remaining
                vehicle.travel_distance_total = ride.trip_distance_remaining
                vehicle.node = ride.pickup_node
                vehicle.origin_node = ride.pickup_node
        elif vehicle.status == VehicleStatus.WITH_PASSENGER:
            ride.trip_distance_remaining = max(0.0, ride.trip_distance_remaining - distance_travelled)
            if ride.trip_distance_remaining <= 1e-6:
                vehicle.status = VehicleStatus.IDLE
                vehicle.node = ride.dropoff_node
                vehicle.ride_id = None
                vehicle.target_node = None
                vehicle.remaining_distance = 0.0
                vehicle.origin_node = vehicle.node
                vehicle.travel_distance_total = 0.0
                revenue = ride.price
                operating_cost = ride.total_distance * self.config.operating_cost_per_km
                step_reward = revenue - operating_cost
                self.metrics["completed_rides"] += 1.0
                self.metrics["earned_revenue"] += revenue
                del self.active_rides[ride.ride_id]
                return step_reward
        return 0.0

    def _progress_reposition(self, vehicle: VehicleState):
        speed = self.config.speed_km_per_min
        distance_travelled = min(speed * self.config.minutes_per_step, vehicle.remaining_distance)
        vehicle.remaining_distance -= distance_travelled
        vehicle.battery = max(0.0, vehicle.battery - distance_travelled * self.config.battery_consumption_per_km)
        self.metrics["distance_travelled"] += distance_travelled
        self.metrics["energy_spent"] += distance_travelled * self.config.battery_consumption_per_km
        if vehicle.remaining_distance <= 1e-6:
            vehicle.node = vehicle.target_node if vehicle.target_node is not None else vehicle.node
            vehicle.status = VehicleStatus.IDLE
            vehicle.target_node = None
            vehicle.remaining_distance = 0.0
            vehicle.origin_node = vehicle.node
            vehicle.travel_distance_total = 0.0

    def _charge_vehicle(self, vehicle: VehicleState):
        vehicle.battery = min(1.0, vehicle.battery + self.config.charge_rate_per_minute * self.config.minutes_per_step)
        if vehicle.battery >= 0.999:
            vehicle.status = VehicleStatus.IDLE

    def _advance_waiting_requests(self) -> int:
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
        rate = self.config.demand_rate if not initial else self.config.demand_rate * 0.5
        new_requests = int(self.np_random.poisson(rate))
        overflow = 0
        for _ in range(new_requests):
            slot = self._find_free_request_slot()
            if slot is None:
                overflow += 1
                self.metrics["overflow_requests"] += 1.0
                continue
            pickup = int(self.np_random.integers(self.num_nodes))
            dropoff = int(self.np_random.integers(self.num_nodes - 1))
            if dropoff >= pickup:
                dropoff += 1
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
        return overflow

    def _advance_clock(self):
        self.current_step += 1
        self.time_of_day += self.config.minutes_per_step / 60.0
        if self.time_of_day >= 24.0:
            self.time_of_day -= 24.0
            self.day_of_week = (self.day_of_week + 1) % 7

    def _request_acceptance_probability(self, request: RequestState) -> float:
        price_ratio = (request.price / max(request.base_price, 1e-6)) - 1.0
        z = (
            self.config.acceptance_base
            + self.config.acceptance_price_weight * price_ratio
            + self.config.acceptance_distance_weight * self._normalize_distance(request.distance)
            + self.config.acceptance_wait_weight * (request.wait_time / self.config.wait_time_normalizer)
            + self.config.acceptance_supply_demand_weight * (self._supply_demand_ratio() - self.config.surge_reference)
            + request.customer_bias
        )
        return float(1.0 / (1.0 + np.exp(-z)))

    def _find_free_request_slot(self) -> int | None:
        for idx, item in enumerate(self.pending_requests):
            if item is None:
                return idx
        return None
