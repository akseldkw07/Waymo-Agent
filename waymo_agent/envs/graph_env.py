from __future__ import annotations

import csv
import warnings
from pathlib import Path
from typing import Any
import typing as t

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .dataclasses import ActiveRide, EnvConfig, RequestState, RequestStatus, VehicleState, VehicleStatus


class RideShareEnv(gym.Env):
    """Gymnasium environment that mirrors the proposal architecture for the Waymo RL project."""

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 6}
    WEATHER_STATES: tuple[str, ...] = ("clear", "rain", "snow")
    render_mode: str | None
    config: EnvConfig
    vehicles: list[VehicleState]
    pending_requests: list[RequestState | None]
    active_rides: dict[int, ActiveRide]
    observation_space: spaces.Space
    action_space: spaces.Space

    def __init__(self, config: EnvConfig | None = None, render_mode: t.Literal["human", "ansi"] = "human"):
        self.config = config or EnvConfig()
        self.render_mode = render_mode
        self.map_dir = Path(self.config.map_dir).expanduser()
        self.map_name = self.config.map_name
        self.node_ids: list[str] = []
        self.node_index: dict[str, int] = {}
        self.node_metadata: list[dict[str, Any]] = []
        self.node_coords: np.ndarray = np.zeros((0, 2), dtype=np.float64)
        self.distance_matrix: np.ndarray = np.zeros((0, 0), dtype=np.float64)
        self.speed_matrix: np.ndarray = np.zeros((0, 0), dtype=np.float64)
        self.graph_edges: list[tuple[int, int]] = []
        self.charging_nodes: list[int] = []
        self._build_graph()
        self.num_nodes = int(self.node_coords.shape[0])
        if self.num_nodes == 0:
            raise ValueError("Map must contain at least one node.")
        self.config.num_nodes = self.num_nodes
        self._build_observation_space()
        self._build_action_space()
        self.np_random: np.random.Generator = np.random.default_rng()
        self.vehicles: list[VehicleState] = []
        self.pending_requests: list[RequestState | None] = []
        self.active_rides: dict[int, ActiveRide] = {}
        self.next_request_id: int = 0
        self.current_step: int = 0
        self.day_of_week: int = 0
        self.time_of_day: float = 0.0
        self.weather_idx: int = 0
        self.metrics: dict[str, float] = {}
        self._mpl = None
        self._figure = None
        self._axes = None
        self._line_cls = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.current_step = 0
        self.day_of_week = int(self.np_random.integers(7))
        self.time_of_day = float(self.np_random.uniform(6.0, 9.0))
        self.weather_idx = int(self.np_random.integers(len(self.WEATHER_STATES)))
        self.metrics = {
            "completed_rides": 0.0,
            "rejected_requests": 0.0,
            "cancelled_requests": 0.0,
            "overflow_requests": 0.0,
            "earned_revenue": 0.0,
            "energy_spent": 0.0,
            "distance_travelled": 0.0,
        }
        self._initialize_vehicles()
        self.pending_requests = [None] * self.config.max_pending_requests
        self.active_rides.clear()
        self.next_request_id = 0
        self._spawn_requests(initial=True)
        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def step(self, action: dict):
        self._validate_action(action)
        reward = 0.0
        rejected = self._apply_pricing_decisions(action)
        reward += rejected * self.config.penalty_rejected
        assigned = self._apply_dispatch_decisions(action)
        reward += assigned * 0.0
        repositioned = self._apply_repositioning(action)
        reward += repositioned * 0.0
        reward += self._apply_charging(action)
        ride_reward = self._advance_vehicle_tasks()
        reward += ride_reward
        cancellations = self._advance_waiting_requests()
        reward += cancellations * self.config.penalty_cancelled
        overflow = self._spawn_requests()
        reward += overflow * self.config.penalty_overflow
        self._advance_clock()
        terminated = self.current_step >= self.config.max_episode_steps
        truncated = False
        observation = self._get_observation()
        info = self._get_info()
        return observation, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        mode = self.render_mode or "human"
        if mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render mode: {mode}")
        if mode == "human":
            return self._render_matplotlib()
        if mode == "ansi":
            return self._render_text()
        return None

    def close(self):
        if self._figure is not None and self._mpl is not None:
            self._mpl.close(self._figure)
        self._figure = None
        self._axes = None
        self._mpl = None
        self._line_cls = None

    # --- environment setup -------------------------------------------------

    def _build_graph(self):
        built = False
        map_candidate = self.map_name
        if map_candidate:
            try:
                self._build_graph_from_csv(map_candidate)
                built = True
            except FileNotFoundError:
                warnings.warn(
                    f"Map '{map_candidate}' not found in '{self.map_dir}'. Falling back to synthetic ring graph.",
                    stacklevel=2,
                )
                self.map_name = None
            except ValueError as exc:
                warnings.warn(
                    f"Failed to load map '{map_candidate}': {exc}. Falling back to synthetic ring graph.",
                    stacklevel=2,
                )
                self.map_name = None
        if not built:
            self._build_ring_graph(max(self.config.num_nodes, 3))
        finite_mask = np.isfinite(self.distance_matrix)
        self.max_distance = float(np.max(self.distance_matrix[finite_mask])) if finite_mask.any() else 0.0

    def _build_graph_from_csv(self, map_name: str):
        node_path = self.map_dir / f"{map_name}_nodes.csv"
        edge_path = self.map_dir / f"{map_name}_edges.csv"
        if not node_path.exists() or not edge_path.exists():
            raise FileNotFoundError(f"Missing node or edge CSV for map '{map_name}'.")

        node_records: list[dict[str, Any]] = []
        coords: list[tuple[float, float]] = []
        node_ids: list[str] = []
        charging_nodes: list[int] = []

        with node_path.open("r", newline="") as node_file:
            reader = csv.DictReader(node_file)
            if reader.fieldnames is None or "node_id" not in reader.fieldnames:
                raise ValueError("nodes.csv must contain a 'node_id' column.")
            for idx, row in enumerate(reader):
                node_id = row.get("node_id", "").strip()
                if not node_id:
                    raise ValueError(f"Row {idx} in nodes CSV is missing 'node_id'.")
                try:
                    x = float(row.get("x", "0.0"))
                    y = float(row.get("y", "0.0"))
                except ValueError as exc:
                    raise ValueError(f"Invalid coordinate for node '{node_id}': {exc}") from exc
                is_charger = self._parse_bool(row.get("is_charger"))
                num_chargers = int(row.get("num_chargers", 0) or 0)
                node_meta: dict[str, Any] = dict(row)
                node_meta.update(
                    {
                        "node_id": node_id,
                        "x": x,
                        "y": y,
                        "is_charger": is_charger,
                        "num_chargers": num_chargers,
                    }
                )
                node_records.append(node_meta)
                node_ids.append(node_id)
                coords.append((x, y))
                if is_charger:
                    charging_nodes.append(idx)

        self.node_ids = node_ids
        self.node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
        self.node_coords = np.array(coords, dtype=np.float64)
        self.node_metadata = node_records
        self.charging_nodes = charging_nodes

        node_count = len(node_ids)
        adjacency = np.full((node_count, node_count), np.inf, dtype=np.float64)
        np.fill_diagonal(adjacency, 0.0)
        speed_matrix = np.full((node_count, node_count), self.config.speed_km_per_min, dtype=np.float64)
        graph_edges: list[tuple[int, int]] = []

        with edge_path.open("r", newline="") as edge_file:
            reader = csv.DictReader(edge_file)
            required = {"edge_id", "src", "dst"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"edges.csv must contain columns: {', '.join(sorted(required))}.")
            for row in reader:
                src_id = row.get("src", "").strip()
                dst_id = row.get("dst", "").strip()
                if src_id not in self.node_index or dst_id not in self.node_index:
                    raise ValueError(f"Edge references unknown nodes: {src_id} -> {dst_id}")
                src_idx = self.node_index[src_id]
                dst_idx = self.node_index[dst_id]
                length_raw = row.get("length_km")
                if length_raw in (None, "", "nan"):
                    delta = self.node_coords[dst_idx] - self.node_coords[src_idx]
                    length_val = float(np.linalg.norm(delta))
                else:
                    length_val = float(length_raw)
                speed_raw = row.get("speed_limit_kmh")
                speed_val = (
                    float(speed_raw) / 60.0 if speed_raw not in (None, "", "nan") else self.config.speed_km_per_min
                )
                adjacency[src_idx, dst_idx] = min(adjacency[src_idx, dst_idx], length_val)
                speed_matrix[src_idx, dst_idx] = speed_val
                if (src_idx, dst_idx) not in graph_edges:
                    graph_edges.append((src_idx, dst_idx))
                if self._parse_bool(row.get("is_two_way")):
                    adjacency[dst_idx, src_idx] = min(adjacency[dst_idx, src_idx], length_val)
                    speed_matrix[dst_idx, src_idx] = speed_val
                    if (dst_idx, src_idx) not in graph_edges:
                        graph_edges.append((dst_idx, src_idx))

        self.graph_edges = graph_edges
        self.distance_matrix = self._floyd_warshall(adjacency)
        self.speed_matrix = speed_matrix
        self.map_name = map_name

    def _build_ring_graph(self, num_nodes: int):
        angles = np.linspace(0, 2 * np.pi, num=num_nodes, endpoint=False)
        radius = 5.0
        coords = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
        diffs = coords[:, None, :] - coords[None, :, :]
        distances = np.linalg.norm(diffs, axis=2)
        self.node_coords = coords.astype(np.float64)
        self.distance_matrix = distances.astype(np.float64)
        self.speed_matrix = np.full_like(distances, self.config.speed_km_per_min, dtype=np.float64)
        self.node_ids = [f"N{idx}" for idx in range(num_nodes)]
        self.node_index = {node_id: idx for idx, node_id in enumerate(self.node_ids)}
        charging_count = max(1, num_nodes // 4)
        self.charging_nodes = list(range(charging_count))
        self.node_metadata = []
        for idx, node_id in enumerate(self.node_ids):
            self.node_metadata.append(
                {
                    "node_id": node_id,
                    "x": float(coords[idx, 0]),
                    "y": float(coords[idx, 1]),
                    "is_charger": idx in self.charging_nodes,
                    "num_chargers": 2 if idx in self.charging_nodes else 0,
                }
            )
        self.graph_edges = [(idx, (idx + 1) % num_nodes) for idx in range(num_nodes)]
        self.map_name = self.map_name or "synthetic_ring"

    @staticmethod
    def _floyd_warshall(adjacency: np.ndarray) -> np.ndarray:
        dist = adjacency.copy()
        n = dist.shape[0]
        for k in range(n):
            dist = np.minimum(dist, dist[:, [k]] + dist[[k], :])
        return dist

    @staticmethod
    def _parse_bool(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}

    def _build_observation_space(self):
        num_v = self.config.num_vehicles
        num_r = self.config.max_pending_requests
        self.observation_space = spaces.Dict(
            {
                "global": spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
                "supply_demand_ratio": spaces.Box(low=0.0, high=10.0, shape=(1,), dtype=np.float32),
                "vehicles": spaces.Box(low=0.0, high=1.0, shape=(num_v, 6), dtype=np.float32),
                "pending_requests": spaces.Box(low=0.0, high=1.0, shape=(num_r, 6), dtype=np.float32),
                "dispatch_mask": spaces.Box(low=0.0, high=1.0, shape=(num_r, num_v + 1), dtype=np.float32),
                "pricing_mask": spaces.Box(low=0.0, high=1.0, shape=(num_r,), dtype=np.float32),
            }
        )

    def _build_action_space(self):
        num_v = self.config.num_vehicles
        num_r = self.config.max_pending_requests
        self.action_space = spaces.Dict(
            {
                "price_adjustments": spaces.Box(low=-1.0, high=1.0, shape=(num_r,), dtype=np.float32),
                "dispatch": spaces.MultiDiscrete(np.full(num_r, num_v + 1, dtype=np.int32)),
                "reposition": spaces.MultiDiscrete(np.full(num_v, self.num_nodes, dtype=np.int32)),
                "toggle_charging": spaces.MultiBinary(num_v),
            }
        )

    def _initialize_vehicles(self):
        self.vehicles = []
        for _ in range(self.config.num_vehicles):
            node = int(self.np_random.integers(self.num_nodes))
            battery = float(self.np_random.uniform(0.6, 1.0))
            self.vehicles.append(VehicleState(node=node, battery=battery, origin_node=node))

    # --- action handling ----------------------------------------------------

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

    # --- dynamics -----------------------------------------------------------

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

    # --- helpers ------------------------------------------------------------

    def _get_observation(self):
        global_features = np.array(
            [
                self.time_of_day / 24.0,
                self.day_of_week / 6.0,
                self.weather_idx / max(1, len(self.WEATHER_STATES) - 1),
            ],
            dtype=np.float32,
        )
        supply_demand = np.array([self._supply_demand_ratio()], dtype=np.float32)
        vehicle_obs = np.zeros((self.config.num_vehicles, 6), dtype=np.float32)
        for idx, vehicle in enumerate(self.vehicles):
            vehicle_obs[idx, 0] = self._normalize_node(vehicle.node)
            vehicle_obs[idx, 1] = np.float32(np.clip(vehicle.battery, 0.0, 1.0))
            vehicle_obs[idx, 2] = 1.0 if vehicle.status == VehicleStatus.IDLE else 0.0
            vehicle_obs[idx, 3] = 1.0 if vehicle.status == VehicleStatus.TO_PICKUP else 0.0
            vehicle_obs[idx, 4] = 1.0 if vehicle.status == VehicleStatus.WITH_PASSENGER else 0.0
            vehicle_obs[idx, 5] = np.float32(self._normalize_distance(vehicle.remaining_distance))
        request_obs = np.zeros((self.config.max_pending_requests, 6), dtype=np.float32)
        dispatch_mask = np.zeros((self.config.max_pending_requests, self.config.num_vehicles + 1), dtype=np.float32)
        pricing_mask = np.zeros((self.config.max_pending_requests,), dtype=np.float32)
        for idx, request in enumerate(self.pending_requests):
            if request is None:
                dispatch_mask[idx, 0] = 1.0
                continue
            request_obs[idx, 0] = self._normalize_node(request.pickup_node)
            request_obs[idx, 1] = self._normalize_node(request.dropoff_node)
            request_obs[idx, 2] = np.float32(np.clip(request.wait_time / self.config.wait_time_normalizer, 0.0, 1.0))
            request_obs[idx, 3] = np.float32(self._normalize_distance(request.distance))
            request_obs[idx, 4] = np.float32(
                np.clip(request.price / max(request.base_price, 1e-6), 0.0, self.config.max_price_multiplier)
            )
            request_obs[idx, 5] = 1.0 if request.status == RequestStatus.ACCEPTED else 0.0
            dispatch_mask[idx, 0] = 1.0
            if request.status == RequestStatus.AWAITING_PRICE:
                pricing_mask[idx] = 1.0
            if request.status == RequestStatus.ACCEPTED:
                for v_idx, vehicle in enumerate(self.vehicles):
                    feasible = (
                        vehicle.status == VehicleStatus.IDLE
                        and vehicle.battery >= self.config.min_battery_for_assignment
                    )
                    dispatch_mask[idx, v_idx + 1] = 1.0 if feasible else 0.0
        observation = {
            "global": global_features,
            "supply_demand_ratio": supply_demand,
            "vehicles": vehicle_obs,
            "pending_requests": request_obs,
            "dispatch_mask": dispatch_mask,
            "pricing_mask": pricing_mask,
        }
        return observation

    def _get_info(self):
        info = {
            "step": self.current_step,
            "metrics": dict(self.metrics),
            "active_rides": len(self.active_rides),
            "pending_requests": sum(1 for r in self.pending_requests if r is not None),
            "map_name": self.map_name,
            "node_ids": list(self.node_ids),
        }
        return info

    def _supply_demand_ratio(self) -> float:
        idle = sum(1 for v in self.vehicles if v.status == VehicleStatus.IDLE)
        outstanding = sum(1 for r in self.pending_requests if r is not None and r.status != RequestStatus.CANCELLED)
        outstanding += len(self.active_rides)
        outstanding = max(outstanding, 1)
        ratio = idle / outstanding
        normalized = ratio / self.config.supply_demand_normalizer
        return float(np.clip(normalized, 0.0, 10.0))

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

    def _distance(self, src: int, dst: int) -> float:
        if src == dst:
            return 0.0
        distance = float(self.distance_matrix[src, dst])
        if not np.isfinite(distance):
            return float(self.max_distance if self.max_distance > 0 else 0.0)
        return distance

    def _normalize_node(self, node: int) -> float:
        if self.num_nodes <= 1:
            return 0.0
        return float(node) / float(self.num_nodes - 1)

    def _normalize_distance(self, value: float) -> float:
        if self.max_distance <= 1e-6:
            return 0.0
        return float(np.clip(value / self.max_distance, 0.0, 1.0))

    def _nearest_charging_station(self, node: int) -> int:
        best_node = self.charging_nodes[0]
        best_distance = self._distance(node, best_node)
        for station in self.charging_nodes[1:]:
            dist = self._distance(node, station)
            if dist < best_distance:
                best_distance = dist
                best_node = station
        return best_node

    # --- rendering ----------------------------------------------------------

    def _render_matplotlib(self):
        try:
            if self._mpl is None:
                import matplotlib.pyplot as plt
                from matplotlib.lines import Line2D

                plt.ion()
                self._mpl = plt
                self._line_cls = Line2D
            elif self._line_cls is None:
                from matplotlib.lines import Line2D

                self._line_cls = Line2D
            if self._figure is None or self._axes is None or not self._mpl.fignum_exists(self._figure.number):
                self._figure, self._axes = self._mpl.subplots(figsize=(10, 6))
        except ImportError as exc:
            raise ImportError("Matplotlib is required for human render mode.") from exc

        ax = self._axes
        ax.clear()
        node_x = self.node_coords[:, 0]
        node_y = self.node_coords[:, 1]

        for src, dst in self.graph_edges:
            ax.plot(
                [node_x[src], node_x[dst]],
                [node_y[src], node_y[dst]],
                color="#dddddd",
                linewidth=1.0,
                zorder=0,
            )

        ax.scatter(node_x, node_y, c="#444444", s=50, zorder=2)
        if self.charging_nodes:
            ax.scatter(
                node_x[self.charging_nodes],
                node_y[self.charging_nodes],
                c="#2ca02c",
                s=80,
                marker="s",
                zorder=3,
            )

        status_colors: dict[VehicleStatus, str] = {
            VehicleStatus.IDLE: "#1f77b4",
            VehicleStatus.TO_PICKUP: "#ff7f0e",
            VehicleStatus.WITH_PASSENGER: "#d62728",
            VehicleStatus.REPOSITIONING: "#9467bd",
            VehicleStatus.CHARGING: "#17becf",
        }

        for idx, vehicle in enumerate(self.vehicles):
            color = status_colors.get(vehicle.status, "#7f7f7f")
            pos = self._vehicle_position(vehicle)
            ax.scatter(pos[0], pos[1], c=color, s=70, edgecolors="black", linewidths=0.6, zorder=4)
            ax.text(
                pos[0],
                pos[1] + 0.2,
                f"V{idx} {vehicle.battery:.0%}",
                fontsize=7,
                ha="center",
                va="bottom",
                zorder=5,
            )

        for request in self.pending_requests:
            if request is None:
                continue
            if request.status == RequestStatus.CANCELLED:
                continue
            pickup = self.node_coords[request.pickup_node]
            dropoff = self.node_coords[request.dropoff_node]
            if request.status == RequestStatus.AWAITING_PRICE:
                color = "#ffbb78"
            elif request.status == RequestStatus.ACCEPTED:
                color = "#2ca02c"
            elif request.status == RequestStatus.ASSIGNED:
                color = "#1f77b4"
            else:
                color = "#c7c7c7"
            ax.scatter(pickup[0], pickup[1], c=color, s=60, marker="^", zorder=6)
            ax.plot(
                [pickup[0], dropoff[0]],
                [pickup[1], dropoff[1]],
                color=color,
                linewidth=1.2,
                alpha=0.5,
                zorder=1,
            )
            ax.text(
                pickup[0],
                pickup[1] - 0.25,
                f"${request.price:.1f} / {request.wait_time:.0f}m",
                fontsize=6,
                ha="center",
                va="top",
                zorder=7,
            )

        for ride in self.active_rides.values():
            pickup = self.node_coords[ride.pickup_node]
            dropoff = self.node_coords[ride.dropoff_node]
            ax.plot(
                [pickup[0], dropoff[0]],
                [pickup[1], dropoff[1]],
                color="#d62728",
                linewidth=2.0,
                alpha=0.7,
                zorder=1,
            )

        padding = 1.5
        ax.set_xlim(node_x.min() - padding, node_x.max() + padding)
        ax.set_ylim(node_y.min() - padding, node_y.max() + padding)
        ax.set_aspect("equal", adjustable="datalim")
        ax.axis("off")

        legend_handles = [
            self._line_cls(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#444444",
                markeredgecolor="black",
                markersize=6,
                label="Node",
            ),
            self._line_cls(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="#2ca02c",
                markeredgecolor="black",
                markersize=6,
                label="Charging",
            ),
            self._line_cls(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor="#ffbb78",
                markeredgecolor="black",
                markersize=6,
                label="Request",
            ),
            self._line_cls([0], [0], color="#d62728", linewidth=2.0, label="Active Ride"),
        ]
        for status, color in status_colors.items():
            legend_handles.append(
                self._line_cls(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markersize=6,
                    label=status.name.replace("_", " ").title(),
                )
            )
        ax.legend(handles=legend_handles, loc="upper right", fontsize=7, frameon=False)

        metrics = self.metrics
        ax.text(
            ax.get_xlim()[0] + 0.1,
            ax.get_ylim()[1] - 0.1,
            f"Completed: {metrics.get('completed_rides', 0):.0f}\n"
            f"Rejected: {metrics.get('rejected_requests', 0):.0f}\n"
            f"Cancelled: {metrics.get('cancelled_requests', 0):.0f}\n"
            f"Revenue: ${metrics.get('earned_revenue', 0):.1f}",
            fontsize=8,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 0.4},
        )

        ax.set_title(
            f"Step {self.current_step} | t={self.time_of_day:.2f}h | Weather={self.WEATHER_STATES[self.weather_idx]}"
        )
        self._figure.canvas.draw()
        self._figure.canvas.flush_events()
        self._mpl.pause(0.001)
        return None

    def _render_text(self) -> str:
        lines = [
            f"Step {self.current_step} | time={self.time_of_day:.2f}h | day={self.day_of_week} | weather={self.WEATHER_STATES[self.weather_idx]}",
            f"Pending: {sum(1 for r in self.pending_requests if r is not None)} | Active rides: {len(self.active_rides)}",
            f"Metrics: {self.metrics}",
        ]
        for idx, vehicle in enumerate(self.vehicles):
            lines.append(
                f"V{idx} node={vehicle.node} battery={vehicle.battery:.2f} "
                f"status={vehicle.status.name} target={vehicle.target_node} remaining={vehicle.remaining_distance:.2f}"
            )
        return "\n".join(lines)

    def _vehicle_position(self, vehicle: VehicleState) -> np.ndarray:
        if vehicle.target_node is not None and vehicle.origin_node is not None and vehicle.travel_distance_total > 1e-6:
            start = self.node_coords[vehicle.origin_node]
            end = self.node_coords[vehicle.target_node]
            progress = 1.0 - (vehicle.remaining_distance / max(vehicle.travel_distance_total, 1e-6))
            progress = float(np.clip(progress, 0.0, 1.0))
            return start + progress * (end - start)
        return self.node_coords[vehicle.node]
