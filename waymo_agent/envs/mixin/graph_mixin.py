from __future__ import annotations

import csv
import warnings
from typing import Any

import networkx as nx
import numpy as np

from .interface import GraphMixinInterface


class RideShareGraphMixin(GraphMixinInterface):
    """Graph and geometry helpers for RideShare environments."""

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

    def _build_graph_from_csv(self, map_name: str):
        node_path = self.map_dir / f"{map_name}_nodes.csv"
        edge_path = self.map_dir / f"{map_name}_edges.csv"
        if not node_path.exists() or not edge_path.exists():
            raise FileNotFoundError(f"Missing node or edge CSV for map '{map_name}'.")

        node_records: list[dict[str, Any]] = []
        coords: list[tuple[float, float]] = []
        node_ids: list[str] = []
        charging_nodes: list[int] = []
        graph = nx.DiGraph()

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
                graph.add_node(node_id, **node_meta)

        self.node_ids = node_ids
        self.node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
        self.node_coords = np.array(coords, dtype=np.float64)
        self.node_metadata = node_records
        self.charging_nodes = charging_nodes

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
                length_raw = row.get("length_km")
                if length_raw in (None, "", "nan"):
                    src_idx = self.node_index[src_id]
                    dst_idx = self.node_index[dst_id]
                    delta = self.node_coords[dst_idx] - self.node_coords[src_idx]
                    length_val = float(np.linalg.norm(delta))
                else:
                    length_val = float(length_raw)
                speed_raw = row.get("speed_limit_kmh")
                speed_val = (
                    float(speed_raw) / 60.0 if speed_raw not in (None, "", "nan") else self.config.speed_km_per_min
                )
                graph.add_edge(
                    src_id,
                    dst_id,
                    length=length_val,
                    speed=speed_val,
                    speed_limit_kmh=float(speed_raw) if speed_raw not in (None, "", "nan") else None,
                    original=row,
                )
                if self._parse_bool(row.get("is_two_way")):
                    graph.add_edge(
                        dst_id,
                        src_id,
                        length=length_val,
                        speed=speed_val,
                        speed_limit_kmh=float(speed_raw) if speed_raw not in (None, "", "nan") else None,
                        original=row,
                    )

        self.graph = graph
        self.map_name = map_name
        self._finalize_graph_structures()

    def _build_ring_graph(self, num_nodes: int):
        self.graph = nx.DiGraph()
        angles = np.linspace(0, 2 * np.pi, num=num_nodes, endpoint=False)
        radius = 5.0
        coords = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
        node_ids = [f"N{idx}" for idx in range(num_nodes)]
        node_records: list[dict[str, Any]] = []
        charging_indices = []
        charging_count = max(1, num_nodes // 4)
        for idx, node_id in enumerate(node_ids):
            is_charger = idx < charging_count
            node_meta = {
                "node_id": node_id,
                "x": float(coords[idx, 0]),
                "y": float(coords[idx, 1]),
                "is_charger": is_charger,
                "num_chargers": 2 if is_charger else 0,
                "zone": "synthetic",
            }
            node_records.append(node_meta)
            if is_charger:
                charging_indices.append(idx)
            self.graph.add_node(node_id, **node_meta)
        for idx, node_id in enumerate(node_ids):
            next_idx = (idx + 1) % num_nodes
            length_val = float(np.linalg.norm(coords[next_idx] - coords[idx]))
            next_id = node_ids[next_idx]
            self.graph.add_edge(node_id, next_id, length=length_val, speed=self.config.speed_km_per_min)
            self.graph.add_edge(next_id, node_id, length=length_val, speed=self.config.speed_km_per_min)
        self.node_ids = node_ids
        self.node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
        self.node_coords = coords.astype(np.float64)
        self.node_metadata = node_records
        self.charging_nodes = charging_indices
        self.map_name = self.map_name or "synthetic_ring"
        self._finalize_graph_structures()

    @staticmethod
    def _parse_bool(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}

    def _finalize_graph_structures(self):
        if not self.node_ids:
            self.graph_edges = []
            self.distance_matrix = None
            return
        self.graph_edges = [
            (self.node_index[src], self.node_index[dst])
            for src, dst in self.graph.edges
            if src in self.node_index and dst in self.node_index
        ]
        coords = self.node_coords
        if coords.size:
            x_span = float(np.max(coords[:, 0]) - np.min(coords[:, 0]))
            y_span = float(np.max(coords[:, 1]) - np.min(coords[:, 1]))
            self.max_distance = float(np.hypot(x_span, y_span))
        else:
            self.max_distance = 0.0
        self.distance_matrix = None

    def _distance(self, src: int, dst: int) -> float:
        if src == dst:
            return 0.0
        if self.node_coords.size == 0:
            return 0.0
        diff = self.node_coords[src] - self.node_coords[dst]
        return float(np.linalg.norm(diff))

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
