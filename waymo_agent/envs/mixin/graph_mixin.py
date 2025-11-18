from __future__ import annotations

import warnings
from typing import Any

import numpy as np

import osmnx as ox

from waymo_agent.osmnx.load_utils import post_load
from .interface import GraphMixinInterface
from ...constants import MAP_DIR


class OSMnxWrapperMixin(GraphMixinInterface):
    """Graph and geometry helpers for RideShare environments."""

    def _build_graph(self):
        if self.map_name:
            graph = ox.load_graphml(MAP_DIR / self.map_name)
            post_load(graph)
            self.graph = graph

    @staticmethod
    def _parse_bool(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}

    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def charging_nodes(self) -> list:
        return [node for node, data in self.graph.nodes(data=True) if data.get("num_chargers", False)]

    def _distance(self, src: int, dst: int) -> float:
        """Get the distance in meters between two nodes."""
        try:
            length = self.graph.edges[src, dst, 0].get("length", 0.0)
            return float(length)
        except KeyError:
            warnings.warn(f"Edge from {src} to {dst} not found in graph; returning inf distance.")
            return float("inf")

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
