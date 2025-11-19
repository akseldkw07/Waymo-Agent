from __future__ import annotations

import typing as t

import networkx as nx
import osmnx as ox

from ...osmnx.charging import get_charging_nodes
from ...osmnx.load_utils import post_load
from .interface import GraphMixinInterface


class OSMnxWrapperMixin(GraphMixinInterface):
    _chargers: dict[str, int]
    """Graph helpers backed by NetworkX + OSMnx data."""

    def _build_graph(self):
        map_candidate = self.map_name or self.config.map_name

        self._load_graphml(map_candidate)

    # ------------------------------------------------------------------ #
    # Graph construction helpers
    # ------------------------------------------------------------------ #
    def _load_graphml(self, name: str):
        map_path = self.config.map_dir / name
        graph = ox.load_graphml(map_path)
        post_load(graph)
        self.graph = graph
        self._chargers = get_charging_nodes(graph)
        self.map_name = map_path.name

    # ------------------------------------------------------------------ #
    # Public helpers expected by other mixins
    # ------------------------------------------------------------------ #
    @property
    def size(self) -> tuple[int, int]:
        return (self.num_nodes, len(self.graph_edges))

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def charging_nodes(self):
        return self._chargers

    def distance(self, src_id: int, dst_id: int, metric: t.Literal["length", "travel_time_min"] = "length"):
        """
        Compute the length or travel time between two nodes in the graph.

        Return the shortest distance / fastest travel time
        """
        src_node = self.node_ids[src_id]
        dst_node = self.node_ids[dst_id]

        length = nx.shortest_path_length(self.graph, src_node, dst_node, weight=metric)
        return length

    def nearest_charging_station(self, node_id: int) -> int:
        """
        Find the nearest charging station to the given node / coordinates.
        """
        ...
