from __future__ import annotations

import typing as t

import networkx as nx
import numpy as np
from scipy.spatial.kdtree import cKDTree

from waymo_agent.osmnx.euclidean_L2_embed import nearest_node_id_from_xy
from waymo_agent.osmnx.shortest_path import calculate_longest_path

from ...osmnx.charging import get_charging_nodes
from ...osmnx.load_graph_safe import load_graph_type_preserved, post_load
from .interface import GraphMixinInterface


class OSMnxWrapperMixin(GraphMixinInterface):
    _chargers: dict[str, int]
    _longest_distance: float
    _longest_route: list[int]
    _num_nodes: int
    _num_edges: int
    G: nx.MultiDiGraph
    pos_raw: dict[int, tuple[float, float]]
    pos_norm: dict[int, tuple[float, float]]

    """
    Mixin for handling graph operations using NetworkX and OSMnx.

    This mixin manages:
    - Loading the graph from a file.
    - Providing access to graph properties (nodes, edges, chargers).
    - Calculating distances between nodes.
    - Finding nearest points of interest (e.g., charging stations).
    """

    def build_graph(self):
        """
        Build the graph from the map file specified in the configuration.
        """
        map_candidate = self.map_name or self.config.map_name

        self._load_graphml(map_candidate)

    # ------------------------------------------------------------------ #
    # Graph construction helpers
    # ------------------------------------------------------------------ #
    def _load_graphml(self, name: str):
        map_path = self.config.map_dir / name
        G = load_graph_type_preserved(map_path)
        post_load(G)
        self.graph = G
        self.G = G
        self._chargers = get_charging_nodes(G)
        self.map_name = map_path.name

        self.node_ids = list(G.nodes)  # index -> node id
        self.node_index = {nid: i for i, nid in enumerate(self.node_ids)}  # node id -> index

        xs = np.array([G.nodes[n]["x_norm"] for n in G.nodes])
        ys = np.array([G.nodes[n]["y_norm"] for n in G.nodes])
        self.node_coords = np.stack([xs, ys], axis=1)  # shape: (N, 2)
        self.kd_tree = cKDTree(self.node_coords)  # type: ignore
        self.pos_raw = {n: (self.graph.nodes[n]["x"], self.graph.nodes[n]["y"]) for n in self.graph.nodes}
        self.pos_norm = {n: (self.graph.nodes[n]["x_norm"], self.graph.nodes[n]["y_norm"]) for n in self.graph.nodes}

    # ------------------------------------------------------------------ #
    # Public helpers expected by other mixins
    # ------------------------------------------------------------------ #
    @property
    def size(self) -> tuple[int, int]:
        return (self.num_nodes, self.num_edges)

    @property
    def num_nodes(self) -> int:
        try:
            return self._num_nodes
        except AttributeError:
            self._num_nodes = self.graph.number_of_nodes()
        return self._num_nodes

    @property
    def num_edges(self) -> int:
        try:
            return self._num_edges
        except AttributeError:
            self._num_edges = self.graph.number_of_edges()
        return self._num_edges

    @property
    def charging_nodes(self):
        return self._chargers

    @property
    def num_charging_stations(self) -> int:
        return len(self._chargers)

    @property
    def longest_distance(self) -> float:
        try:
            return self._longest_distance
        except AttributeError:
            calculate_longest_path(self.graph, weight="length")
            max_length, longest_path = calculate_longest_path(self.graph, weight="length")

            self._longest_distance, self._longest_route = max_length, longest_path
            return self._longest_distance

    def distance(self, src_id: int, dst_id: int, metric: t.Literal["length", "travel_time_minutes"] = "length"):
        """
        Compute the length or travel time between two nodes in the graph.

        Return the shortest distance / fastest travel time
        """
        src_node = self.node_ids[src_id]
        dst_node = self.node_ids[dst_id]

        try:
            length = nx.shortest_path_length(self.graph, src_node, dst_node, weight=metric)
            return float(length)
        except nx.NetworkXNoPath:
            return float("inf")

    def nearest_node_id(self, x_norm: float, y_norm: float) -> int:
        return nearest_node_id_from_xy(self.kd_tree, self.node_ids, x_norm, y_norm)
