from __future__ import annotations

import typing as t
from functools import cached_property

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree  # type: ignore

from waymo_agent.graph_env.df_utils import edges_to_df, nodes_to_df
from waymo_agent.osmnx.distances import calculate_longest_path
from waymo_agent.osmnx.euclidean_L2_embed import _recover_L2_params_env, nearest_node_id_from_xy
from waymo_agent.osmnx.post_process_enrichment import assign_lambda_values

from ...osmnx.charging import get_charging_nodes
from ...osmnx.load_graph_safe import load_graph_type_preserved, post_load
from .interface import GymEnvInterface


class OSMnxWrapperMixin(GymEnvInterface):
    _chargers: dict[str, int]
    _longest_distance: float
    _longest_route: list[int]
    _num_nodes: int
    _num_edges: int
    _all_pairs_shortest_path: dict[int, dict[int, list[int]]]
    _all_pairs_shortest_length: dict[int, dict[int, float]]
    G: nx.MultiDiGraph
    pos_raw: dict[int, tuple[float, float]]
    pos_norm: dict[int, tuple[float, float]]
    kd_tree: cKDTree
    route_edges_df: dict[tuple[int, int], pd.DataFrame]

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
        self._graph_auxiliary()
        self._graph_to_df()

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

        assign_lambda_values(G, self.config)

    def _graph_auxiliary(self):
        G = self.graph
        self.node_ids = list(G.nodes)  # index -> node id
        self.node_index = {nid: i for i, nid in enumerate(self.node_ids)}  # node id -> index

        xs = np.array([G.nodes[n]["x_norm"] for n in G.nodes])
        ys = np.array([G.nodes[n]["y_norm"] for n in G.nodes])
        self.node_coords = np.stack([xs, ys], axis=1)  # shape: (N, 2)

        self.kd_tree = cKDTree(self.node_coords)  # type: ignore
        self.pos_raw = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes}
        self.pos_norm = {n: (G.nodes[n]["x_norm"], G.nodes[n]["y_norm"]) for n in G.nodes}

    def _graph_to_df(self):
        G = self.graph
        self.node_df = nodes_to_df(G)
        # TODO reduce # edges by transitioning to undirected graph (this needs to happen far upstream)
        self.edge_df = edges_to_df(G)
        self.route_edges_df = {}

        # Recover L2 normalization parameters for coordinate conversion
        self.l2_recovery = _recover_L2_params_env(self)
        degree_centrality = nx.closeness_centrality(G)
        self.node_df["degree_centrality"] = self.node_df["node_id"].map(degree_centrality)

    @cached_property
    def EdgeDFEnriched(self):
        cols_node = ["node_id", "x", "y", "x_norm", "y_norm"]
        merged_src = pd.merge(
            self.edge_df,
            self.node_df[cols_node],
            left_on="source",
            right_on="node_id",
            how="left",
            validate="m:1",
            suffixes=("", "_src"),
        )
        for col in cols_node:
            if col not in self.edge_df.columns:
                merged_src = merged_src.rename(columns={col: f"{col}_src"})
        merged = pd.merge(
            merged_src,
            self.node_df[cols_node],
            left_on="target",
            right_on="node_id",
            how="left",
            validate="m:1",
            suffixes=("", "_tgt"),
        )
        for col in cols_node:
            if col not in self.edge_df.columns:
                merged = merged.rename(columns={col: f"{col}_tgt"})
        merged.sort_values(by=["source", "target", "travel_time_minutes"], ascending=[True, True, True], inplace=True)
        merged.drop_duplicates(subset=["source", "target"], inplace=True)
        return merged

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
    def pairwise_paths(self):
        try:
            return self._all_pairs_shortest_path
        except AttributeError:
            weight = "length"
            self._all_pairs_shortest_path = dict(nx.all_pairs_dijkstra_path(self.graph, weight=weight))
            return self._all_pairs_shortest_path

    @property
    def pairwise_distances(self):
        try:
            return self._all_pairs_shortest_length
        except AttributeError:
            weight = "length"
            self._all_pairs_shortest_length = dict(nx.all_pairs_dijkstra_path_length(self.graph, weight=weight))
            return self._all_pairs_shortest_length

    @property
    def longest_distance(self) -> float:
        try:
            return self._longest_distance
        except AttributeError:
            calculate_longest_path(self.graph, weight="length")
            max_length, longest_path = calculate_longest_path(self.graph, weight="length")

            self._longest_distance, self._longest_route = max_length, longest_path
            return self._longest_distance

    def nearest_node_id(self, x_norm: t.Sequence[float], y_norm: t.Sequence[float]):
        nodes = []
        for x, y in zip(x_norm, y_norm):
            nodeid = nearest_node_id_from_xy(self.kd_tree, self.node_ids, x, y)
            nodes.append(nodeid)
        return nodes

    def calc_shortest_paths(self, source_nodes: t.Sequence[int], target_nodes: t.Sequence[int], weight: str = "length"):
        """
        Calculate shortest paths and distances between source and target nodes.

        Args:
            source_nodes (Iterable[int]): Iterable of source node IDs.
            target_nodes (Iterable[int]): Iterable of target node IDs.
            weight (str): Edge attribute to use as weight for shortest path calculation.
        """
        assert weight == "length", "Currently only 'length' weight is supported."
        assert len(source_nodes) == len(target_nodes), "Source and target nodes must have the same length."

        paths = [self.pairwise_paths[src][tgt] for src, tgt in zip(source_nodes, target_nodes)]
        distances = [self.pairwise_distances[src][tgt] for src, tgt in zip(source_nodes, target_nodes)]
        ret = pd.DataFrame({"paths": paths, "distances": distances})
        return ret
