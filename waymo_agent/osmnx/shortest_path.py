import typing as t

import networkx as nx
import numpy as np  # already imported above in your file


def calculate_longest_path(G: nx.MultiDiGraph, weight: str = "length") -> tuple[float, list[int]]:
    """
    Calculate the longest shortest path in the graph G using the specified edge weight.

    Args:
        G (nx.MultiDiGraph): The input graph.
        weight (str): The edge attribute to use as weight for shortest path calculation.

    Returns:
        tuple[float, list[int]]: A tuple (length, path), where `length` is the length
        of the longest shortest path in the graph and `path` is the corresponding
        list of node IDs along that path. If the graph is empty or disconnected in a
        way that yields no paths, returns (0.0, []).
    """
    max_length: float = 0.0
    src_best: t.Any | None = None
    dst_best: t.Any | None = None

    # All-pairs shortest path lengths
    all_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight=weight))

    for src_node, lengths in all_lengths.items():
        if not lengths:
            continue
        # node with max distance from this source
        local_dst, local_max = max(lengths.items(), key=lambda kv: kv[1])
        if local_max > max_length:
            max_length = float(local_max)
            src_best = src_node
            dst_best = local_dst

    if src_best is None or dst_best is None or max_length == 0.0:
        return 0.0, []

    # Recover the corresponding path
    longest_path = nx.shortest_path(G, source=src_best, target=dst_best, weight=weight)
    return max_length, longest_path


def cheapest_path():
    """
    TODO implement cheapest path calculation

    Take in graph, config, start node, end node
    Use config cost parameters to weight edges based on length, travel time,
    """
