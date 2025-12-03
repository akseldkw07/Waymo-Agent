"""
Notebook functions for inspecting OSMnx graphs.
"""

import logging
import random
import typing as t

import networkx as nx
from matplotlib.axes import Axes

logger = logging.getLogger(__name__)


def inspect_graph(G: nx.MultiDiGraph) -> None:
    """
    Inspects the graph by printing basic information about nodes and edges.

    Args:
        G (nx.MultiDiGraph): The graph to inspect.
    """

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    print(f"Graph has {num_nodes} nodes and {num_edges} edges.")

    if num_nodes == 0:
        print("Graph has no nodes.")
    else:
        sample_node_id = random.choice(list(G.nodes(data=False)))
        sample_node = G.nodes[sample_node_id]
        print(f"Sample node -{sample_node_id=}- attributes: {sample_node}\n")

    if num_edges == 0:
        print("Graph has no edges.")
    else:
        sample_edge = random.choice(list(G.edges(data=True)))
        print(f"Sample edge attributes: {sample_edge[2]}")


def edge_attr_counts(G: nx.MultiDiGraph, attr_str: str):
    """
    Computes the distribution of a given edge attribute in the graph.

    Args:
        G (nx.MultiDiGraph): The graph to analyze.

    Returns:
        Dict[str, int]: A dictionary with speed limits as keys and their counts as values.

    NOTE this only works for attributes that are integers or lists of integers.
    """
    speed_counts: dict[str, int] = {}
    for u, v, k, data in G.edges(keys=True, data=True):
        attr = data.get(attr_str)
        if isinstance(attr, list):
            for val in attr:
                speed_counts[val] = speed_counts.get(val, 0) + 1
        elif attr is not None:
            speed_counts[attr] = speed_counts.get(attr, 0) + 1
    return speed_counts


def max_speed_distribution(G: nx.MultiDiGraph):
    return edge_attr_counts(G, "maxspeed")


def plot_edge_or_node_attr(
    G: nx.MultiDiGraph, attr: str, edge_node: t.Literal["edge", "node"] = "edge", ax: Axes | None = None
):
    vals = []

    if edge_node == "edge":
        for u, v, k, data in G.edges(keys=True, data=True):
            value = data.get(attr)
            if isinstance(value, (int, float)):
                vals.append(value)
            elif isinstance(value, list):
                vals.extend([v for v in value if isinstance(v, (int, float))])
    elif edge_node == "node":
        for node, data in G.nodes(data=True):
            value = data.get(attr)
            if isinstance(value, (int, float)):
                vals.append(value)
            elif isinstance(value, list):
                vals.extend([v for v in value if isinstance(v, (int, float))])

    if ax is None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

    if vals:
        ax.hist(vals, bins="rice", edgecolor="black")
        ax.set_xlabel(f"{attr}")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{attr} Distribution")
    else:
        ax.text(0.5, 0.5, "No plottable data", ha="center", va="center", transform=ax.transAxes)


def plot_maxspeed(G: nx.MultiDiGraph, ax: Axes | None = None):
    return plot_edge_or_node_attr(G, "maxspeed", ax=ax)


def collect_unique_graph_attr(G: nx.MultiDiGraph, attr: str) -> set[t.Any]:
    """Helper to collect all unique values of a given edge attribute in the graph."""
    values = set()
    for u, v, k, data in G.edges(keys=True, data=True):
        value = data.get(attr)
        if value is not None:
            if isinstance(value, list):
                values.update(value)
            else:
                values.add(value)

    if not values:
        attrs = set()
        for i in range(5):
            sample_edge = random.choice(list(G.edges(data=True)))
            attrs.update(sample_edge[2].keys())

        logger.warning(f"No values found for attribute '{attr}'. Sample edge attributes: {attrs}")
    return values


def sum_graph_attr(G: nx.MultiDiGraph, attr: str, edge_node: t.Literal["edge", "node"] = "edge") -> float:
    """Helper to sum up all numeric values of a given edge attribute in the graph."""
    total = 0.0
    if edge_node == "edge":
        for u, v, k, data in G.edges(keys=True, data=True):
            value = data.get(attr)
            if isinstance(value, (int, float)):
                total += value
            elif isinstance(value, list):
                raise ValueError(f"Attribute '{attr}' has list values on edge ({u}, {v}, {k}), cannot sum.")
    elif edge_node == "node":
        for node, data in G.nodes(data=True):
            value = data.get(attr)
            if isinstance(value, (int, float)):
                total += value
    return total
