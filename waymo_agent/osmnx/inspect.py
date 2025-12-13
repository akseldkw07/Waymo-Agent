"""
Notebook functions for inspecting OSMnx graphs.
"""

import logging
import random
import typing as t
from collections import defaultdict
from math import ceil

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from waymo_agent.data_classes.osmnx_constants import OSMNXConstants

logger = logging.getLogger(__name__)

ATTR_LITERAL = OSMNXConstants.NODE_ATTR_LITERAL | OSMNXConstants.EDGE_ATTR_LITERAL
EDGE_NODE_LITERAL = t.Literal["edge", "node"]


def attrs_in_graph(G: nx.MultiDiGraph, edge_node: EDGE_NODE_LITERAL = "edge"):
    """
    Collect all unique attribute keys present in the graph's edges or nodes.

    Args:
        G (nx.MultiDiGraph): The graph to inspect.
        edge_node (str): Specify whether to inspect 'edge' or 'node' attributes.

    Returns:
        Set[str]: A set of unique attribute keys.
    """
    print(f"Inspecting {edge_node.upper()} attributes in the graph.")
    attrs = defaultdict(set)
    if edge_node == "edge":
        for u, v, k, data in G.edges(keys=True, data=True):
            for key in OSMNXConstants.EDGE_ATTR_LITERAL.__args__:
                attrs[key].add(type(data.get(key)).__name__)
    elif edge_node == "node":
        for node, data in G.nodes(data=True):
            for key in OSMNXConstants.NODE_ATTR_LITERAL.__args__:
                attrs[key].add(type(data.get(key)).__name__)
    return attrs


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
        print(f"Sample edge attributes: {sample_edge[2]}\n")

    print(OSMNXConstants.graph_edge_units())


def edge_attr_counts(G: nx.MultiDiGraph, attr_str: ATTR_LITERAL):
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


def hist_edge_or_node_attr(
    G: nx.MultiDiGraph, attr_str: ATTR_LITERAL, edge_node: EDGE_NODE_LITERAL = "edge", ax: Axes | None = None
):
    vals = []

    if edge_node == "edge":
        for u, v, k, data in G.edges(keys=True, data=True):
            value = data.get(attr_str)
            if isinstance(value, (int, float)):
                vals.append(value)
            elif isinstance(value, list):
                vals.extend([v for v in value if isinstance(v, (int, float))])
    elif edge_node == "node":
        for node, data in G.nodes(data=True):
            value = data.get(attr_str)
            if isinstance(value, (int, float)):
                vals.append(value)
            elif isinstance(value, list):
                vals.extend([v for v in value if isinstance(v, (int, float))])

    if ax is None:
        fig, ax = plt.subplots()

    if vals:
        try:
            ax.hist(vals, bins="rice", edgecolor="black")
            ax.set_xlabel(f"{attr_str}")
            ax.set_ylabel("Frequency")
            ax.set_title(f"{attr_str} Distribution")
        except Exception:
            ax.text(0.5, 0.5, f"Error plotting {attr_str}", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, f"No plottable data for {attr_str}!!", ha="center", va="center", transform=ax.transAxes)


def hist_ALL(
    G: nx.MultiDiGraph, edge_node: EDGE_NODE_LITERAL = "edge", fig_ax: tuple[Figure, np.ndarray] | None = None
):
    if edge_node == "edge":
        attrs = attrs_in_graph(G, edge_node="edge")
    elif edge_node == "node":
        attrs = attrs_in_graph(G, edge_node="node")

    if fig_ax is None:
        figwidth = 2
        figlen = ceil(len(attrs) / figwidth)

        fig, axes = plt.subplots(figlen, figwidth, figsize=(6 * figwidth, 4 * figlen))
        plt.close(fig)  # to avoid showing empty plot immediately
    else:
        fig, axes = fig_ax
        assert axes.size >= len(attrs)

    attr_keys = list(attrs.keys())
    for idx, ax in enumerate(axes.ravel()):
        if idx < len(attr_keys):
            attr = attr_keys[idx]
            hist_edge_or_node_attr(G, attr, edge_node=edge_node, ax=ax)
        else:
            ax.axis("off")
    fig.subplots_adjust(hspace=0.45)
    return fig


def plot_maxspeed(G: nx.MultiDiGraph, ax: Axes | None = None):
    return hist_edge_or_node_attr(G, "maxspeed", ax=ax)


def collect_unique_graph_attr(G: nx.MultiDiGraph, attr_str: ATTR_LITERAL) -> set[t.Any]:
    """Helper to collect all unique values of a given edge attribute in the graph."""
    values = set()
    for u, v, k, data in G.edges(keys=True, data=True):
        value = data.get(attr_str)
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

        logger.warning(f"No values found for attribute '{attr_str}'. Sample edge attributes: {attrs}")
    return values


def sum_graph_attr(G: nx.MultiDiGraph, attr_str: ATTR_LITERAL, edge_node: EDGE_NODE_LITERAL = "edge") -> float:
    """Helper to sum up all numeric values of a given edge attribute in the graph."""
    total = 0.0
    if edge_node == "edge":
        attr_values = nx.get_edge_attributes(G, attr_str)
        total = sum(attr_values.values(), total)
    elif edge_node == "node":
        attr_values = nx.get_node_attributes(G, attr_str)
        total = sum(attr_values.values(), total)
    return total
