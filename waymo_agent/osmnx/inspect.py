import random
import typing as t

import networkx as nx
from matplotlib.axes import Axes


def inspect(G: nx.MultiDiGraph) -> None:
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
        sample_node = random.choice(list(G.nodes(data=True)))
        print(f"Sample node attributes: {sample_node[1]}\n")

    if num_edges == 0:
        print("Graph has no edges.")
    else:
        sample_edge = random.choice(list(G.edges(data=True)))
        print(f"Sample edge attributes: {sample_edge[2]}")


def max_speed_distribution(G: nx.MultiDiGraph) -> t.Dict[str, int]:
    """
    Computes the distribution of maximum speed limits in the graph.

    Args:
        G (nx.MultiDiGraph): The graph to analyze.

    Returns:
        Dict[str, int]: A dictionary with speed limits as keys and their counts as values.
    """
    speed_counts: t.Dict[str, int] = {}
    for u, v, k, data in G.edges(keys=True, data=True):
        maxspeed = data.get("maxspeed")
        if isinstance(maxspeed, list):
            for speed in maxspeed:
                speed_counts[speed] = speed_counts.get(speed, 0) + 1
        elif maxspeed is not None:
            speed_counts[maxspeed] = speed_counts.get(maxspeed, 0) + 1
    return speed_counts


def plot_maxspeed(G: nx.MultiDiGraph, ax: Axes | None = None): ...
