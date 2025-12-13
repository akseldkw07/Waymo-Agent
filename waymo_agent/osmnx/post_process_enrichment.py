"""
Assign charging stations, lambda values
"""

import typing as t

import networkx as nx
import numpy as np

from waymo_agent.osmnx.euclidean_L2_embed import PRECISION, embed_L2

from .inspect import sum_graph_attr

if t.TYPE_CHECKING:
    from ..data_classes.config import EnvConfig


def enrich_graph(G: nx.MultiDiGraph, config: "EnvConfig | None" = None):
    """
    Enrich the graph with charging stations and lambda values.
    1. Assign charging stations to nodes proportional to its degree centrality.
    2. Assign lambda values to each node proportional to its degree centrality.
    3. Calculate travel time for each edge based on maxspeed and length.
    4. Embed L2 normalized coordinates into each node.
    """
    if config is None:
        from ..data_classes.config import EnvConfig

        config = EnvConfig()

    # _assign_chargers(G, config) TODO re-enable when chargers are used
    assign_lambda_values(G, config)
    _calculate_time_edge(G)
    embed_L2(G)


def _assign_chargers(G: nx.MultiDiGraph, config: "EnvConfig"):
    """
    Assign charging stations to nodes proportional to its degree centrality.
    """
    degree_centrality = nx.degree_centrality(G)
    num_charging_nodes = int(len(G.nodes) * config.chargable_node_ratio)
    centralities = [degree_centrality[node] for node in G.nodes]
    charging_nodes = np.random.choice(
        list(G.nodes),
        size=num_charging_nodes,
        replace=False,
        p=np.array(centralities) / np.sum(centralities),
    )

    for node in G.nodes:
        G.nodes[node]["num_chargers"] = node in charging_nodes
    print(f"Assigned {len(charging_nodes)} charging nodes.")


def assign_lambda_values(G: nx.MultiDiGraph, config: "EnvConfig"):
    """
    Assign lambda values to each node proportional to its degree centrality and config.lambda_per_node
    """
    degree_centrality = nx.degree_centrality(G)
    max_centrality = max(degree_centrality.values())
    min_centrality = min(degree_centrality.values())

    target_lambda = config.lambda_per_node * len(G.nodes)

    # Normalize degree centrality to [0, 1]
    if max_centrality > min_centrality:
        scaled_centrality = {
            node: (degree_centrality[node] - min_centrality) / (max_centrality - min_centrality) for node in G.nodes
        }
    else:
        # If all centralities are equal, assign equal values
        scaled_centrality = {node: 1.0 for node in G.nodes}

    # Scale so that the sum of lambda values is total_lambda
    sum_scaled = sum(scaled_centrality.values())
    if sum_scaled > 0:
        lambda_values = {node: (scaled_centrality[node] / sum_scaled) * target_lambda for node in G.nodes}
    else:
        lambda_values = {node: target_lambda / len(G.nodes) for node in G.nodes}

    # Add noise to lambda values
    noise_std = config.lambda_variation_coef * (target_lambda / len(G.nodes))
    for node in G.nodes:
        noise = 0.0
        # noise = np.random.normal(0, noise_std)
        noise = np.random.uniform(-noise_std, noise_std)

        lambda_values[node] = max(0.0, lambda_values[node] + noise)
        G.nodes[node]["lambda"] = round(float(lambda_values[node]), PRECISION)

    sum_lambda = sum_graph_attr(G, "lambda", "node")
    for node in G.nodes:
        G.nodes[node]["lambda"] = G.nodes[node]["lambda"] * (target_lambda / sum_lambda)

    sum_lambda = sum_graph_attr(G, "lambda", "node")
    print(f"Assigned lambda values to nodes. Total lambda: {sum_lambda:.4f} (target: {target_lambda:.4f})")


def _calculate_time_edge(G: nx.MultiDiGraph):
    """
    TODO this is redundant, should've just used os.routing.add_edge_travel_times()
    https://osmnx.readthedocs.io/en/stable/user-reference.html#osmnx.routing.add_edge_travel_times
    """
    for u, v, k, data in G.edges(keys=True, data=True):
        speed = data.get("maxspeed", 30)  # default to 30 km/h if not specified
        length = data.get("length", 0)  # length in meters
        length_km = length / 1000.0  # convert to km
        time_minutes = _calculate_time(speed, length_km)
        G.edges[u, v, k]["travel_time_minutes"] = round(time_minutes, PRECISION)


def _calculate_time(speed_kmh: float, distance_km: float) -> float:
    """
    Calculate time in minutes given speed in km/h and distance in km.
    """
    if speed_kmh <= 0:
        raise ValueError("Speed must be greater than zero.")
    time_hours = distance_km / speed_kmh
    time_minutes = time_hours * 60.0
    return round(time_minutes, PRECISION)
