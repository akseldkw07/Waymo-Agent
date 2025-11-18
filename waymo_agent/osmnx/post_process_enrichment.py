"""
Assign charging stations, lambda values
"""

import networkx as nx
from ..envs.dataclasses import EnvConfig
import numpy as np

ENV = EnvConfig()


def enrich_graph(G: nx.MultiDiGraph, config: EnvConfig = ENV):
    """
    Enrich the graph with charging stations and lambda values.
    1. Assign charging stations to nodes proportional to its degree centrality.
    2. Assign lambda values to each node proportional to its degree centrality.
    """
    _assign_chargers(G, config)
    _assign_lambda_values(G, config)


def _assign_chargers(G: nx.MultiDiGraph, config: EnvConfig):
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


def _assign_lambda_values(G: nx.MultiDiGraph, config: EnvConfig):
    """
    Assign lambda values to each node proportional to its degree centrality.
    """
    degree_centrality = nx.degree_centrality(G)
    max_centrality = max(degree_centrality.values())
    min_centrality = min(degree_centrality.values())

    total_lambda = config.lambda_per_node * len(G.nodes)

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
        lambda_values = {node: (scaled_centrality[node] / sum_scaled) * total_lambda for node in G.nodes}
    else:
        lambda_values = {node: total_lambda / len(G.nodes) for node in G.nodes}

    # Add noise to lambda values
    noise_std = config.lambda_variation_coef * (total_lambda / len(G.nodes))
    for node in G.nodes:
        noise = 0.0
        # noise = np.random.normal(0, noise_std)
        noise = np.random.uniform(-noise_std, noise_std)

        lambda_values[node] = max(0.0, lambda_values[node] + noise)
        G.nodes[node]["lambda"] = lambda_values[node]
