import networkx as nx


def get_charging_nodes(G: nx.MultiDiGraph):
    """
    Retrieves the list of node IDs that have charging stations.

    Args:
        G (nx.MultiDiGraph): The graph to inspect.

    Returns:
        List[int]: A list of node IDs with charging stations.
    """
    charging_nodes: dict[str, int] = {}
    for node_id, data in G.nodes(data=True):
        if (num_chargers := data.get("charging_station", 0)) > 0:
            charging_nodes[node_id] = num_chargers
    return charging_nodes
