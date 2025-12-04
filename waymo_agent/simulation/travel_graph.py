import networkx as nx

from waymo_agent.data_classes.dataclasses import PathPosition


def step_along_route(G: nx.MultiDiGraph, pos: PathPosition, speed_kmh: float, num_mins: float = 1) -> PathPosition:
    """
    Move along the current route for num_mins minutes at speed_kmh.
    Handles crossing multiple edges if distance allows.
    TODO simplify,
    """
    raise NotImplementedError("I will implement this function later, with transition model")
    # Convert km/h to m/min
    speed_m_per_min = speed_kmh * 1000.0 / 60.0
    remaining_dist = speed_m_per_min * num_mins

    route = pos.route
    edge_idx = pos.edge_idx
    dist_on_edge = pos.dist_on_edge

    while remaining_dist > 0 and edge_idx < len(route) - 1:
        u = route[edge_idx]
        v = route[edge_idx + 1]

        # pick one edge (first key) – you can be more careful here if MultiDiGraph
        key = 0
        edge_data = G[u][v][key]
        edge_len = float(edge_data.get("length", 0.0))  # meters

        # distance remaining on this edge
        dist_left_on_edge = max(0.0, edge_len - dist_on_edge)

        if remaining_dist < dist_left_on_edge:
            # we stay on the current edge
            dist_on_edge += remaining_dist
            remaining_dist = 0.0
        else:
            # we reach the end of this edge and move to the next
            remaining_dist -= dist_left_on_edge
            edge_idx += 1
            dist_on_edge = 0.0  # new edge, starting at its beginning

    return PathPosition(route=route, edge_idx=edge_idx, dist_on_edge=dist_on_edge)
