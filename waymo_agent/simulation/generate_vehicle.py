import random

import networkx as nx
import numpy as np

from waymo_agent.data_classes.dataclasses import VehicleState, VehicleStatusEnum


def init_vehicle(G: nx.MultiDiGraph, vehicle_id: int, node_id: int | None = None) -> VehicleState:
    """
    Initialize a vehicle at a given node or a random node in the graph.
    """
    node = G.nodes.get(node_id, random.choice(list(G.nodes(data=True))))

    vehicle_state = VehicleState(
        vehicle_id=vehicle_id,
        loc_norm=(node["x_norm"], node["y_norm"]),
        battery=float(np.random.uniform(0.6, 1.0)),
        status=VehicleStatusEnum.IDLE,
    )
    return vehicle_state


def init_vehicles(
    G: nx.MultiDiGraph,
    num_vehicles: int,
    rng: np.random.Generator | None = None,
) -> list[VehicleState]:
    """
    Initialize `num_vehicles` vehicles at random nodes in the graph.

    Uses numpy to sample node indices and batteries in parallel.
    """
    if rng is None:
        rng = np.random.default_rng()

    node_ids = list(G.nodes)
    sampled_idx = rng.integers(0, len(node_ids), size=num_vehicles)
    batteries = rng.uniform(0.6, 1.0, size=num_vehicles)
    status = np.full(shape=num_vehicles, fill_value=int(VehicleStatusEnum.IDLE), dtype=np.int8)

    vehicles = []
    for i in range(num_vehicles):
        node = G.nodes[node_ids[sampled_idx[i]]]
        vehicle_state = VehicleState(
            vehicle_id=i,
            loc_norm=(node["x_norm"], node["y_norm"]),
            battery=float(batteries[i]),
            status=VehicleStatusEnum(status[i]),
        )
        vehicles.append(vehicle_state)
    return vehicles
