import typing as t

import networkx as nx
import osmnx as ox

from .osmnx_constants import OSMNXConstants as C

# 1. Define the road types you want to KEEP
# Try removing 'secondary' if this is still too dense

MAJOR_ROAD_TYPES_SET = C.MAJOR_ROAD_TYPES_SET


def sparsify_graph(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Sparsifies a graph by keeping only major roads and removing
    all disconnected nodes and redundant pass-through nodes.

    Args:
        G (nx.MultiDiGraph): The original graph from OSMnx. This graph is
            assumed to have already been simplified by OSMnx.

    Returns:
        nx.MultiDiGraph: A new, simplified graph containing only the
                         largest connected component of major roads.
    """
    # 1. Copy so we don't mutate the original
    G_major = G.copy()

    # 2. Remove non-major edges
    edges_to_remove = []
    for u, v, key, data in G_major.edges(keys=True, data=True):
        highway_type = data.get("highway", "default")
        assert isinstance(highway_type, str), f"Unexpected highway type format: {highway_type}"

        if highway_type not in MAJOR_ROAD_TYPES_SET:
            edges_to_remove.append((u, v, key))

    print(f"Removing {len(edges_to_remove)} non-major edges...")
    G_major.remove_edges_from(edges_to_remove)

    # 3. Keep only the largest weakly connected component
    components = list(nx.weakly_connected_components(G_major))
    if not components:
        print("Warning: No components found. Returning empty graph.")
        return nx.MultiDiGraph()

    largest_component = max(components, key=len)
    G_sparsified = t.cast(nx.MultiDiGraph, G_major.subgraph(largest_component).copy())

    print(f"Graph after edge removal: {G_sparsified.number_of_nodes()=}, {G_sparsified.number_of_edges()=}")

    print("Cleaning all edge attributes before simplification...")
    keys_merged = set()
    for u, v, key, data in G_sparsified.edges(keys=True, data=True):
        for attr_key, attr_value in data.items():
            data[attr_key] = _merge_instructions(attr_key, attr_value, u, v, key, keys_merged)

    print(f"Merged attributes: {keys_merged}")
    print("Simplifying topology...")
    G_sparsified.graph["simplified"] = False
    G_simplified = ox.simplify_graph(G_sparsified)

    assert isinstance(G_simplified, nx.MultiDiGraph)
    print(f"Final simplified graph: {G_simplified.number_of_nodes()=}, {G_simplified.number_of_edges()=}")
    return G_simplified


def _merge_instructions(attr_key: str, attr_value: t.Any, u: int, v: int, key: int, keys_merged: set) -> t.Any:
    """
    Merges list or dict attributes into a single representative value.

    Args:
        attr_key (str): The attribute key.
        attr_value (Any): The attribute value to merge.
        u (int): The start node of the edge.
        v (int): The end node of the edge.
        key (int): The edge key.

    Returns:
        Any: The merged attribute value.
    """
    if isinstance(attr_value, list):
        keys_merged.add(attr_key)
        if len(attr_value) == 0:
            return None
        if len(attr_value) == 1:
            return attr_value[0]
        else:
            return tuple(attr_value)
    elif isinstance(attr_value, dict):
        keys_merged.add(attr_key)
        return str(attr_value)
    return attr_value
