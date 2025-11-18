from .pre_processing import preprocess_graph
import typing as t

import networkx as nx
import osmnx as ox

from .osmnx_constants import OSMNXConstants as C


def sparsify_graph(G_: nx.MultiDiGraph) -> nx.MultiDiGraph:
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
    G = G_.copy()

    # 2. Remove non-major edges
    G_ret = keep_major_roads_only(G)
    G_ret = preprocess_graph(G_ret)

    # 3. Remove Lincoln Tunnel
    G_ret = remove_lincoln_tunnel(G_ret)
    G_ret = preprocess_graph(G_ret)

    # 4 Remove nodes that are now isolated, or have degree 2 (i.e., pass-through nodes)
    G_ret = remove_pass_through_nodes(G_ret)
    G_ret = preprocess_graph(G_ret)

    # 5. Clip to Manhattan core
    G_ret = clip_manhattan_core(G_ret)
    G_ret = preprocess_graph(G_ret)

    assert isinstance(G_ret, nx.MultiDiGraph)
    print(f"Final simplified graph: {G_ret.number_of_nodes()=}, {G_ret.number_of_edges()=}")
    return G_ret


def keep_major_roads_only(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Keeps only major roads in the graph without further simplification.

    Args:
        G (nx.MultiDiGraph): The original graph from OSMnx.

    Returns:
        nx.MultiDiGraph: A new graph containing only major roads.
    """
    # 2. Remove non-major edges
    edges_to_remove = []
    for u, v, key, data in G.edges(keys=True, data=True):
        highway_type = data.get("highway", "default")
        assert isinstance(highway_type, str), f"Unexpected highway type format: {highway_type}"

        if highway_type not in C.MAJOR_ROAD_TYPES_SET():
            edges_to_remove.append((u, v, key))

    G_major = _remove_edges_and_simplify(G, edges_to_remove)
    return G_major


def remove_pass_through_nodes(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Removes pass-through nodes (nodes with degree 2) from the graph.

    Args:
        G (nx.MultiDiGraph): The original graph from OSMnx.

    Returns:
        nx.MultiDiGraph: A new graph with pass-through nodes removed.
    """
    nodes_to_remove = []
    for node in G.nodes():
        preds = set(G.predecessors(node))
        succs = set(G.successors(node))
        neighbors = preds | succs

        # e.g. “exactly 2 neighbors and no self-loop”
        if len(neighbors) <= 2:
            nodes_to_remove.append(node)

    print(f"Removing {len(nodes_to_remove)} isolated or pass-through nodes...")
    G.remove_nodes_from(nodes_to_remove)
    G_simplified = _simplify_graph_safely(G)
    G_ret = _keep_largest_weakly_connected_component(G_simplified)
    return G_ret


def clip_manhattan_core(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    G_clipped = ox.truncate.truncate_graph_bbox(G, C.MANHATTAN_BOX, truncate_by_edge=False)
    return G_clipped


def _remove_edges_and_simplify(G_local: nx.MultiDiGraph, edges_to_remove: list) -> nx.MultiDiGraph:
    """
    Remove edges that represent non-major roads and simplify the graph topology.

    Args:
        G_local (nx.MultiDiGraph): The graph to modify.
        edges_to_remove (List[Tuple[int, int, int]]): List of edges to remove.
    Returns:
        nx.MultiDiGraph: The modified graph.
    """
    print(f"Removing {len(edges_to_remove)} non-major edges...")
    G_local.remove_edges_from(edges_to_remove)

    # 3. Keep only the largest weakly connected component
    G_sparsified = _keep_largest_weakly_connected_component(G_local)

    print(f"Graph after edge removal: {G_sparsified.number_of_nodes()=}, {G_sparsified.number_of_edges()=}")

    G_simplified = _simplify_graph_safely(G_sparsified)

    return G_simplified


import networkx as nx
from .osmnx_constants import OSMNXConstants as C

# very rough box around the Lincoln Tunnel entrance in west midtown
LINCOLN_WEST = -74
LINCOLN_EAST = -73.990
LINCOLN_SOUTH = 40.755
LINCOLN_NORTH = 40.763
LINCOLN_BOX = (LINCOLN_WEST, LINCOLN_SOUTH, LINCOLN_EAST, LINCOLN_NORTH)


def _in_bbox(x: float, y: float, box: tuple[float, float, float, float]) -> bool:
    west, south, east, north = box
    return west <= x <= east and south <= y <= north


def remove_lincoln_tunnel(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Remove Lincoln Tunnel edges and nearby motorway/trunk ramps in west midtown.
    """
    G2 = G.copy()
    edges_to_remove: list[tuple] = []

    for u, v, k, data in G2.edges(keys=True, data=True):
        name = data.get("name")
        highway = data.get("highway")

        # normalise
        if isinstance(name, list):
            names = name
        elif isinstance(name, str):
            names = [name]
        else:
            names = []

        if isinstance(highway, list):
            htypes = highway
        elif isinstance(highway, str):
            htypes = [highway]
        else:
            htypes = []

        # node coords (lon=x, lat=y)
        ux = G2.nodes[u]["x"]
        uy = G2.nodes[u]["y"]
        vx = G2.nodes[v]["x"]
        vy = G2.nodes[v]["y"]

        in_lincoln_box = _in_bbox(ux, uy, LINCOLN_BOX) or _in_bbox(vx, vy, LINCOLN_BOX)

        # 1) any edge explicitly named Lincoln Tunnel
        name_is_lincoln = any("Lincoln Tunnel" in n for n in names)

        # 2) or any motorway/trunk segment inside the small box
        is_major_highway = any(h in {"motorway", "motorway_link", "trunk", "trunk_link"} for h in htypes)

        if name_is_lincoln or (in_lincoln_box and is_major_highway):
            edges_to_remove.append((u, v, k))

    G2.remove_edges_from(edges_to_remove)

    # clean up isolated nodes created by this removal
    isolates = list(nx.isolates(G2))
    G2.remove_nodes_from(isolates)
    G2 = _simplify_graph_safely(G2)

    return G2


def _keep_largest_weakly_connected_component(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    components = list(nx.weakly_connected_components(G))
    if not components:
        print("Warning: No components found. Returning empty graph.")
        return nx.MultiDiGraph()

    largest_component = max(components, key=len)
    G_ret = t.cast(nx.MultiDiGraph, G.subgraph(largest_component).copy())
    return G_ret


def _simplify_graph_safely(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    print("Cleaning all edge attributes before simplification...")
    keys_merged = set()
    for u, v, key, data in G.edges(keys=True, data=True):
        for attr_key, attr_value in data.items():
            data[attr_key] = _merge_instructions(attr_key, attr_value, u, v, key, keys_merged)

    G.graph["simplified"] = False
    G_simplified = ox.simplify_graph(G)
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
    # special cases
    if attr_key in {"osmid"}:
        val = attr_value if not isinstance(attr_value, list) else attr_value[0]
        return int(val)
    if attr_key in {"oneway", "reversed"} and isinstance(attr_value, list):
        val = round(sum([True, False, False]) / len(attr_value))
        return bool(val)

    # general cases
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
