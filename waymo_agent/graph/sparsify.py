import networkx as nx
import osmnx as ox
import typing as t

# 1. Define the road types you want to KEEP
# Try removing 'secondary' if this is still too dense
MAJOR_ROAD_TYPES = ["motorway", "motorway_link", "trunk", "primary", "secondary"]


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
        if isinstance(highway_type, list):
            highway_type = highway_type[0]

        if highway_type not in MAJOR_ROAD_TYPES:
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

    print(f"Graph after edge removal: {G_sparsified.number_of_nodes()} nodes")

    # 4. *** FIX: PRE-CLEANING STEP ***
    # We must run this again to prevent the 'unhashable type: list'
    # error in the simplify_graph step.
    print("Cleaning edge attributes before simplification...")
    for u, v, key, data in G_sparsified.edges(keys=True, data=True):
        if isinstance(data.get("highway"), list):
            data["highway"] = data["highway"][0]
        if isinstance(data.get("name"), list):
            data["name"] = data["name"][0]

    # 5. *** CRITICAL STEP: SIMPLIFY TOPOLOGY AGAIN ***
    # This removes all the "pass-through" nodes we created
    # by deleting the minor roads. This is what will
    # get your node count down.
    print("Simplifying topology...")
    G_sparsified.graph["simplified"] = False
    G_simplified = ox.simplify_graph(G_sparsified)

    assert isinstance(G_simplified, nx.MultiDiGraph)
    print(f"Final simplified graph: {G_simplified.number_of_nodes()} nodes")
    return G_simplified
