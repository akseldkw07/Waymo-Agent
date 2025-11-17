import networkx as nx


def sparsify_graph(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Sparsifies a graph by keeping only major roads and removing
    all disconnected nodes.

    Args:
        G (nx.MultiDiGraph): The original graph from OSMnx.

    Returns:
        nx.MultiDiGraph: A new graph containing only the
                         largest connected component of major roads.
    """

    # 1. Define the road types you want to KEEP
    # All other types will be discarded.
    major_road_types = ["motorway", "motorway_link", "trunk", "primary"]

    # 2. Create a new graph containing ONLY edges that match our list
    # We make a copy so we don't destroy the original 'G'
    G_major = G.copy()

    # Get a list of all edges that are *not* major roads
    edges_to_remove = []
    for u, v, key, data in G_major.edges(keys=True, data=True):
        highway_type = data.get("highway", "default")

        # Handle cases where highway type is a list
        if isinstance(highway_type, list):
            highway_type = highway_type[0]

        if highway_type not in major_road_types:
            edges_to_remove.append((u, v, key))

    # Remove the non-major edges from our copied graph
    G_major.remove_edges_from(edges_to_remove)

    # 3. Remove all disconnected nodes
    # After removing edges, many nodes will be "isolates" (no edges).
    # We find the largest "strongly connected component" to keep only
    # the main, interconnected road network.

    # Get all connected components
    # (We use 'weakly' for street networks, as 'strongly' is too strict for one-way roads)
    components = list(nx.weakly_connected_components(G_major))

    if not components:
        print("Warning: No components found. Returning empty graph.")
        return nx.MultiDiGraph()

    # Find the largest component (the one with the most nodes)
    largest_component = max(components, key=len)

    # Create a new graph containing ONLY the nodes from that component
    G_sparsified = G_major.subgraph(largest_component).copy()

    assert isinstance(G_sparsified, nx.MultiDiGraph)

    return G_sparsified
