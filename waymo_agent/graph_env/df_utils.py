import networkx as nx
import pandas as pd


def nodes_to_df(G: nx.MultiDiGraph) -> pd.DataFrame:
    """
    Convert the nodes of a NetworkX MultiDiGraph to a pandas DataFrame.

    Args:
        G (nx.MultiDiGraph): The input graph.

    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to a node in the graph,
                      with columns for node attributes.
    """
    nodes_data = []
    attrs = {}
    for node_id, attrs in G.nodes(data=True):
        node_record = {"node_id": node_id}
        node_record.update(attrs)
        nodes_data.append(node_record)

    nodes_df = pd.DataFrame(nodes_data, columns=["node_id"] + list(attrs.keys()))
    return nodes_df


def edges_to_df(G: nx.MultiDiGraph) -> pd.DataFrame:
    """
    Convert the edges of a NetworkX MultiDiGraph to a pandas DataFrame.

    Args:
        G (nx.MultiDiGraph): The input graph.

    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to an edge in the graph,
                      with columns for edge attributes.
    """
    edges_data = []
    attrs = {}
    for u, v, key, attrs in G.edges(keys=True, data=True):
        edge_record = {"source": u, "target": v, "key": key}
        attrs_clean = attrs | {"geometry": -1}
        edge_record.update(attrs_clean)
        edges_data.append(edge_record)

    edges_df = pd.DataFrame(edges_data, columns=["source", "target", "key"] + list(attrs.keys()))
    return edges_df
