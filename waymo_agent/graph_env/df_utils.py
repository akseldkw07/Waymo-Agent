import typing as t

import networkx as nx
import pandas as pd

from waymo_agent.data_classes.enriched_df_base import BASE_CLASS_ATTRS, EnrichedDF


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


T = t.TypeVar("T", bound=pd.DataFrame)


def validate_typed_df_keys(
    df: pd.DataFrame | dict, df_type: t.Type[T], action: t.Literal["warn", "raise"] = "raise"
) -> bool:
    """
    Validate that a pandas DataFrame conforms to the specified typed DataFrame structure.
    TODO convert down to warning instead of raising
    """
    exp_cols = df_type.target_dtypes.keys() if isinstance(df_type, EnrichedDF) else df_type.__annotations__.keys()
    expected_columns = set(exp_cols) - BASE_CLASS_ATTRS
    actual_columns = set(df.keys()) - BASE_CLASS_ATTRS

    missing = expected_columns - actual_columns
    extra = actual_columns - expected_columns

    if action == "raise":
        assert (
            not missing and not extra
        ), f"DataFrame columns do not match expected structure. Missing: {missing}, Extra: {extra}"

    if missing:
        print(f"WARNING: Missing columns in DataFrame: {missing}")
    if extra:
        print(f"WARNING: Extra columns in DataFrame: {extra}")

    return not missing and not extra
