import networkx as nx
import numpy as np
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


def masked_assign(dst: pd.DataFrame, mask: np.ndarray, src: pd.DataFrame, cols: list[str] | None = None) -> None:

    assert mask.sum() == len(src), f"Mask sum {mask.sum()} != src len {len(src)}"
    if cols is None:
        assert list(dst.columns) == list(src.columns), f"Dst columns {dst.columns} != src columns {src.columns}"

        for c in dst.columns:
            dst.loc[mask, c] = src[c].to_numpy()
    else:
        assert set(cols).issubset(set(dst.columns)) and set(cols).issubset(
            set(src.columns)
        ), f"Cols {cols} not subset of dst columns {dst.columns} and src columns   {src.columns}"
        for c in cols:
            dst.loc[mask, c] = src[c].to_numpy()


def trim_true_mask(mask: np.ndarray, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    assert mask.dtype == bool
    rng = rng or np.random.default_rng()

    true_idx = np.flatnonzero(mask)

    if len(true_idx) <= n:
        return mask  # nothing to trim

    keep = rng.choice(true_idx, size=n, replace=False)
    new_mask = np.zeros_like(mask, dtype=bool)
    new_mask[keep] = True
    return new_mask
