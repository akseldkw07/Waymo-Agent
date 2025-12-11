import networkx as nx
import numpy as np
import osmnx as ox
from scipy.spatial.kdtree import cKDTree

PRECISION = 4
import typing as t

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

if t.TYPE_CHECKING:
    from waymo_agent.graph_env.mixin.graph_mixin import OSMnxWrapperMixin


def embed_L2(G: nx.MultiDiGraph):
    """
    Embed L2 normalized coordinates into each node.
    """
    G_proj = ox.project_graph(G)

    xs = np.array([d["x"] for _, d in G_proj.nodes(data=True)])
    ys = np.array([d["y"] for _, d in G_proj.nodes(data=True)])

    x0 = xs.mean()
    y0 = ys.mean()
    xs_c = xs - x0
    ys_c = ys - y0
    max_abs = max(np.abs(xs_c).max(), np.abs(ys_c).max())  # largest radius

    for n, d in G_proj.nodes(data=True):
        d["x_centered"] = float(d["x"] - x0)
        d["y_centered"] = float(d["y"] - y0)
        d["x_norm"] = float((d["x"] - x0) / max_abs)
        d["y_norm"] = float((d["y"] - y0) / max_abs)

    for n in G.nodes:
        G.nodes[n]["x_centered"] = round(G_proj.nodes[n]["x_centered"], PRECISION)
        G.nodes[n]["y_centered"] = round(G_proj.nodes[n]["y_centered"], PRECISION)
        G.nodes[n]["x_norm"] = round(G_proj.nodes[n]["x_norm"], PRECISION)
        G.nodes[n]["y_norm"] = round(G_proj.nodes[n]["y_norm"], PRECISION)


def _nearest_node_from_xy(kd_tree: "cKDTree", x_norm: float, y_norm: float) -> int:
    """
    Return the *index* of the closest node in self.node_coords to (x_norm, y_norm).
    """
    dist, idx = kd_tree.query([x_norm, y_norm], k=1)
    return int(idx)  # idx is 0..num_nodes-1


def nearest_node_id_from_xy(kd_tree: "cKDTree", node_ids: list[int], x_norm: float, y_norm: float):
    idx = _nearest_node_from_xy(kd_tree, x_norm, y_norm)
    return node_ids[idx]


def interpolate_position_on_edge(
    env: "OSMnxWrapperMixin",
    starts: pd.Series | np.ndarray,
    ends: pd.Series | np.ndarray,
    dist_on_edge: pd.Series | np.ndarray,
):
    """
    1. Get edges
    2. For each edge, get start and end node coordinates
    3. Interpolate position based on dist_on_edge and edge length
    """
    edge_df = f_edges_start_end_node(env.EdgeDFEnriched, starts, ends)
    lengths = edge_df["length"].to_numpy()
    # avoid divide-by-zero; clip fraction to [0, 1]
    lengths_safe = np.maximum(lengths, 1e-6)
    perc_done = (np.asarray(dist_on_edge) / lengths_safe).clip(0.0, 1.0)

    ret_df = pd.DataFrame(index=edge_df.index)
    ret_df["perc_done"] = perc_done

    for val in ["x", "y", "x_norm", "y_norm"]:
        src = edge_df[f"{val}_src"].to_numpy()
        tgt = edge_df[f"{val}_tgt"].to_numpy()
        interpolated = src + perc_done * (tgt - src)
        ret_df[val] = interpolated

    return ret_df


def f_edges_start_end_node(
    edge_df: pd.DataFrame, starts: pd.Series | np.ndarray, ends: pd.Series | np.ndarray
) -> pd.DataFrame:
    """
    Given series of start and end node ids, return DataFrame of edges matching those start-end pairs.
    Args:
        edge_df: DataFrame of edges
        starts: pd.Series of start node ids
        ends: pd.Series of end node ids
    Returns:
        pd.DataFrame: DataFrame of edges matching the start-end pairs.
    """
    order_df = pd.DataFrame({"source": starts, "target": ends, "_order": np.arange(len(starts))})
    # Subset edge_df to just the path edges, with correct order
    # Merge to get only edges along this path, in order
    merged = pd.merge(order_df, edge_df, on=["source", "target"], how="left", sort=False, validate="1:m")
    # Sort by _order to preserve path order
    merged = merged.sort_values("_order").reset_index(drop=True)
    # Drop _order
    merged = merged.drop(columns=["_order"])
    return merged
