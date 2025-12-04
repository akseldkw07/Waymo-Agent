import networkx as nx
import numpy as np
import osmnx as ox
from scipy.spatial.kdtree import cKDTree

PRECISION = 4


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
