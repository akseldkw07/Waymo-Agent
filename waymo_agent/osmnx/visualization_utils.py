import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np


def get_edge_color_by_speed(G: nx.MultiDiGraph):
    speeds = []
    for _, _, data in G.edges(data=True):
        maxspeed = data.get("maxspeed", 0)
        assert isinstance(maxspeed, (int, float)), f"Unexpected maxspeed type: {type(maxspeed)}"
        speeds.append(maxspeed)

    speeds = np.array(speeds, dtype=float)
    norm = colors.Normalize(vmin=speeds.min(), vmax=speeds.max())
    cmap = cm.get_cmap("inferno")

    edge_colors = [cmap(norm(s)) for s in speeds]

    max_width, min_width = 3.0, 0.5
    edge_widths = min_width + (norm(speeds) * (max_width - min_width))
    return edge_colors, edge_widths


def get_node_colors_and_sizes(G: nx.MultiDiGraph):
    """
    Get node colors and sizes based on charger status and lambda values.

    Returns:
        node_colors (list): List of colors for each node.
        node_sizes (list): List of sizes for each node.
    """
    # --- node color: charger vs non-charger ---
    # e.g. bright cyan for chargers, dim gray for others
    node_colors = ["#88fe01" if G.nodes[n].get("num_chargers", False) else "#ffffff" for n in G.nodes]

    # --- node size: scale with lambda ---
    lambdas = np.array([G.nodes[n].get("lambda", 0.0) for n in G.nodes], dtype=float)
    if lambdas.max() > 0:
        # rescale to something visible, say [10, 80] points
        lam_norm: np.ndarray = (lambdas - lambdas.min()) / (lambdas.max() - lambdas.min() + 1e-9)
        node_sizes = (0.5 + 15 * 5 * lam_norm).round(2).tolist()
    else:
        node_sizes = np.full_like(lambdas, 5).tolist()
    return node_colors, node_sizes
