import matplotlib.cm as cm
import matplotlib.colors as colors
import networkx as nx
import numpy as np

from waymo_agent.data_classes.dataclasses import EnvConfig


def get_edge_color_by_speed(G: nx.MultiDiGraph, cmap_name: str = "plasma"):
    speeds = []
    for _, _, data in G.edges(data=True):
        maxspeed = data.get("maxspeed", 0)
        assert isinstance(maxspeed, (int, float)), f"Unexpected maxspeed type: {type(maxspeed)}"
        speeds.append(maxspeed)

    speeds = np.array(speeds, dtype=float)
    norm = colors.Normalize(vmin=speeds.min(), vmax=speeds.max())
    cmap = cm.get_cmap(cmap_name)

    edge_colors: list[tuple[float, float, float, float]] = [
        tuple(np.array(cmap(norm(s).item())).tolist()) for s in speeds
    ]

    max_width, min_width = 3.0, 0.5
    edge_widths: np.ndarray = min_width + (norm(speeds) * (max_width - min_width))
    edge_widths_list: list[float] = edge_widths.round(2).tolist()

    return edge_colors, edge_widths_list


def get_node_colors_and_sizes(G: nx.MultiDiGraph, config: EnvConfig | None = None):
    """
    Get node colors and sizes based on charger status and lambda values.

    Returns:
        node_colors (list): List of colors for each node.
        node_sizes (list): List of sizes for each node.
    """
    config = config or EnvConfig()
    node_width_scale = config.ox_plot_requests["route_linewidth"]
    node_size = config.ox_plot_default["node_size"]
    node_size = node_size if isinstance(node_size, (int, float)) else 5.0
    # --- node color: charger vs non-charger ---
    # e.g. bright cyan for chargers, dim gray for others
    node_colors = ["#88fe01" if G.nodes[n].get("num_chargers", False) else "#ffffff" for n in G.nodes]

    # --- node size: scale with lambda ---
    lambdas = np.array([G.nodes[n].get("lambda", 0.0) for n in G.nodes], dtype=float)
    if lambdas.max() > 0:
        # rescale to something visible, say [10, 80] points
        lam_norm: np.ndarray = (lambdas - lambdas.min()) / (lambdas.max() - lambdas.min() + 1e-9)
        node_sizes: list[float] = (0.5 + node_width_scale * 10 * lam_norm).round(2).tolist()
    else:
        node_sizes: list[float] = np.full_like(lambdas, node_size).tolist()
    return node_colors, node_sizes
