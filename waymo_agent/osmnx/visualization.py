from matplotlib.axes import Axes
import matplotlib.cm as cm
import matplotlib.colors as colors
import networkx as nx
import numpy as np

from waymo_agent.data_classes import PlotConfig
from waymo_agent.data_classes.vehicles import VehicleDF
from waymo_agent.osmnx.euclidean_L2_embed import x_y_normed_to_orig


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


def get_node_colors_and_sizes(G: nx.MultiDiGraph, plt_cfg: PlotConfig | None = None):
    """
    Get node colors and sizes based on charger status and lambda values.

    Returns:
        node_colors (list): List of colors for each node.
        node_sizes (list): List of sizes for each node.
    """
    plt_cfg = plt_cfg or PlotConfig()
    node_width_scale = plt_cfg.ox_plot_requests["route_linewidth"]
    node_size = plt_cfg.ox_plot_default["node_size"]
    node_size = node_size if isinstance(node_size, (int, float)) else 5.0
    # --- node color: charger vs non-charger ---
    # e.g. bright cyan for chargers, dim gray for others
    node_colors = ["#88fe01" if G.nodes[n].get("num_chargers", False) else plt_cfg.node_color for n in G.nodes]

    # --- node size: scale with lambda ---
    lambdas = np.array([G.nodes[n].get("lambda", 0.0) for n in G.nodes], dtype=float)

    if lambdas.max() > 0:
        # rescale to something visible, say [10, 80] points
        lam_norm: np.ndarray = (lambdas - lambdas.min()) / (lambdas.max() - lambdas.min() + 1e-9)
        node_sizes: list[float] = (0.5 + node_width_scale * 10 * lam_norm).round(2).tolist()
    else:
        node_sizes: list[float] = np.full_like(lambdas, node_size).tolist()
    return node_colors, node_sizes


def plot_cars(veh: VehicleDF, cfg: PlotConfig, l2_recovery: dict[str, float], ax: Axes):
    colors = veh["status"].map(cfg.car_status_colors).to_list()

    lon, lat = x_y_normed_to_orig(l2_recovery, veh["loc_x_norm"], veh["loc_y_norm"])
    ax.scatter(lon, lat, c=colors, s=100, zorder=10, marker="X", edgecolors="black", linewidths=2)


def x_y_coords(
    ax: Axes,
    l2_recovery: dict[str, float],
    current_step: int,
    active_rides_routes: np.ndarray,
    request_routes: np.ndarray,
    veh: VehicleDF,
):
    # Show both raw and normalized coordinates on axes:
    # bottom x-axis: lon with normalized x in parentheses
    # left y-axis: lat with normalized y in parentheses
    l2_recovery["x0"]
    l2_recovery["y0"]
    l2_recovery["max_abs"]

    # Add a title with coordinate system info
    ax.set_title(
        f"Step {current_step} | "
        f"Vehicles: {len(veh)} | "
        f"Active Rides: {len(active_rides_routes)} | "
        f"Pending Requests: {len(request_routes)}\n"
        f"Coordinates: Raw (Normalized)",
        fontsize=12,
        fontweight="bold",
        color="black",
        pad=20,
        bbox=dict(boxstyle="round,pad=0.8", facecolor="white", edgecolor="black", linewidth=2.5),
    )
