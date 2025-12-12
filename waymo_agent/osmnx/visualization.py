import typing as t

import matplotlib.cm as cm
import matplotlib.colors as colors
import networkx as nx
import numpy as np
import osmnx as ox
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator

from waymo_agent.data_classes.active_rides import ActiveRideDF
from waymo_agent.data_classes.config_plot import PlotConfig
from waymo_agent.data_classes.requests import RequestDF
from waymo_agent.data_classes.vehicles import VehicleDF
from waymo_agent.osmnx.euclidean_L2_embed import x_y_normed_to_orig


def get_edge_color_by_speed(G: nx.MultiDiGraph, plt_cfg: PlotConfig | None = None):
    plt_cfg = plt_cfg or PlotConfig()
    speeds = []
    for _, _, data in G.edges(data=True):
        maxspeed = data.get("maxspeed", 0)
        assert isinstance(maxspeed, (int, float)), f"Unexpected maxspeed type: {type(maxspeed)}"
        speeds.append(maxspeed)

    speeds = np.array(speeds, dtype=float)
    norm = colors.Normalize(vmin=speeds.min(), vmax=speeds.max())
    cmap = cm.get_cmap(plt_cfg.edge_cmap)

    edge_colors: list[tuple[float, float, float, float]] = [
        tuple(np.array(cmap(norm(s).item())).tolist()) for s in speeds
    ]

    # max_width, min_width = 3.0, 0.5
    min_width, max_width = plt_cfg.unused_route_widths
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
    node_width_scale = plt_cfg.unused_route_widths[1]
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


@t.overload
def plot_routes(
    graph: nx.MultiDiGraph,
    routes: ActiveRideDF | RequestDF,
    plt_cfg: PlotConfig | None,
    fig_ax: tuple[Figure, Axes],
) -> tuple[Figure, Axes]: ...


@t.overload
def plot_routes(
    graph: nx.MultiDiGraph,
    routes: ActiveRideDF | RequestDF,
    plt_cfg: PlotConfig | None = None,
    fig_ax: None = None,
): ...


def plot_routes(
    graph: nx.MultiDiGraph,
    routes: ActiveRideDF | RequestDF,
    plt_cfg: PlotConfig | None = None,
    fig_ax: tuple[Figure, Axes] | None = None,
):
    """
    Plot routes (either active rides or requests) on the graph.
    # TODO add direction arrows, make narrower
    """
    if len(routes) == 0:
        return fig_ax
    assert isinstance(
        routes, (ActiveRideDF, RequestDF)
    ), f"routes must be ActiveRideDF or RequestDF, got {type(routes)}"
    plt_cfg = plt_cfg or PlotConfig()
    if fig_ax is None:
        fig, ax = ox.plot_graph(graph, **plt_cfg.ox_plot_default)
    else:
        fig, ax = fig_ax
    content = "rides" if isinstance(routes, ActiveRideDF) else "requests"
    colors = (
        plt_cfg.active_ride_color
        if content == "rides"
        else routes["status"].map(plt_cfg.request_status_colors).to_list()
    )
    linewidth = plt_cfg.active_ride_width if content == "rides" else plt_cfg.request_line_width

    keep_keys = ["show", "close", "bgcolor"]
    defaults: dict[str, t.Any] = {k: v for k, v in plt_cfg.ox_plot_default.items() if k in keep_keys}
    defaults["edge_alpha"] = plt_cfg.ride_edge_alpha

    fig, ax = ox.plot_graph_routes(
        graph, routes["route_nodes"].to_list(), route_colors=colors, route_linewidth=linewidth, ax=ax, **defaults
    )
    return fig, ax


def plot_cars(veh: VehicleDF, cfg: PlotConfig, l2_recovery: dict[str, float], ax: Axes):
    colors = veh["status"].map(cfg.car_status_colors).to_list()

    lon, lat = x_y_normed_to_orig(l2_recovery, veh["loc_x_norm"], veh["loc_y_norm"])
    ax.scatter(lon, lat, c=colors, s=100, zorder=10, marker="X", edgecolors="black", linewidths=2)


def set_xlabel_ylabel_title(
    ax: Axes, l2_recovery: dict[str, float], current_step: int, num_veh_req_rides: dict[str, int]
):
    """
    Format and display x and y coordinates on the plot axes, including normalized values.
    """
    # CRITICAL: osmnx turns axes off - turn them back on
    ax.axis("on")
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)

    # Extract params
    x0 = l2_recovery["x0"]
    y0 = l2_recovery["y0"]
    max_abs = l2_recovery["max_abs"]

    # Set tick formatters
    def fmt_x(val, pos):
        x_norm = (val - x0) / max_abs
        return f"{val:.3f}\n({x_norm:.2f})"

    def fmt_y(val, pos):
        y_norm = (val - y0) / max_abs
        return f"{val:.3f}\n({y_norm:.2f})"

    ax.xaxis.set_major_formatter(FuncFormatter(fmt_x))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_y))

    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))  # Increase from default ~5 to 6

    # Style ticks - increase pad for more space between labels and ticks
    ax.tick_params(axis="both", which="major", labelsize=10, colors="black", length=6, width=2, pad=12)

    # Force update then style labels
    ax.figure.canvas.draw_idle()
    for label in ax.get_xticklabels():
        label.set_bbox(dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3", linewidth=1.5))
    for label in ax.get_yticklabels():
        label.set_bbox(dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3", linewidth=1.5))

    # Axis labels - increase labelpad for more space
    ax.set_xlabel(
        "Longitude",
        fontsize=12,
        fontweight="bold",
        labelpad=15,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black", linewidth=2),
    )
    ax.set_ylabel(
        "Latitude",
        fontsize=12,
        fontweight="bold",
        labelpad=15,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black", linewidth=2),
    )

    # Title
    ax.set_title(
        f"Step {current_step} | Vehicles: {num_veh_req_rides['veh']} | Active Rides: {num_veh_req_rides['rides']} | "
        f"Pending Requests: {num_veh_req_rides['req']}\nCoordinates: Raw (Normalized)",
        fontsize=14,
        fontweight="bold",
        color="black",
        pad=20,
        bbox=dict(boxstyle="round,pad=0.8", facecolor="white", edgecolor="black", linewidth=2.5),
    )
