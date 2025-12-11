from __future__ import annotations

import typing as t
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from waymo_agent.data_classes import PlotConfig, RequestStatusEnum as RSE
from waymo_agent.osmnx.visualization import get_edge_color_by_speed, get_node_colors_and_sizes, plot_cars, x_y_coords

from ...osmnx.osmnx_constants import Plot_graph_TypedDict
from .interface import GymEnvInterface


class RenderingMixin(GymEnvInterface):
    """
    Mixin for rendering the environment using Matplotlib and OSMnx.

    This mixin provides functionality to:
    - Render the road network graph.
    - Overlay vehicle positions and active ride routes.
    - Save rendered frames to disk.
    """

    @t.overload
    def render(self, *args: t.Any, **kwargs: t.Any) -> tuple[Figure, Axes]: ...
    @t.overload
    def render(self, plt_cfg: PlotConfig | None = None) -> tuple[Figure, Axes]: ...
    def render(self, plt_cfg: PlotConfig | None = None) -> tuple[Figure, Axes]:
        """
        Render the environment.

        Args:
            **kwargs: Additional arguments passed to the rendering backend.

        Returns:
            The figure and axes objects of the plot.
        """
        mode = self.render_mode or "human"
        if mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render mode: {mode}")
        if mode == "ansi":
            raise NotImplementedError("ANSI rendering is not implemented yet.")
        fig, ax = self._render_map(plt_cfg=plt_cfg)
        return fig, ax

    def _render_map(self, plt_cfg: PlotConfig | None = None):
        """
        Render the current environment state on the map.
        """
        edge_colors, edge_widths = self._edge_styles()
        node_colors, node_sizes = self._node_styles()
        plt_cfg = plt_cfg or self.plt_cfg

        ox_kwargs: Plot_graph_TypedDict = (
            self.plt_cfg.ox_plot_default
            | plt_cfg.ox_plot_default
            | Plot_graph_TypedDict(
                {
                    "edge_color": edge_colors,  # type: ignore
                    "edge_linewidth": edge_widths,
                    "node_color": node_colors,
                    "node_size": node_sizes,
                    "bgcolor": "silver",
                }
            )
        )

        # Plot the base graph
        fig, ax = ox.plot_graph(self.graph, **ox_kwargs)

        # Plot active rides & requests as routes
        active_rides = self.observation_curr["active_rides"]
        f_plot_rides = ~active_rides.complete & (active_rides.ride_id != self.config.invalid_id) & active_rides.f_valid
        self.f_plot_rides = f_plot_rides
        active_rides_routes = active_rides.route_nodes[f_plot_rides]

        plot_request_types = np.array([RSE.AWAITING_PRICE, RSE.ACCEPTED, RSE.ASSIGNED])
        pending_requests = self.observation_curr["pending_requests"]
        f_pending_requests = np.isin(pending_requests.status, plot_request_types)
        request_routes = pending_requests.route_nodes[f_pending_requests]

        # TODO add direction arrows, make narrower
        if len(active_rides_routes) > 0:
            fig, ax = ox.plot_graph_routes(
                self.graph, active_rides_routes, **self.plt_cfg.ox_plot_active_rides, ax=ax, show=False
            )
        if len(request_routes) > 0:
            fig, ax = ox.plot_graph_routes(
                self.graph, request_routes, **self.plt_cfg.ox_plot_requests, ax=ax, show=False
            )
        veh = self.observation_curr["vehicles"]
        plot_cars(veh, self.plt_cfg, self.l2_recovery, ax)
        x_y_coords(ax, self.l2_recovery, self.current_step, active_rides_routes, request_routes, veh)

        print(
            f"Rendered step {self.current_step} with {len(active_rides_routes)} active rides, {len(request_routes)} pending requests, and {len(self.observation_curr['vehicles'])} vehicles."
        )

        return fig, ax

    def save_render(self, path: str | Path):
        """
        Save the current render to a file.
        """
        fig, ax = self.render(show=False)
        fig.savefig(path, dpi=self.plt_cfg.ox_plot_default.get("dpi", 2_000))
        plt.close(fig)

    def close(self):
        """Close the rendering."""
        plt.close("all")

    # ------------------------------------------------------------------ #
    # Style helpers
    # ------------------------------------------------------------------ #
    def _edge_styles(self):
        edge_colors, edge_widths = get_edge_color_by_speed(self.graph)
        return edge_colors, edge_widths

    def _node_styles(self):
        node_colors, node_sizes = get_node_colors_and_sizes(self.graph)
        # TODO: Update node colors based on vehicle positions?
        # For now, keep static styles or implement dynamic coloring here
        return node_colors, node_sizes
