from __future__ import annotations

import typing as t
from pathlib import Path

import osmnx as ox

from waymo_agent.utils.visualization_utils import get_edge_color_by_speed, get_node_colors_and_sizes

from ...osmnx.osmnx_constants import Plot_graph_TypedDict
from .interface import GraphMixinInterface


class RideShareRenderingMixin(GraphMixinInterface):
    """Matplotlib-based rendering that overlays fleet state on top of the OSMnx map."""

    def render(self, **kwargs: t.Unpack[Plot_graph_TypedDict]):
        mode = self.render_mode or "human"
        if mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render mode: {mode}")
        if mode == "ansi":
            raise NotImplementedError("ANSI rendering is not implemented yet.")
        return self._render_map(**kwargs)

    def _render_map(self, **kwargs: t.Unpack[Plot_graph_TypedDict]):
        """
        Render the current environment state on the map.

        TODO render
            - vehicle positions
            - current requests
            - active rides (nodes/edges) use ox.plot_graph_routes()
        """
        self.logger.warning("Rendering incomplete - this is a placeholder implementation.")
        edge_colors, edge_widths = self._edge_styles()
        node_colors, node_sizes = self._node_styles()

        ox_kwargs: Plot_graph_TypedDict = (
            self.config.ox_plot_default
            | kwargs
            | Plot_graph_TypedDict(
                {
                    "edge_color": edge_colors,  # type: ignore
                    "edge_linewidth": edge_widths,
                    "node_color": node_colors,
                    "node_size": node_sizes,
                }
            )
        )

        ox.plot_graph(self.graph, **ox_kwargs)

    def save_render(self, path: str | Path):
        """
        Save the current render to a file.
        """
        ...

    def close(self):
        """Close the rendering."""

    # ------------------------------------------------------------------ #
    # Style helpers
    # ------------------------------------------------------------------ #
    def _edge_styles(self):
        edge_colors, edge_widths = get_edge_color_by_speed(self.graph)
        # TODO overwrite based on active and requested rides
        # Use nx library
        return edge_colors, edge_widths

    def _node_styles(self):
        node_colors, node_sizes = get_node_colors_and_sizes(self.graph)
        # TODO overwrite based on 1) vehicle positions 2) charger availability 3) pending request 4) active rides
        return node_colors, node_sizes
