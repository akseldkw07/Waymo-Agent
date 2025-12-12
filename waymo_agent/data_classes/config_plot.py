from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

from waymo_agent.data_classes.osmnx_constants import DEFAULT_OX_PLOT_NOTEBOOK, Plot_graph_TypedDict

from .requests import RequestStatusEnum as RSE
from .vehicles import VehicleStatusEnum as VSE

if t.TYPE_CHECKING:
    from .osmnx_constants import DEFAULT_OX_PLOT_NOTEBOOK, Plot_graph_TypedDict


@dataclass
class PlotConfig:
    node_color: str = "black"
    unused_route_widths: tuple[float, float] = (0.5, 3)  # min, max

    active_ride_color: str = "limegreen"
    active_ride_width: float = 2.5
    ride_edge_alpha: float = 1.0

    edge_cmap: str = "Wistia"  # https://matplotlib.org/stable/users/explain/colors/colormaps.html

    ox_plot_default: Plot_graph_TypedDict = field(default_factory=lambda: DEFAULT_OX_PLOT_NOTEBOOK)

    car_status_colors: dict[str | float, str] = field(
        default_factory=lambda: {
            VSE.IDLE.name: "gray",
            VSE.IDLE: "gray",
            VSE.TO_PICKUP.name: "lightskyblue",
            VSE.TO_PICKUP: "lightskyblue",
            VSE.WITH_PASSENGER.name: "limegreen",
            VSE.WITH_PASSENGER: "limegreen",
            VSE.CHARGING.name: "red",
            VSE.CHARGING: "red",
        }
    )

    request_line_width: float = 2.0
    request_status_colors: dict[str | int, str] = field(
        default_factory=lambda: {
            RSE.AWAITING_PRICE.name: "gray",
            RSE.AWAITING_PRICE: "gray",
            RSE.ACCEPTED.name: "lightpink",
            RSE.ACCEPTED: "lightpink",
            RSE.ASSIGNED.name: "lightskyblue",
            RSE.ASSIGNED: "lightskyblue",
        }
    )
