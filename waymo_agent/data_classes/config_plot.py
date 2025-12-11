from __future__ import annotations

import typing as t
from dataclasses import dataclass, field


from .vehicles import VehicleStatusEnum

if t.TYPE_CHECKING:
    from ..osmnx.osmnx_constants import Plot_graph_TypedDict, Plot_route_TypedDict


@dataclass
class PlotConfig:
    node_color: str = "black"
    ox_plot_requests: Plot_route_TypedDict = field(
        default_factory=lambda: {"route_color": "cyan", "route_linewidth": 2.0}
    )
    ox_plot_active_rides: Plot_route_TypedDict = field(
        default_factory=lambda: {"route_color": "limegreen", "route_linewidth": 3.0}
    )

    ox_plot_default: Plot_graph_TypedDict = field(
        default_factory=lambda: {
            "node_size": 5,
            "node_color": "white",
            "node_alpha": 0.85,
            "figsize": (10, 10),
            "bgcolor": "silver",
            "show": False,
            "edge_linewidth": 1.0,
        }
    )

    car_status_colors: dict[str | int, str] = field(
        default_factory=lambda: {
            VehicleStatusEnum.IDLE.name: "blue",
            VehicleStatusEnum.IDLE: "blue",
            VehicleStatusEnum.TO_PICKUP.name: "orange",
            VehicleStatusEnum.TO_PICKUP: "orange",
            VehicleStatusEnum.WITH_PASSENGER.name: "green",
            VehicleStatusEnum.WITH_PASSENGER: "green",
            VehicleStatusEnum.CHARGING.name: "red",
            VehicleStatusEnum.CHARGING: "red",
        }
    )
