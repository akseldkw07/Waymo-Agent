import typing as t
from collections.abc import Iterable, Sequence
from pathlib import Path

import networkx as nx
from matplotlib.axes import Axes

EXPECTED_NODE_ATTR_TYPES = {
    "y": float,
    "x": float,
    "highway": str,
    "street_count": int,
    "lambda": float,
    "x_centered": float,
    "y_centered": float,
    "x_norm": float,
    "y_norm": float,
    "ref": str,
}

EXPECTED_EDGE_ATTR_TYPES = {
    # "osmid": int,  # sometimes list[int] → treat as int or list[int]
    "highway": str,
    "maxspeed": float,
    # "name": str,
    "oneway": bool,
    # "reversed": bool,
    "length": float,
    # "geometry": LineString,  # Shapely LineString
    "lanes": int,  # list or str → treat as int
    "unit": str,
    "travel_time_minutes": float,
    "bridge": str,
    # "ref": str,
    # "tunnel": str,
    "width": str,
    "access": str,
    "junction": str,
}


class Plot_route_TypedDict(t.TypedDict, total=True):
    route_color: str
    route_linewidth: float


class Plot_graph_TypedDict(t.TypedDict, total=False):
    # G: nx.MultiGraph | nx.MultiDiGraph
    ax: Axes | None
    figsize: tuple[float, float]
    bgcolor: str
    node_color: str | Sequence[str]
    node_size: t.Required[float | Sequence[float]]
    node_alpha: float | None
    node_edgecolor: str | Iterable[str]
    node_zorder: int
    edge_color: str | Iterable[str]
    edge_linewidth: float | Sequence[float]
    edge_alpha: float | None
    bbox: tuple[float, float, float, float] | None
    show: bool
    close: bool
    save: bool
    filepath: str | Path | None
    dpi: int


DEFAULT_OX_PLOT_NOTEBOOK: Plot_graph_TypedDict = {
    "node_size": 5,
    "node_color": "white",
    "node_alpha": 0.5,
    "figsize": (10, 10),
    "bgcolor": "black",
    "show": True,
}


class OSMNXConstants:
    """
    Constants related to OSMnx road types and their configurations.
    """

    # OSMnx
    ROAD_TYPE_LITERAL = t.Literal[
        "motorway",
        "motorway_link",
        "trunk",
        "trunk_link",
        "primary",
        "primary_link",
        "secondary",
        "secondary_link",
        "tertiary",
        "tertiary_link",
        "residential",
        "unclassified",
        "service",
        "living_street",
        "road",
        "pedestrian",
        "track",
        "bus_guideway",
        "footway",
        "cycleway",
        "path",
    ]
    ROAD_PRIORITY: list[ROAD_TYPE_LITERAL] = [
        # ----- Major Highways -----
        "motorway",
        "motorway_link",
        "trunk",
        "trunk_link",
        # ----- Major City Roads -----
        "primary",
        "primary_link",
        "secondary",
        "secondary_link",
        "tertiary",
        "tertiary_link",
        # ----- Local & Minor Roads -----
        "unclassified",  # (Roads that don't fit other categories)
        "residential",
        "living_street",  # (Residential street where pedestrians have priority)
        # ----- Special Use (Non-driving) -----
        "service",  # (Service roads, alleys, parking lot lanes)
        "pedestrian",
        "track",
        "bus_guideway",
        "footway",
        "cycleway",
        "path",
    ]

    NODE_ATTR_LITERAL = t.Literal[
        "y", "x", "highway", "street_count", "lambda", "x_centered", "y_centered", "x_norm", "y_norm", "ref"
    ]
    EDGE_ATTR_LITERAL = t.Literal[
        "access",
        "bridge",
        "geometry",
        "highway",
        "junction",
        "lanes",
        "length",
        "maxspeed",
        "name",
        "oneway",
        "osmid",
        "ref",
        "reversed",
        "travel_time_minutes",
        "tunnel",
        "unit",
        "width",
    ]

    @classmethod
    def MAJOR_ROAD_TYPES_SET(cls):
        cutoff = cls.ROAD_PRIORITY.index("secondary") + 1
        return set(cls.ROAD_PRIORITY[:cutoff])

    @classmethod
    def HIGHWAY_TYPES_SET(cls):
        cutoff = cls.ROAD_PRIORITY.index("trunk_link") + 1
        return set(cls.ROAD_PRIORITY[:cutoff])

    COLOR_CONFIG = {
        "motorway": {"color": "#ff0000", "width": 3.0},  # Red, Thickest
        "motorway_link": {"color": "#ff0000", "width": 3.0},
        "trunk": {"color": "#ff6600", "width": 2.5},  # Orange, Thick
        "trunk_link": {"color": "#ff6600", "width": 2.5},
        "primary": {"color": "#ffac12", "width": 2.0},  # Gold, Medium-Thick
        "primary_link": {"color": "#ffac12", "width": 2.0},
        "secondary": {"color": "#f4f458", "width": 1.0},  # Yellow, Medium
        "secondary_link": {"color": "#f4f458", "width": 1.0},
        "tertiary": {"color": "#ffffff", "width": 0.8},  # White, Thin
        "tertiary_link": {"color": "#ffffff", "width": 0.8},
        "residential": {"color": "#555555", "width": 0.5},  # Grey, Thinnest
        "unclassified": {"color": "#555555", "width": 0.5},
        "service": {"color": "#777777", "width": 0.4},  # Light Grey, Thinnest
        "living_street": {"color": "#777777", "width": 0.4},
        "pedestrian": {"color": "#999999", "width": 0.3},  # Lighter Grey, Thinnest
    }

    @staticmethod
    def get_edge_colors(G: nx.MultiDiGraph):
        edge_colors = []
        edge_widths = []

        for u, v, data in G.edges(data=True):
            # Get highway type (handle lists if necessary)
            highway_type = data.get("highway", "default")
            if isinstance(highway_type, list):
                highway_type = highway_type[0]

            # Fetch style from config, or use default if type not found
            style = OSMNXConstants.COLOR_CONFIG.get(highway_type, OSMNXConstants.COLOR_CONFIG["unclassified"])

            edge_colors.append(style["color"])
            edge_widths.append(style["width"])

        return edge_colors, edge_widths

    NORTH_125TH = 40.820
    SOUTH_LIMIT = 40.69
    WEST_LIMIT = -74.03
    EAST_LIMIT = -73.927  # cut before Randalls (≈ -73.92)
    MANHATTAN_BOX = (WEST_LIMIT, SOUTH_LIMIT, EAST_LIMIT, NORTH_125TH)
