import logging

import networkx as nx
import typing as t

from .osmnx_constants import OSMNXConstants as C
from .utils import _max_speed_int

logger = logging.getLogger(__name__)


def preprocess_graph(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Preprocesses the graph by cleaning edge attributes.
        1. _max_speed: Convert 'maxspeed' attributes to integers.
            Ex: "25 mph" -> 25
        2. _clean_road_type: Ensure 'highway' attributes are valid ROAD_TYPE_LITERAL.
            If multiple road types are present, select the one with the highest priority.

    """
    G_cleaned = G.copy()
    _clean_max_speed(G_cleaned)
    _clean_road_type(G_cleaned)
    return G_cleaned


def _clean_max_speed(G: nx.MultiDiGraph) -> None:
    """
    Cleans the 'maxspeed' attribute of all edges in the
    graph by converting them to integers.

    NOTE speed is stored in mph after conversion
    """

    for u, v, k, data in G.edges(keys=True, data=True):
        maxspeed = data.get("maxspeed")

        if maxspeed is None:
            max_speed_mph = 30  # Default to 30 mph if not specified
        else:
            try:
                max_speed_mph = _max_speed_int(maxspeed)
            except Exception as e:
                raise ValueError(f"Error parsing maxspeed: {e}. {data=}")
        data["maxspeed"] = max_speed_mph


def _clean_road_type(G: nx.MultiDiGraph) -> None:
    """
    Cleans the 'highway' attribute of all edges in the graph
    by ensuring they are valid ROAD_TYPE_LITERAL.
    Selects the highest priority road type if multiple are present.
    """
    for u, v, k, data in G.edges(keys=True, data=True):
        road_type = data.get("highway")

        if isinstance(road_type, str) and road_type in C.ROAD_PRIORITY:
            data["highway"] = road_type
        elif isinstance(road_type, t.Iterable):  # check after str because str is Iterable
            # If it's a list, take the road type with the highest priority
            for rt in C.ROAD_PRIORITY:
                if rt in road_type:
                    data["highway"] = rt
                    break
        else:
            logger.warning(
                f"Invalid road type: {road_type}, associated with edge ({u}, {v}, {k}). Setting to 'unclassified'."
            )
            data["highway"] = "unclassified"  # Default if invalid
