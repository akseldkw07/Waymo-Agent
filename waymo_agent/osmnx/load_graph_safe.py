"""
When loading graphs from OSM data, some edge attributes like 'maxspeed' may be stored as strings or lists.
This module provides a post-processing function to ensure these attributes are converted to numeric types for consistent usage.
"""

from __future__ import annotations
import ast
import contextlib
import pathlib

import networkx as nx
import osmnx as ox
from waymo_agent.osmnx.osmnx_constants import EXPECTED_EDGE_ATTR_TYPES, EXPECTED_NODE_ATTR_TYPES
from waymo_agent.osmnx.utils import _coerce_speed, safe_literal_eval


def fix_speed(G: nx.MultiDiGraph) -> None:
    """
    Ensure all 'maxspeed' attributes in the graph are numeric (float).
    Non-numeric values are removed.
    """
    for _, _, _, data in G.edges(keys=True, data=True):
        maxspeed = _coerce_speed(data.get("maxspeed"))
        if maxspeed is None:
            if "maxspeed" in data:
                raise ValueError(f"Could not coerce maxspeed value: {data['maxspeed']}")
                del data["maxspeed"]
        else:
            data["maxspeed"] = maxspeed


def coerce_node_attr_types(G: nx.MultiDiGraph) -> None:
    """
    NOTE not used currently. This is handled during graph loading via node_dtypes.
    """
    raise NotImplementedError("coerce_node_attr_types is not used currently.")
    for _, data in G.nodes(data=True):
        for attr, value in data.items():
            expected_type: type = EXPECTED_NODE_ATTR_TYPES[attr]
            value = safe_literal_eval(value)
            if isinstance(value, expected_type):
                continue
            else:
                data[attr] = expected_type(value)


def coerce_edge(data: dict):
    for attr, value in data.items():
        expected_type: type | None = EXPECTED_EDGE_ATTR_TYPES.get(attr)
        # print(f"Coercing edge attribute {attr} with value '{value}' to type {expected_type}")
        if expected_type is None:
            continue
        with contextlib.suppress(SyntaxError, ValueError):
            value = ast.literal_eval(value)
        # print(f"  After literal_eval: value='{value}' of type {type(value)}")
        if isinstance(value, expected_type):
            data[attr] = value
        else:
            if isinstance(value, str) and expected_type in (float, int):
                # print(f"  Converting string '{value}' to {expected_type}")
                try:
                    data[attr] = expected_type(value)
                except ValueError as e:
                    print(f"Could not convert edge attribute {attr} value '{value}' to {expected_type}: {e}")
            elif isinstance(value, list) and expected_type in (float, int):
                fixed = [expected_type(v) for v in value if isinstance(v, (str, int, float))]
                data[attr] = fixed


def coerce_edge_attr_types(G: nx.MultiDiGraph) -> None:
    """
    Coerce specific edge attributes to their appropriate types.
    Currently focuses on 'maxspeed'.
    """
    for _, _, data in G.edges(data=True):
        coerce_edge(data)


def post_load(G: nx.MultiDiGraph) -> None:
    """
    Fix the graph after loading from OSM data.
    """
    fix_speed(G)
    # coerce_node_attr_types(G)
    coerce_edge_attr_types(G)


def load_graph_type_preserved(filepath: str | pathlib.Path) -> nx.MultiDiGraph:
    """
    Load a graph from a GraphML file while preserving attribute types.

    Args:
        filepath (str): Path to the GraphML file.
    """
    G = ox.load_graphml(filepath, node_dtypes=EXPECTED_NODE_ATTR_TYPES)
    return G
