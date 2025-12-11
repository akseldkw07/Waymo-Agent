from __future__ import annotations

import enum
import typing as t

import numpy as np
import pandas as pd


class WeatherEnum(enum.IntEnum):
    CLEAR = 0
    RAINY = 1
    SNOWY = 2


class PathPositionDF(pd.DataFrame):
    route_nodes: np.ndarray  # list[int] per row
    curr_start_node: np.ndarray  # current start node in route_nodes
    curr_end_node: np.ndarray  # current end node in route_nodes
    route_dist_on_edge: np.ndarray  # how much distance has been traveled on the CURRENT edge


class EnvInfoTypedDict(t.TypedDict):
    step: dict
    metrics: dict[str, float]
    map_name: str
    error: str
