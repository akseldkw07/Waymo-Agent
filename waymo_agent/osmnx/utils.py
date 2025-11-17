import typing as t

import networkx as nx


def _max_speed_int(speed: list[str] | str | int | float) -> int:
    """Helper to convert maxspeed attribute to integer in mph."""
    if isinstance(speed, list):
        speed = speed[0]  # Take the first value if it's a list
    if isinstance(speed, str):
        speed = speed.lower().strip()
        if "mph" in speed:
            speed = speed.replace("mph", "").strip()
        elif "km/h" in speed:
            kmh_value = float(speed.replace("km/h", "").strip())
            return int(kmh_value * 0.621371)  # Convert km/h to mph

        return int(float(speed))
    elif isinstance(speed, (int, float)):
        return int(speed)


def _collect_unique_graph_attr(G: nx.MultiDiGraph, attr: str) -> set[t.Any]:
    """Helper to collect all unique values of a given edge attribute in the graph."""
    values = set()
    for u, v, k, data in G.edges(keys=True, data=True):
        value = data.get(attr)
        if value is not None:
            if isinstance(value, list):
                values.update(value)
            else:
                values.add(value)
    return values
