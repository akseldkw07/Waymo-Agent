from __future__ import annotations

import networkx as nx


def _coerce_speed(value: object) -> float | None:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if not cleaned:
            return None
        for suffix in ("mph", "km/h", "kph"):
            if suffix in cleaned:
                cleaned = cleaned.replace(suffix, "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def post_load(G: nx.MultiDiGraph) -> None:
    """
    Post-process the loaded graph by ensuring 'maxspeed' attributes are numeric.
    """
    for _, _, _, data in G.edges(keys=True, data=True):
        maxspeed = _coerce_speed(data.get("maxspeed"))
        if maxspeed is None:
            continue
        data["maxspeed"] = maxspeed * 1.60934  # Convert mph to kph (heuristic default)
        data["unit"] = "kph"
