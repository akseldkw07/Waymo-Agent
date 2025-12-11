import typing as t

import numpy as np
import pandas as pd

from waymo_agent.data_classes.active_rides import ActiveRideDF
from waymo_agent.osmnx.euclidean_L2_embed import f_edges_start_end_node

if t.TYPE_CHECKING:
    from waymo_agent.graph_env.mixin.graph_mixin import OSMnxWrapperMixin


def precompute_path_timesteps(env: "OSMnxWrapperMixin", route: list[int]):
    """
    Construct an enriched edge dataframe for a given route, matching the notebook logic.
    Args:
        env: OSMnxWrapperMixin
        route: list of node ids (ints)
    Returns:
        pd.DataFrame: DataFrame with columns:
          ["source","target","key","osmid","maxspeed","name","length","travel_time_minutes","CumTravelTime",
           "node_id","x","y","x_norm","y_norm"]
    """
    # Compute starts and ends of route edges
    starts = route[:-1]
    ends = route[1:]
    merged = f_edges_start_end_node(env.EdgeDFEnriched, pd.Series(starts), pd.Series(ends))
    # Compute cumulative travel time
    merged["CumTravelTime"] = merged["travel_time_minutes"].cumsum()

    dist_rem = merged["length"][::-1].cumsum()[::-1]
    dist_rem.iloc[-1] = 0.0  # ensure last entry is exactly 0
    merged["DistanceRemaining"] = dist_rem
    # Select columns in the specified order
    cols = [
        "source",
        "target",
        "osmid",
        "maxspeed",
        "name",
        "length",
        "travel_time_minutes",
        "CumTravelTime",
        "DistanceRemaining",
        "node_id_src",
        "x_src",
        "y_src",
        "x_norm_src",
        "y_norm_src",
        "node_id_tgt",
        "x_tgt",
        "y_tgt",
        "x_norm_tgt",
        "y_norm_tgt",
    ]
    result = merged[cols]
    env.route_edges_df[(route[0], route[-1])] = result
    return result


def step_along_route(env: "OSMnxWrapperMixin", pos: ActiveRideDF):
    """
    Move each entry in pos one step along its route.
    Each row in pos is advanced one environment time-step along its route.
    Returns a new PathPositionDF with updated positions.
    """
    pos_new = pos.copy(deep=True)

    for row in pos.itertuples():
        # 1 Find the enriched route edges for this row's route
        row_index = int(row.Index)  # type: ignore
        if not pos.f_valid[row_index]:
            continue
        try:
            route_nodes: list[int] = row.route_nodes  # type: ignore
            start: int = route_nodes[0]  # type: ignore
            end: int = route_nodes[-1]  # type: ignore
        except Exception:
            continue
        route_enriched = env.route_edges_df.get((start, end), precompute_path_timesteps(env, route_nodes))

        # 2 Interpolate time-step position on edge
        curr_start: int = row.curr_start_node  # type: ignore
        curr_end: int = row.curr_end_node  # type: ignore
        dist_on_edge: float = row.route_dist_on_edge  # type: ignore

        curr_time = edge_progress_to_time(route_enriched, curr_start, curr_end, dist=dist_on_edge)

        # 3 Advance by one time-step
        time_step_minutes = env.config.time_step_delta.seconds / 60.0
        next_time = curr_time + time_step_minutes

        # 4 Find new edge and dist_on_edge
        path_row, frac = time_to_edge_progress(route_enriched, next_time)

        # Update pos_new
        pos_new.at[row_index, "curr_start_node"] = path_row["source"]
        pos_new.at[row_index, "curr_end_node"] = path_row["target"]
        pos_new.at[row_index, "route_dist_on_edge"] = frac * path_row["length"]
        pos_new.at[row_index, "trip_distance_remaining_meters"] = path_row["DistanceRemaining"]
        pos_new.at[row_index, "is_complete"] = frac >= 1.0 and path_row["DistanceRemaining"] <= 0.0
    return pos_new


def time_to_edge_progress(route_df: pd.DataFrame, t: float):
    """
    Given a route dataframe and an elapsed time `t` (same units as
    `travel_time_minutes`), return:
        (source_node, target_node, fraction_on_edge)

    `fraction_on_edge` is in [0, 1], indicating how far along the current
    edge the traveler is.
    """
    if route_df.empty:
        raise ValueError("route_df is empty")

    cum = route_df["CumTravelTime"].to_numpy(dtype=float)
    seg = route_df["travel_time_minutes"].to_numpy(dtype=float)

    # Clamp time to [0, total_route_time]
    t_clamped = float(np.clip(t, 0.0, cum[-1]))

    # Find the first segment whose CumTravelTime >= t_clamped
    idx = int(np.searchsorted(cum, t_clamped, side="right"))
    if idx >= len(cum):
        idx = len(cum) - 1

    prev_cum = 0.0 if idx == 0 else cum[idx - 1]
    time_on_edge = t_clamped - prev_cum
    edge_time = seg[idx]

    if edge_time <= 0:
        frac = 1.0
    else:
        frac = time_on_edge / edge_time
    frac = float(np.clip(frac, 0.0, 1.0))

    row = route_df.iloc[idx]

    return row, frac


def edge_progress_to_time(
    route_df: pd.DataFrame, start_node: int, end_node: int, frac: float | None = None, dist: float | None = None
) -> float:
    """
    Inverse of `time_to_edge_progress`.

    Given:
        - start_node, end_node: the edge being traversed
        - frac: fraction along that edge in [0, 1]
        - dist: distance along that edge (optional, overrides frac if provided)

    Return:
        absolute time along the route (same units as `travel_time_minutes`).
    """
    if route_df.empty:
        raise ValueError("route_df is empty")

    mask = (route_df["source"] == start_node) & (route_df["target"] == end_node)
    if not mask.any():
        raise KeyError(f"Edge ({start_node} -> {end_node}) not found in route_df")

    idx = int(route_df.index[mask][0])

    cum = route_df["CumTravelTime"].to_numpy(dtype=float)
    seg = route_df["travel_time_minutes"].to_numpy(dtype=float)

    prev_cum = 0.0 if idx == 0 else cum[idx - 1]
    edge_time = seg[idx]

    frac = frac or (dist / route_df.iloc[idx]["length"] if dist is not None else None)
    if frac is None:
        raise ValueError("Either frac or dist must be provided")

    frac_clamped = float(np.clip(frac, 0.0, 1.0))
    return float(prev_cum + frac_clamped * edge_time)
