from dataclasses import dataclass
import numpy as np

from waymo_agent.graph_env.ENV import RideShareEnv


@dataclass
class EpisodeMetrics:
    total_return: float
    n_completed: int
    n_assigned: int
    n_accepted: int
    n_rejected: int
    mean_wait_s: float


def compute_episode_metrics(env: RideShareEnv) -> EpisodeMetrics:
    # assumes env exposes observation_curr with pending_requests/active_rides
    req = env.observation_curr["pending_requests"]
    env.observation_curr["active_rides"]

    status = req["status"].to_numpy()
    n_accepted = int((status == 1).sum())
    n_assigned = int((status == 3).sum())
    n_completed = int((status == 5).sum())
    n_rejected = int(((status == -1) | (status == -2)).sum())

    # wait_time is timedelta64
    wt = req["wait_time"].to_numpy()
    # safe conversion to seconds
    wt_s = (wt / np.timedelta64(1, "s")).astype(float)
    mean_wait_s = float(np.nanmean(wt_s)) if wt_s.size else 0.0

    return EpisodeMetrics(
        total_return=0.0,  # fill during rollout
        n_completed=n_completed,
        n_assigned=n_assigned,
        n_accepted=n_accepted,
        n_rejected=n_rejected,
        mean_wait_s=mean_wait_s,
    )
