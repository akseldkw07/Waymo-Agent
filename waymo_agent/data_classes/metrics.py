"""
Formalize data classes for metrics
"""

import numpy as np
import pandas as pd


class TimeSeriesMetricsDF(pd.DataFrame):
    step: np.ndarray
    timestamp: np.ndarray
    error_msg: np.ndarray

    rewards: np.ndarray
    supply_demand_ratio: np.ndarray

    # perc_cars_idle: np.ndarray
    # perc_cars_on_ride: np.ndarray
    # # perc_cars_charging: np.ndarray # not currently applicable

    # avg_wait_time_seconds: np.ndarray

    # num_new_requests: np.ndarray

    # num_pending_requests: np.ndarray
    # num_completed_requests: np.ndarray
    # num_cancelled_requests: np.ndarray
    # num_rejected_requests: np.ndarray
    # num_overflow_requests: np.ndarray

    @classmethod
    def from_breadcrumbs(cls, breadcrumbs: list[dict]):
        ret = {key: np.array([bc[key] for bc in breadcrumbs]) for key in cls.__annotations__.keys()}
        return cls(ret)

    @classmethod
    def generate_empty(cls):
        ret = {key: np.array([]) for key, val in cls.__annotations__.items()}
        ret["timestamp"] = np.array([], dtype="datetime64[ns]")
        return cls(ret)
