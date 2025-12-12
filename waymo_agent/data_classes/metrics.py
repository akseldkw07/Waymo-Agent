"""
Formalize data classes for metrics
"""

import numpy as np
import pandas as pd


class TimeSeriesMetricsDF(pd.DataFrame):
    step: pd.Series
    timestamp: pd.Series
    error_msg: pd.Series

    rewards: pd.Series
    supply_demand_ratio: pd.Series  # NOTE, this is normed between 0 and 1

    # perc_cars_idle: pd.Series
    # perc_cars_on_ride: pd.Series
    # # perc_cars_charging: pd.Series # not currently applicable

    # avg_wait_time_seconds: pd.Series

    # num_new_requests: pd.Series

    # num_pending_requests: pd.Series
    # num_"is_complete"d_requests: pd.Series
    # num_cancelled_requests: pd.Series
    # num_rejected_requests: pd.Series
    # num_overflow_requests: pd.Series

    @classmethod
    def from_breadcrumbs(cls, breadcrumbs: list[dict]):
        ret = {key: np.array([bc[key] for bc in breadcrumbs]) for key in cls.__annotations__.keys()}
        return cls(ret)

    @classmethod
    def generate_empty(cls):
        ret = {key: np.array([]) for key, val in cls.__annotations__.items()}
        ret["timestamp"] = np.array([], dtype="datetime64[ns]")
        return cls(ret)
