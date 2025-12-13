import typing as t
from math import ceil

import numpy as np
import pandas as pd

from waymo_agent.data_classes import EnvConfig, RequestDF, VehicleDF
from waymo_agent.data_classes.enriched_df_base import EnrichedDF

if t.TYPE_CHECKING:
    pass


class SupplyDemandDF(EnrichedDF):
    sd_current: pd.Series
    vehicle_per_node: pd.Series
    lambda_per_node: pd.Series

    cols_to_pass_to_model: t.ClassVar[list[str]] = ["sd_current", "vehicle_per_node", "lambda_per_node"]
    target_dtypes = {
        "sd_current": np.float32,
        "vehicle_per_node": np.float32,
        "lambda_per_node": np.float32,
    }

    @classmethod
    def space_config(cls, config: EnvConfig | None = None, *args, **kwargs):
        length = cls.calc_width()
        shape = (length,)

        cfg = config or EnvConfig()
        high_val = ceil(max(cfg.vehicle_per_node, cfg.lambda_per_node) * 10)

        low = np.zeros(shape, dtype=np.float32)
        high = np.full(shape, high_val, dtype=np.float32)

        return {"shape": shape, "low": low, "high": high}

    @classmethod
    def get_sd_ratio(cls, config: EnvConfig, requests: RequestDF, vehicles: VehicleDF):

        z = (vehicles.f_idle.sum() / (requests.f_awaiting_price.sum() + 0.01)) - 1.0
        sd_current = 1 / (1 + np.exp(-z))  # sigmoid function

        df = pd.DataFrame(
            {
                "sd_current": [sd_current],
                "vehicle_per_node": [config.vehicle_per_node],
                "lambda_per_node": [config.lambda_per_node],
            }
        )
        return cls(df)

    @classmethod
    def from_obs_numpy(cls, obs_array: np.ndarray | pd.DataFrame):
        """
        Might come in as (size, ) or (val, size)
        """
        if isinstance(obs_array, np.ndarray) and obs_array.ndim == 1:
            obs_array = obs_array.reshape(1, -1)
            ret = pd.DataFrame(obs_array, columns=cls.cols_to_pass_to_model)
        return cls(pd.DataFrame(obs_array, columns=cls.cols_to_pass_to_model))
