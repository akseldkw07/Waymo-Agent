"""
TODO
"""

import pathlib

import numpy as np
import pandas as pd

from waymo_agent.constants import DATA_DIR
from waymo_agent.data_classes.config import EnvConfig

from . import RequestDF


class CustomerDFGenator:
    size: int = 50_000
    loc = 0
    scale = 2
    temp_low = 1.2
    temp_high = 2.5

    filename: str = "customers.parquet"

    @staticmethod
    def generate():
        CLS = CustomerDFGenator
        bias = np.random.normal(loc=CLS.loc, scale=CLS.scale, size=CLS.size)
        temperature = np.random.uniform(low=CLS.temp_low, high=CLS.temp_high, size=CLS.size)
        cust_df = CustomerDF({"cust_id": np.arange(CLS.size), "bias": bias, "temperature": temperature})
        return cust_df

    @staticmethod
    def save(path: pathlib.Path = DATA_DIR, filename: str | None = None):
        cust_df = CustomerDFGenator.generate()
        filename = filename or CustomerDFGenator.filename
        path = path / filename
        cust_df.to_parquet(path)
        print(f"Saved CustomerDF to {path}")


class CustomerDF(pd.DataFrame):
    cust_id: np.ndarray
    bias: np.ndarray
    temperature: np.ndarray


def price_acceptance_probability(
    cust: CustomerDF,
    requests: RequestDF,
    prices: np.ndarray,
    supply_demand_ratio: float,
    config: EnvConfig | None = None,
):
    """
    Compute the probability that a customer will accept a ride request
    based on the pricing and other factors.
    TODO implement the acceptance model based on margin (price - cost), supply-demand ratio, customer bias and temperature.
    Return a float between 0.0 and 1.0
    """
    config = config or EnvConfig()
    z = (
        cust.bias
        + (prices - requests.est_cost) * config.acceptance_margin_weight
        + config.acceptance_supply_demand_weight * supply_demand_ratio
    ) / cust.temperature

    acceptance_prob = 1.0 / (1.0 + np.exp(-z))

    return acceptance_prob, z
