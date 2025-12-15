"""
TODO
"""

import pathlib
import typing as t

import numpy as np
import pandas as pd
import torch
from scipy.special import expit

from waymo_agent.constants import DATA_DIR
from waymo_agent.data_classes.config import EnvConfig

from . import RequestDF


class CustomerDFGenator:
    size: int = 50_000
    loc = -1.0
    scale = 2
    temp_low = 1.5
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
    requests: RequestDF,
    prices: np.ndarray | torch.Tensor,
    supply_demand_ratio_z: float,  # this is a scalar, sigmoid of supply-demand ratio
    config: EnvConfig | None = None,
):
    """
    Compute the probability that a customer will accept a ride request
    based on the pricing and other factors.
    TODO implement the acceptance model based on margin (price - cost), supply-demand ratio, customer bias and temperature.
    Return a float between 0.0 and 1.0
    """
    assert len(requests) == len(prices), f"Length of requests {len(requests)=} != length of prices {len(prices)=}"
    config = config or EnvConfig()
    prices = prices.numpy(force=True) if isinstance(prices, torch.Tensor) else prices

    profit_margin = (prices - requests["est_cost"]) / (requests["est_cost"] + 1e-5)

    z = (
        requests.cust_bias
        + profit_margin * config.acceptance_margin_weight
        + config.acceptance_supply_demand_weight * supply_demand_ratio_z
    ) / requests.cust_temperature

    acceptance_prob: np.ndarray = expit(z)

    return acceptance_prob, z


def price_acceptance_probability_old(
    cust: CustomerDF,
    requests: RequestDF,
    prices: np.ndarray,
    supply_demand_ratio_z: float,  # this is a scalar, sigmoid of supply-demand ratio
    config: EnvConfig | None = None,
):
    """
    Compute the probability that a customer will accept a ride request
    based on the pricing and other factors.
    TODO implement the acceptance model based on margin (price - cost), supply-demand ratio, customer bias and temperature.
    Return a float between 0.0 and 1.0
    """
    assert len(requests) == len(prices), f"Length of requests {len(requests)=} != length of prices {len(prices)=}"
    config = config or EnvConfig()

    cust_enrich = t.cast(CustomerDF, pd.merge(requests[["cust_id", "est_cost"]], cust, on="cust_id", how="left"))
    profit_margin = (prices - cust_enrich["est_cost"]) / (cust_enrich["est_cost"] + 1e-5)

    z = (
        cust_enrich.bias
        + profit_margin * config.acceptance_margin_weight
        + config.acceptance_supply_demand_weight * supply_demand_ratio_z
    ) / cust_enrich.temperature

    acceptance_prob: np.ndarray = expit(z)

    return acceptance_prob, z
