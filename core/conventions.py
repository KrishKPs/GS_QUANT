"""Annualisation constants and small shared helpers.

Volatility scales with sqrt(time), means scale with time, covariance with time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252
MIN_OBS = 60  # refuse regressions / decompositions below this


def annualise_mean(x: float | pd.Series, periods: int = PERIODS_PER_YEAR):
    """Mean of per-period returns -> annual return. mu * 252."""
    return x * periods


def annualise_vol(x: float | pd.Series, periods: int = PERIODS_PER_YEAR):
    """Per-period stdev -> annual volatility. sigma * sqrt(252)."""
    return x * np.sqrt(periods)


def annualise_cov(m: pd.DataFrame | np.ndarray, periods: int = PERIODS_PER_YEAR):
    """Per-period covariance -> annual covariance. Sigma * 252."""
    return m * periods


def hac_lag(n_obs: int) -> int:
    """Newey-West Bartlett lag: L = floor(4*(T/100)^(2/9))."""
    return int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))


def check_obs(n_obs: int, what: str = "estimation") -> None:
    """Fail loudly when the sample is too short to be stable."""
    if n_obs < MIN_OBS:
        raise ValueError(
            f"{what} needs at least {MIN_OBS} observations, got {n_obs}. "
            "Widen the date window or drop short-history tickers."
        )


def normalise_weights(weights: pd.Series) -> pd.Series:
    """Weights sum to 1. Fails on a zero-sum book rather than dividing by 0."""
    total = float(weights.sum())
    if np.isclose(total, 0.0):
        raise ValueError("Weights sum to zero; cannot normalise.")
    return weights / total
