"""Adapter over `gs_quant.timeseries` — the primary analytics library here.

Everything in this file is offline, deterministic maths from Goldman's open
source toolkit: no Marquee/GsSession needed, no market data fetched, nothing
predicted. The measures that gs-quant does not ship (HAC inference, Euler risk
decomposition, Ledoit-Wolf, Cornish-Fisher, the optimisers) are built on top of
these primitives in the sibling modules.

gs-quant conventions worth knowing:
  * `volatility` returns *percent* annualised -> divided by 100 here.
  * `Window(None, 0)` means "full sample, no ramp-up".
  * `beta`/`correlation` take `prices=False` when fed return series.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from gs_quant.timeseries import Returns, Window
from gs_quant.timeseries import beta as _beta
from gs_quant.timeseries import correlation as _correlation
from gs_quant.timeseries import max_drawdown as _max_drawdown
from gs_quant.timeseries import prices as _prices
from gs_quant.timeseries import returns as _returns
from gs_quant.timeseries import volatility as _volatility
from gs_quant.timeseries.statistics import LinearRegression
from gs_quant.timeseries.statistics import exponential_std as _exponential_std
from gs_quant.timeseries.statistics import percentile as _percentile
from gs_quant.timeseries.statistics import winsorize as _winsorize

FULL = Window(None, 0)  # full sample, no ramp
PCT = 100.0


def returns(prices: pd.DataFrame | pd.Series, log: bool = False):
    """gs_quant.timeseries.returns per column. Simple by default."""
    kind = Returns.LOGARITHMIC if log else Returns.SIMPLE
    if isinstance(prices, pd.Series):
        return _returns(prices, type=kind)
    return prices.apply(lambda c: _returns(c, type=kind))


def wealth(r: pd.Series) -> pd.Series:
    """gs-quant price index from a return series: W_t = prod(1 + r)."""
    return _prices(r).rename("wealth")


def volatility(r: pd.Series) -> float:
    """Full-sample annualised volatility (decimal) via gs-quant."""
    return float(_volatility(wealth(r), FULL).iloc[-1]) / PCT


def rolling_volatility(r: pd.Series, window: int = 126) -> pd.Series:
    """Rolling annualised volatility (decimal) via gs-quant."""
    return (_volatility(wealth(r), Window(window, window)) / PCT).dropna().rename("rolling_vol")


def ewma_volatility(r: pd.Series, lam: float = 0.94) -> pd.Series:
    """gs-quant exponentially weighted std (per-period, decimal)."""
    return _exponential_std(r, lam).dropna().rename("ewma_vol")


def max_drawdown(r: pd.Series) -> float:
    """Worst peak-to-trough of the realised wealth path, via gs-quant."""
    return float(_max_drawdown(wealth(r), FULL).iloc[-1])


def beta(r: pd.Series, market: pd.Series) -> float:
    """Full-sample beta of a return series against a market return series."""
    return float(_beta(r, market, FULL, prices=False).iloc[-1])


def correlation(x: pd.Series, y: pd.Series) -> float:
    """Full-sample correlation of two return series."""
    return float(_correlation(x, y, FULL, type_=Returns.SIMPLE).iloc[-1])


def percentile(x: pd.Series, pct: float) -> float:
    """Full-sample percentile (pct in 0-100) — the quantile behind historical VaR."""
    return float(_percentile(x, pct, FULL).iloc[-1])


def winsorize(x: pd.Series, limit: float = 2.5) -> pd.Series:
    """gs-quant winsorisation at `limit` standard deviations.

    Robustness toggle only — never applied silently.
    """
    return _winsorize(x, limit, FULL)


def ols(y: pd.Series, X: pd.DataFrame) -> Tuple[pd.Series, float, float, pd.Series]:
    """gs-quant LinearRegression -> (betas, intercept, R^2, fitted).

    Same normal equations as OLS; used as the primary beta estimator, with
    statsmodels layered on only for HAC-robust inference.
    """
    df = pd.concat([y.rename("__y"), X], axis=1).dropna()
    cols: List[str] = list(X.columns)
    lr = LinearRegression([df[c] for c in cols], df["__y"], fit_intercept=True)
    betas = pd.Series({c: float(lr.coefficient(i + 1)) for i, c in enumerate(cols)}, name="beta")
    return betas, float(lr.coefficient(0)), float(lr.r_squared()), lr.fitted_values()
