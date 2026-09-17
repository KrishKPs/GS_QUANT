"""Ex-post performance statistics on realised returns.

None of these imply anything about future performance — they describe what the
book already did.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from . import gsq
from .conventions import PERIODS_PER_YEAR


def volatility(r: pd.Series, periods: int = PERIODS_PER_YEAR) -> float:
    """Annualised volatility via `gs_quant.timeseries.volatility` (sigma*sqrt(252))."""
    return gsq.volatility(r) if periods == PERIODS_PER_YEAR else float(
        r.std(ddof=1) * np.sqrt(periods))


def rolling_volatility(r: pd.Series, window: int = 126) -> pd.Series:
    """Rolling annualised vol (gs-quant) — descriptive view of regime shifts."""
    return gsq.rolling_volatility(r, window)


def ewma_volatility(r: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics-style exponentially weighted vol via gs-quant."""
    return gsq.ewma_volatility(r, lam)


def annual_return(r: pd.Series, periods: int = PERIODS_PER_YEAR) -> float:
    """Geometric (CAGR) annual return from the realised wealth path."""
    w = float((1 + r).prod())
    return float(w ** (periods / len(r)) - 1) if w > 0 else float("nan")


def sharpe(r: pd.Series, periods: int = PERIODS_PER_YEAR) -> float:
    """(mean/std) * sqrt(252). Feed it *excess* returns."""
    sd = r.std(ddof=1)
    return float("nan") if sd == 0 else float(r.mean() / sd * np.sqrt(periods))


def sortino(r: pd.Series, mar: float = 0.0, periods: int = PERIODS_PER_YEAR) -> float:
    """(mean - MAR)/DD * sqrt(252), DD = sqrt(mean(min(r-MAR,0)^2))."""
    downside = np.minimum(r - mar, 0.0)
    dd = np.sqrt(np.mean(downside ** 2))
    return float("nan") if dd == 0 else float((r.mean() - mar) / dd * np.sqrt(periods))


def information_ratio(r: pd.Series, benchmark: pd.Series,
                      periods: int = PERIODS_PER_YEAR) -> float:
    """Active return / tracking error, annualised."""
    active = (r - benchmark).dropna()
    sd = active.std(ddof=1)
    return float("nan") if sd == 0 else float(active.mean() / sd * np.sqrt(periods))


def market_beta(r: pd.Series, market: pd.Series) -> float:
    """Beta against a market return series, via `gs_quant.timeseries.beta`."""
    return gsq.beta(r, market)


def treynor(r: pd.Series, market_beta: float, periods: int = PERIODS_PER_YEAR) -> float:
    """Excess return per unit of *market* beta (from the Tier 1 regression)."""
    return float("nan") if market_beta == 0 else float(r.mean() * periods / market_beta)


def wealth(r: pd.Series) -> pd.Series:
    """W_t = prod(1 + r), via `gs_quant.timeseries.prices`."""
    return gsq.wealth(r)


def drawdown(r: pd.Series) -> pd.Series:
    """d_t = W_t / max_{s<=t} W_s - 1 (<= 0)."""
    w = wealth(r)
    return (w / w.cummax() - 1).rename("drawdown")


def max_drawdown(r: pd.Series) -> float:
    """min_t d_t (negative), via `gs_quant.timeseries.max_drawdown`."""
    return gsq.max_drawdown(r)


def drawdown_duration(r: pd.Series) -> int:
    """Longest run of consecutive periods spent below a prior peak."""
    under = drawdown(r) < 0
    if not under.any():
        return 0
    groups = (~under).cumsum()[under]
    return int(groups.value_counts().max())


def calmar(r: pd.Series, periods: int = PERIODS_PER_YEAR) -> float:
    """Annual return / |max drawdown|."""
    mdd = abs(max_drawdown(r))
    return float("nan") if mdd == 0 else float(annual_return(r, periods) / mdd)


def ulcer_index(r: pd.Series) -> float:
    """sqrt(mean(d_t^2)) — depth *and* duration of drawdown in one number."""
    return float(np.sqrt(np.mean(drawdown(r) ** 2)))


def moments(r: pd.Series) -> pd.Series:
    """Mean, vol, skew and excess kurtosis.

    Reported because Sharpe assumes normality and returns are fat-tailed and
    left-skewed — that is what licenses Cornish-Fisher VaR over Gaussian.
    """
    return pd.Series({
        "mean_daily": float(r.mean()),
        "vol_daily": float(r.std(ddof=1)),
        "skew": float(r.skew()),
        "excess_kurtosis": float(r.kurtosis()),  # pandas already subtracts 3
    }, name="moments")


def performance_summary(r: pd.Series, excess: Optional[pd.Series] = None,
                        benchmark: Optional[pd.Series] = None,
                        market_beta: Optional[float] = None) -> pd.Series:
    """One tidy column of ex-post statistics for display."""
    e = r if excess is None else excess
    out = {
        "annual_return": annual_return(r),
        "annual_vol": volatility(r),
        "sharpe": sharpe(e),
        "sortino": sortino(e),
        "max_drawdown": max_drawdown(r),
        "drawdown_days": drawdown_duration(r),
        "calmar": calmar(r),
        "ulcer_index": ulcer_index(r),
    }
    out.update(moments(r).to_dict())
    if benchmark is not None:
        out["information_ratio"] = information_ratio(r, benchmark)
    if market_beta is not None:
        out["treynor"] = treynor(e, market_beta)
    return pd.Series(out, name="value")
