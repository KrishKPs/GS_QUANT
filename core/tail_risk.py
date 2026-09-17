"""VaR and Expected Shortfall at a 1-day horizon, reported as positive losses.

Scale to h days with sqrt(h) if needed. Quantiles and moments of the realised
distribution — no model is fitted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import gsq


def historical_var(r: pd.Series, alpha: float = 0.05) -> float:
    """-quantile(r, alpha) via `gs_quant.timeseries.percentile`.

    Assumption-free but bounded by the observed history.
    """
    return float(-gsq.percentile(r.dropna(), alpha * 100))


def gaussian_var(r: pd.Series, alpha: float = 0.05) -> float:
    """-(mu + z_alpha * sigma), z_alpha = Phi^-1(alpha)."""
    mu, sd = r.mean(), r.std(ddof=1)
    return float(-(mu + stats.norm.ppf(alpha) * sd))


def cornish_fisher_z(alpha: float, skew: float, excess_kurt: float) -> float:
    """z_cf = z + (z^2-1)S/6 + (z^3-3z)K/24 - (2z^3-5z)S^2/36.

    Invariant: z_cf == z when S = K = 0.
    """
    z = stats.norm.ppf(alpha)
    return float(
        z
        + (z ** 2 - 1) * skew / 6
        + (z ** 3 - 3 * z) * excess_kurt / 24
        - (2 * z ** 3 - 5 * z) * skew ** 2 / 36
    )


def cornish_fisher_var(r: pd.Series, alpha: float = 0.05) -> float:
    """Gaussian VaR with the sample's skew and excess kurtosis folded back in.

    Invariant: reduces to gaussian_var when the sample is symmetric/mesokurtic.
    """
    mu, sd = r.mean(), r.std(ddof=1)
    z_cf = cornish_fisher_z(alpha, float(r.skew()), float(r.kurtosis()))
    return float(-(mu + z_cf * sd))


def historical_es(r: pd.Series, alpha: float = 0.05) -> float:
    """Mean loss in the alpha tail (empirical CVaR)."""
    x = r.dropna()
    tail = x[x <= gsq.percentile(x, alpha * 100)]
    return float(-tail.mean()) if len(tail) else float("nan")


def gaussian_es(r: pd.Series, alpha: float = 0.05) -> float:
    """-(mu - sigma * phi(z_alpha)/alpha)."""
    mu, sd = r.mean(), r.std(ddof=1)
    z = stats.norm.ppf(alpha)
    return float(-(mu - sd * stats.norm.pdf(z) / alpha))


def tail_table(r: pd.Series, alphas=(0.01, 0.05)) -> pd.DataFrame:
    """VaR/ES grid. ES is coherent (sub-additive) where VaR is not."""
    rows = {}
    for a in alphas:
        rows[f"{1 - a:.0%}"] = {
            "VaR_historical": historical_var(r, a),
            "VaR_gaussian": gaussian_var(r, a),
            "VaR_cornish_fisher": cornish_fisher_var(r, a),
            "ES_historical": historical_es(r, a),
            "ES_gaussian": gaussian_es(r, a),
        }
    return pd.DataFrame(rows).rename_axis("measure")
