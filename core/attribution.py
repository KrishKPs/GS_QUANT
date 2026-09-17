"""Factor regression: r_p = alpha + F*beta + eps, with HAC (Newey-West) errors.

This is attribution of *realised* return. It forecasts nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import gsq
from .conventions import PERIODS_PER_YEAR, check_obs, hac_lag


@dataclass
class Attribution:
    """OLS factor attribution with HAC-robust inference.

    Invariant: Var(fitted) + Var(resid) == Var(r_p) (OLS orthogonality).
    """

    betas: pd.Series          # factor exposures
    tstats: pd.Series         # HAC t-stats (|t| >= 2 -> reliable)
    pvalues: pd.Series
    alpha_daily: float
    alpha_ann: float          # daily alpha * 252
    alpha_tstat: float
    r2: float
    adj_r2: float
    resid: pd.Series
    fitted: pd.Series
    hac_lag: int
    n_obs: int

    @property
    def summary(self) -> pd.DataFrame:
        """Tidy exposure table for display."""
        return pd.DataFrame(
            {"beta": self.betas, "t_stat": self.tstats, "p_value": self.pvalues,
             "reliable": self.tstats.abs() >= 2},
        ).rename_axis("factor")


def factor_regression(
    excess_returns: pd.Series,
    factors: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> Attribution:
    """Betas from `gs_quant.timeseries.LinearRegression`; HAC errors from statsmodels.

    gs-quant gives the point estimates (same normal equations, beta_hat =
    (F'F)^-1 F'r); statsmodels supplies the Bartlett-kernel HAC covariance that
    gs-quant does not ship. The two coefficient sets are asserted equal.

    HAC lag L = floor(4*(T/100)^(2/9)); daily residuals are autocorrelated and
    heteroskedastic, so plain OLS t-stats would be overconfident.
    """
    F = factors[columns] if columns else factors
    df = pd.concat([excess_returns.rename("y"), F], axis=1, join="inner").dropna()
    check_obs(len(df), "Factor regression")

    names = list(F.columns)
    betas, alpha, r2, _ = gsq.ols(df["y"], df[names])  # gs-quant point estimates

    y, X = df["y"], sm.add_constant(df[names], has_constant="add")
    L = hac_lag(len(df))
    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": L})  # HAC inference only
    assert np.allclose(fit.params[names].values, betas.values, atol=1e-8)

    return Attribution(
        betas=betas,
        tstats=fit.tvalues[names].rename("t_stat"),
        pvalues=fit.pvalues[names].rename("p_value"),
        alpha_daily=alpha,
        alpha_ann=alpha * PERIODS_PER_YEAR,
        alpha_tstat=float(fit.tvalues["const"]),
        r2=r2,
        adj_r2=float(fit.rsquared_adj),
        resid=fit.resid.rename("resid"),
        fitted=fit.fittedvalues.rename("fitted"),
        hac_lag=L,
        n_obs=int(len(df)),
    )


def rolling_betas(
    excess_returns: pd.Series,
    factors: pd.DataFrame,
    window: int = 252,
) -> pd.DataFrame:
    """Rolling OLS betas — shows that exposures drift (a stated limitation).

    Descriptive only: each window is its own in-sample attribution.
    """
    df = pd.concat([excess_returns.rename("y"), factors], axis=1, join="inner").dropna()
    if len(df) <= window:
        raise ValueError(f"Need more than {window} observations for rolling betas.")
    X = sm.add_constant(df[factors.columns], has_constant="add")
    from statsmodels.regression.rolling import RollingOLS

    res = RollingOLS(df["y"], X, window=window).fit()
    return res.params[list(factors.columns)].dropna()
