"""Classical hypothesis tests. Each result *licenses* a modelling choice elsewhere.

Descriptive statistics about the sample — nothing here fits a predictive model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

from .conventions import hac_lag


def adf_test(x: pd.Series, alpha: float = 0.05) -> pd.Series:
    """Augmented Dickey-Fuller. H0: a unit root (non-stationary).

    Licenses: regressing on a factor series at all — a non-stationary
    regressor makes the betas spurious.
    """
    stat, p, lags, nobs, crit, _ = adfuller(x.dropna(), autolag="AIC")
    return pd.Series({"statistic": stat, "p_value": p, "lags": lags,
                      "crit_5pct": crit["5%"], "stationary": p < alpha}, name=x.name)


def jarque_bera(x: pd.Series, alpha: float = 0.05) -> pd.Series:
    """JB = (T/6)(S^2 + K^2/4) ~ chi2_2. H0: normally distributed.

    Licenses: Cornish-Fisher VaR and Expected Shortfall over Gaussian VaR —
    if normality is rejected, the Gaussian tail is the wrong tail.
    """
    r = x.dropna()
    s, k = float(r.skew()), float(r.kurtosis())
    jb = len(r) / 6.0 * (s ** 2 + k ** 2 / 4.0)
    p = float(stats.chi2.sf(jb, 2))
    return pd.Series({"statistic": jb, "p_value": p, "skew": s, "excess_kurtosis": k,
                      "normal": p >= alpha}, name=x.name)


def ljung_box(x: pd.Series, lags: int = 10, alpha: float = 0.05) -> pd.Series:
    """Ljung-Box Q. H0: no autocorrelation up to `lags`.

    Licenses: HAC (Newey-West) standard errors — autocorrelated residuals make
    plain OLS t-stats overconfident.
    """
    res = acorr_ljungbox(x.dropna(), lags=[lags], return_df=True)
    stat, p = float(res["lb_stat"].iloc[0]), float(res["lb_pvalue"].iloc[0])
    return pd.Series({"statistic": stat, "p_value": p, "lags": lags,
                      "autocorrelated": p < alpha}, name=x.name)


def diagnostics_table(residuals: pd.Series, factors: pd.DataFrame) -> pd.DataFrame:
    """Every test with the modelling choice its result licenses."""
    rows = []
    jb = jarque_bera(residuals)
    lb = ljung_box(residuals)
    rows.append({"test": "Jarque-Bera (residuals)", "statistic": jb["statistic"],
                 "p_value": jb["p_value"],
                 "conclusion": "normal" if jb["normal"] else "non-normal (fat tails / skew)",
                 "licenses": "Cornish-Fisher VaR and ES over Gaussian VaR"})
    rows.append({"test": "Ljung-Box (residuals, 10 lags)", "statistic": lb["statistic"],
                 "p_value": lb["p_value"],
                 "conclusion": "autocorrelated" if lb["autocorrelated"] else "no autocorrelation",
                 "licenses": f"HAC (Newey-West) errors at lag {hac_lag(len(residuals))}"})
    for col in factors.columns:
        a = adf_test(factors[col])
        rows.append({"test": f"ADF ({col})", "statistic": a["statistic"], "p_value": a["p_value"],
                     "conclusion": "stationary" if a["stationary"] else "unit root",
                     "licenses": "using this factor as a regressor"})
    return pd.DataFrame(rows).set_index("test")
