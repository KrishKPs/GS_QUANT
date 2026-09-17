"""Where the risk actually lives: variance shares and Euler volatility shares."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .covariance import nearest_psd


def variance_decomposition(
    betas: pd.Series,
    factor_cov: pd.DataFrame,
    resid_var: float,
) -> pd.DataFrame:
    """Component contribution to variance: CCV_i = beta_i * (Sigma_F beta)_i.

    Invariants: sum(CCV) == beta' Sigma_F beta (systematic variance), and with
    the idiosyncratic row appended the shares sum to 1.
    Shares account for factor correlation through the cross-terms, so this is
    not just squared betas.
    """
    b = betas.astype(float)
    S = nearest_psd(pd.DataFrame(factor_cov).loc[b.index, b.index])
    ccv = b * S.values.dot(b.values)
    systematic = float(b.values @ S.values @ b.values)
    total = systematic + float(resid_var)
    out = pd.concat([ccv, pd.Series({"IDIOSYNCRATIC": float(resid_var)})])
    return pd.DataFrame(
        {"variance_contribution": out, "share": out / total}
    ).rename_axis("factor").assign(
        systematic_variance=systematic, total_variance=total
    )


def euler_decomposition(weights: pd.Series, cov: pd.DataFrame) -> pd.DataFrame:
    """Marginal and component contributions to *volatility*.

    MCR_i = (Sigma w)_i / sigma_p, CCR_i = w_i * MCR_i.
    Invariant (Euler, sigma_p homogeneous of degree 1 in w): sum(CCR) == sigma_p.
    Requires a PSD Sigma or the contributions are meaningless — enforced here.
    """
    w = weights.astype(float)
    S = nearest_psd(pd.DataFrame(cov).loc[w.index, w.index])
    var = float(w.values @ S.values @ w.values)
    if var <= 0:
        raise ValueError("Portfolio variance is zero; risk contributions undefined.")
    sigma = np.sqrt(var)
    mcr = pd.Series(S.values.dot(w.values) / sigma, index=w.index)
    ccr = w * mcr
    return pd.DataFrame(
        {"weight": w, "mcr": mcr, "ccr": ccr, "pct_risk": ccr / sigma}
    ).rename_axis("asset").assign(portfolio_vol=sigma)


def portfolio_vol(weights: pd.Series, cov: pd.DataFrame) -> float:
    """sigma_p = sqrt(w' Sigma w)."""
    w = weights.astype(float)
    S = nearest_psd(pd.DataFrame(cov).loc[w.index, w.index])
    return float(np.sqrt(w.values @ S.values @ w.values))
