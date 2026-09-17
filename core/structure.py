"""Structural diagnostics: PCA, effective bets, multicollinearity, correlation.

Pure linear algebra describing structure that already exists in the sample.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from . import gsq
from .covariance import nearest_psd


def pca(cov: pd.DataFrame, use_correlation: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Eigen-decomposition of Sigma -> (spectrum, loadings).

    Eigenvalues are variance along independent directions; the cumulative
    column answers "how many real independent bets does this book hold?".
    """
    S = pd.DataFrame(nearest_psd(pd.DataFrame(cov)))
    if use_correlation:
        d = np.sqrt(np.diag(S.values))
        S = pd.DataFrame(S.values / np.outer(d, d), index=S.index, columns=S.columns)
    vals, vecs = np.linalg.eigh(S.values)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    names = [f"PC{i + 1}" for i in range(len(vals))]
    spectrum = pd.DataFrame({
        "eigenvalue": vals,
        "explained": vals / vals.sum(),
        "cumulative": np.cumsum(vals / vals.sum()),
    }, index=pd.Index(names, name="component"))
    loadings = pd.DataFrame(vecs, index=S.index, columns=names)
    return spectrum, loadings


def effective_bets(cov: pd.DataFrame) -> float:
    """N_eff = exp(-sum p_i ln p_i), p_i = lambda_i / sum(lambda).

    Invariant: 1 <= N_eff <= n. A 20-name book can hold 3 effective bets —
    this turns hidden concentration into one number.
    """
    vals = np.linalg.eigvalsh(np.asarray(nearest_psd(pd.DataFrame(cov)), dtype=float))
    p = np.clip(vals, 0, None)
    p = p[p > 0] / p.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def vif(factors: pd.DataFrame) -> pd.Series:
    """VIF_j = 1/(1 - R^2_j) from regressing factor j on the others (gs-quant OLS).

    Flags the multicollinearity (classically HML vs MOM) that destabilises the
    Tier 1 betas.
    """
    x = factors.dropna()
    out = {}
    for col in x.columns:
        others = x.drop(columns=[col])
        _, _, r2, _ = gsq.ols(x[col], others)
        out[col] = float("inf") if r2 >= 1 else 1.0 / (1.0 - r2)
    return pd.Series(out, name="VIF").rename_axis("factor")


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Plain Pearson correlation, for display."""
    return df.corr()


def concentration(weights: pd.Series) -> pd.Series:
    """Herfindahl and the weight-vs-risk gap's simplest summary."""
    w = weights.astype(float)
    return pd.Series({"herfindahl": float((w ** 2).sum()),
                      "effective_positions": float(1.0 / (w ** 2).sum())})
