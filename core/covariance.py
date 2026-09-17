"""Covariance estimators. Estimators of present structure, not forecasts."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

COND_WARN = 1e10


def nearest_psd(m) -> pd.DataFrame | np.ndarray:
    """Symmetrise, clip negative eigenvalues at 0, restore the diagonal.

    Invariant: min eigenvalue of the result >= 0 (to floating tolerance).
    """
    idx = m.index if isinstance(m, pd.DataFrame) else None
    a = np.asarray(m, dtype=float)
    a = (a + a.T) / 2.0
    vals, vecs = np.linalg.eigh(a)
    if (vals >= 0).all():
        out = a
    else:
        out = vecs @ np.diag(np.clip(vals, 0.0, None)) @ vecs.T
        out = (out + out.T) / 2.0
        d_old, d_new = np.diag(a).copy(), np.diag(out).copy()
        scale = np.where(d_new > 0, np.sqrt(np.clip(d_old, 0, None) / np.where(d_new > 0, d_new, 1)), 1.0)
        out = out * np.outer(scale, scale)  # rescale so variances survive clipping
    return pd.DataFrame(out, index=idx, columns=idx) if idx is not None else out


def sample_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """Plain sample covariance, routed through nearest_psd."""
    return nearest_psd(returns.cov())


def check_condition(cov) -> float:
    """Condition number, warning when the matrix is effectively singular."""
    c = float(np.linalg.cond(np.asarray(cov, dtype=float)))
    if c > COND_WARN:
        warnings.warn(f"Covariance is ill-conditioned (cond={c:.2e}); prefer shrinkage.")
    return c


def ledoit_wolf_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """Sigma_hat = delta*T + (1-delta)*S with the closed-form LW intensity.

    Shrinkage trades a little bias for a large variance reduction and
    guarantees invertibility. It is an *estimator* of present structure, not a
    predictor. Invariant: symmetric PSD and invertible.
    """
    from sklearn.covariance import LedoitWolf

    x = returns.dropna()
    lw = LedoitWolf().fit(x.values)
    out = pd.DataFrame(lw.covariance_, index=x.columns, columns=x.columns)
    out.attrs["shrinkage"] = float(lw.shrinkage_)
    return nearest_psd(out)


def ewma_cov(returns: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    """RiskMetrics: Sigma_t = lam*Sigma_{t-1} + (1-lam) r_t r_t'.

    Written as the equivalent exponentially weighted sum of outer products.
    Emphasises recent structure; still describes the sample, predicts nothing.
    """
    if not 0 < lam < 1:
        raise ValueError("lambda must be in (0, 1).")
    x = returns.dropna()
    r = x.values - x.values.mean(axis=0)
    T = len(r)
    w = lam ** np.arange(T - 1, -1, -1)
    w /= w.sum()
    cov = np.einsum("t,ti,tj->ij", w, r, r)
    return nearest_psd(pd.DataFrame(cov, index=x.columns, columns=x.columns))


ESTIMATORS = {"Sample": sample_cov, "Ledoit-Wolf": ledoit_wolf_cov, "EWMA (0.94)": ewma_cov}
