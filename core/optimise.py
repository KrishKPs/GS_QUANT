"""Deterministic portfolio construction. No return forecast anywhere.

Every optimiser below needs only the covariance matrix, which is what keeps
these inside the prime directive: given Sigma the solution is a deterministic
convex program, not a learned mapping.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .covariance import nearest_psd
from .risk_decomp import euler_decomposition


def _prep(cov: pd.DataFrame) -> tuple:
    S = nearest_psd(pd.DataFrame(cov))
    return S, np.asarray(S, dtype=float), list(S.index)


def min_variance(cov: pd.DataFrame, long_only: bool = False) -> pd.Series:
    """min w'Sigma w s.t. 1'w = 1.

    Closed form w* = Sigma^-1 1 / (1' Sigma^-1 1) when shorts are allowed
    (solved as a linear system, never an explicit inverse); SLSQP otherwise.
    Invariant: its variance is <= that of any other feasible portfolio.
    """
    S, A, names = _prep(cov)
    n = len(names)
    if not long_only:
        z = np.linalg.solve(A, np.ones(n))
        w = z / z.sum()
    else:
        w = _solve_qp(lambda x: x @ A @ x, n, jac=lambda x: 2 * A @ x)
    return pd.Series(w, index=names, name="min_variance")


def risk_parity(cov: pd.DataFrame) -> pd.Series:
    """Equal risk contribution via the convex Spinu/Maillard formulation.

    min 0.5 w'Sigma w - (1/n) sum(ln w_i), then normalise to sum 1.
    Invariant: at the solution all CCR_i are equal (Euler contributions, §1.4).
    """
    S, A, names = _prep(cov)
    n = len(names)
    scale = float(np.mean(np.diag(A)))  # unit-variance scaling; solution is scale-invariant
    A = A / scale

    def obj(w):
        return 0.5 * w @ A @ w - np.mean(np.log(w))

    def grad(w):
        return A @ w - 1.0 / (n * w)

    res = minimize(obj, np.ones(n) / np.sqrt(n), jac=grad, method="L-BFGS-B",
                   bounds=[(1e-10, None)] * n,
                   options={"ftol": 1e-18, "gtol": 1e-14, "maxiter": 10000})
    if np.max(np.abs(grad(res.x))) > 1e-6:
        raise RuntimeError(f"Risk parity did not converge: {res.message}")
    return pd.Series(res.x / res.x.sum(), index=names, name="risk_parity")


def max_diversification(cov: pd.DataFrame) -> pd.Series:
    """Maximise DR = (sum w_i sigma_i) / sigma_p, long-only, fully invested."""
    S, A, names = _prep(cov)
    sd = np.sqrt(np.diag(A))
    w = _solve_qp(lambda x: -(x @ sd) / np.sqrt(max(x @ A @ x, 1e-300)), len(names))
    return pd.Series(w, index=names, name="max_diversification")


def diversification_ratio(weights: pd.Series, cov: pd.DataFrame) -> float:
    """DR = (sum w_i sigma_i) / sigma_p. Equals 1 for a single asset."""
    S, A, names = _prep(cov)
    w = weights.reindex(names).astype(float).values
    return float((w @ np.sqrt(np.diag(A))) / np.sqrt(w @ A @ w))


def _solve_qp(fun, n: int, jac=None) -> np.ndarray:
    """Long-only, fully-invested SLSQP helper."""
    res = minimize(fun, np.ones(n) / n, jac=jac, method="SLSQP",
                   bounds=[(0.0, 1.0)] * n,
                   constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1.0,
                                 "jac": lambda x: np.ones_like(x)}],
                   options={"ftol": 1e-14, "maxiter": 1000})
    if not res.success:
        raise RuntimeError(f"Optimiser failed: {res.message}")
    return res.x


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Percent contribution to volatility — the comparison axis for §Tier 5."""
    return euler_decomposition(weights, cov)["pct_risk"]


def efficient_frontier(mean_returns: pd.Series, cov: pd.DataFrame,
                       n_points: int = 25) -> pd.DataFrame:
    """EX-POST / DESCRIPTIVE ONLY — label it as such in any UI.

    Markowitz QP over a grid of target *historical mean* returns. This is the
    only place expected returns appear; it draws the realised trade-off curve
    and forecasts nothing. Historical mean is a weak proxy for expected return.
    """
    S, A, names = _prep(cov)
    mu = mean_returns.reindex(names).astype(float).values
    n = len(names)
    rows = []
    for target in np.linspace(mu.min(), mu.max(), n_points):
        cons = [{"type": "eq", "fun": lambda x: x.sum() - 1.0},
                {"type": "eq", "fun": lambda x, t=target: x @ mu - t}]
        res = minimize(lambda x: x @ A @ x, np.ones(n) / n, jac=lambda x: 2 * A @ x,
                       method="SLSQP", bounds=[(0.0, 1.0)] * n, constraints=cons,
                       options={"ftol": 1e-12, "maxiter": 500})
        if res.success:
            rows.append({"target_return": target, "vol": float(np.sqrt(res.x @ A @ res.x)),
                         **dict(zip(names, res.x))})
    return pd.DataFrame(rows)
