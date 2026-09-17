"""FastAPI JSON API over the gs-quant analytics engine.

Transport only: every number comes from `core/`. Nothing here forecasts.
"""
from __future__ import annotations

import warnings
from datetime import date
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core import data as D
from core import diagnostics as DG
from core import optimise as OPT
from core import risk_metrics as RM
from core import structure as STR
from core import tail_risk as TR
from core.attribution import factor_regression, rolling_betas
from core.conventions import annualise_cov
from core.covariance import check_condition, ewma_cov, ledoit_wolf_cov, sample_cov
from core.risk_decomp import euler_decomposition, variance_decomposition

ESTIMATORS = {"sample": sample_cov, "ledoit_wolf": ledoit_wolf_cov, "ewma": ewma_cov}
ALPHA = 0.05

app = FastAPI(title="Portfolio Attribution API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Holding(BaseModel):
    ticker: str
    weight: float


class AnalyzeRequest(BaseModel):
    holdings: List[Holding] = Field(min_length=1)
    start: date
    end: date
    estimator: str = "ledoit_wolf"


def _f(x) -> Optional[float]:
    """JSON-safe float (NaN/inf -> null)."""
    v = float(x)
    return None if not np.isfinite(v) else v


def _series(s: pd.Series) -> List[Optional[float]]:
    return [_f(v) for v in s.values]


def _dates(idx: pd.Index) -> List[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.DatetimeIndex(idx)]


@lru_cache(maxsize=32)
def _load(tickers: tuple, start: str, end: str):
    """Cached network calls — prices and Ken French factors."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prices = D.load_prices(list(tickers), start, end)
        skipped = [str(w.message) for w in caught]
    return prices, D.load_factors(start, end), skipped


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "engine": "gs-quant"}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    weights = pd.Series({h.ticker.upper().strip(): h.weight for h in req.holdings})
    try:
        prices, factors, skipped = _load(tuple(sorted(weights.index)),
                                         req.start.isoformat(), req.end.isoformat())
    except Exception as exc:  # bad tickers / empty window — say what and how to fix
        raise HTTPException(422, str(exc))

    returns = D.to_returns(prices)
    try:
        weights = D.align_weights(returns, weights)
        port = D.portfolio_return(returns, weights)
        aligned = D.align(port, factors)
        att = factor_regression(aligned["PORT_EXCESS"], aligned[D.factor_names(aligned)])
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    names = D.factor_names(aligned)
    r_p, r_x = aligned["PORT"], aligned["PORT_EXCESS"]
    asset_returns = returns.loc[aligned.index]
    covs = {k: fn(asset_returns) for k, fn in ESTIMATORS.items()}
    cov = covs.get(req.estimator, covs["ledoit_wolf"])
    euler = euler_decomposition(weights, cov)
    var_dec = variance_decomposition(att.betas, aligned[names].cov(), att.resid.var(ddof=1))

    counts, edges = np.histogram(r_p.values, bins=60)
    dd = RM.drawdown(r_p)
    rb = rolling_betas(r_x, aligned[names]).iloc[::5] if att.n_obs > 300 else pd.DataFrame()

    return {
        "meta": {
            "trading_days": att.n_obs,
            "start": _dates(aligned.index)[0],
            "end": _dates(aligned.index)[-1],
            "skipped": skipped,
            "hac_lag": att.hac_lag,
            "estimator": req.estimator if req.estimator in covs else "ledoit_wolf",
        },
        "metrics": {
            "r2": _f(att.r2), "adj_r2": _f(att.adj_r2),
            "alpha_ann": _f(att.alpha_ann), "alpha_tstat": _f(att.alpha_tstat),
            "vol_ann": _f(RM.volatility(r_p)), "sharpe": _f(RM.sharpe(r_x)),
            "var95_1d": _f(TR.cornish_fisher_var(r_p, ALPHA)),
            "es95_1d": _f(TR.historical_es(r_p, ALPHA)),
            "effective_bets": _f(STR.effective_bets(cov)),
            "n_assets": int(len(weights)),
        },
        "weights": {k: _f(v) for k, v in weights.items()},
        "exposures": [{"factor": f, "beta": _f(att.betas[f]), "tstat": _f(att.tstats[f])}
                      for f in names],
        "risk_shares": [{"factor": "IDIO" if f == "IDIOSYNCRATIC" else f, "share": _f(v)}
                        for f, v in var_dec["share"].items()],
        "asset_risk": [{"asset": a, "weight": _f(euler.loc[a, "weight"]),
                        "share": _f(euler.loc[a, "pct_risk"]),
                        "mcr": _f(euler.loc[a, "mcr"])} for a in euler.index],
        "rolling_betas": {"dates": _dates(rb.index),
                          "series": {c: _series(rb[c]) for c in rb.columns}} if len(rb)
        else {"dates": [], "series": {}},
        "performance": {
            "annual_return": _f(RM.annual_return(r_p)),
            "sortino": _f(RM.sortino(r_x)), "calmar": _f(RM.calmar(r_p)),
            "ulcer": _f(RM.ulcer_index(r_p)),
            "information_ratio": _f(RM.information_ratio(r_p, aligned["MKT"])),
            "treynor": _f(RM.treynor(r_x, float(att.betas.get("MKT", np.nan)))),
            "max_drawdown": _f(RM.max_drawdown(r_p)),
            "drawdown_days": int(RM.drawdown_duration(r_p)),
            "skew": _f(r_p.skew()), "kurtosis": _f(r_p.kurtosis()),
            "drawdown_curve": {"dates": _dates(dd.index[::3]), "values": _series(dd.iloc[::3])},
            "wealth_curve": {"dates": _dates(r_p.index[::3]),
                             "values": _series(RM.wealth(r_p).iloc[::3])},
            "return_hist": {"bins": [_f(e) for e in edges[:-1]], "counts": counts.tolist()},
            "rolling_vol": {"dates": _dates(RM.rolling_volatility(r_p).index[::3]),
                            "values": _series(RM.rolling_volatility(r_p).iloc[::3])},
        },
        "var": {
            "levels": [
                {"alpha": a, "label": f"{1 - a:.0%}",
                 "historical": _f(TR.historical_var(r_p, a)),
                 "gaussian": _f(TR.gaussian_var(r_p, a)),
                 "cornish_fisher": _f(TR.cornish_fisher_var(r_p, a)),
                 "es_historical": _f(TR.historical_es(r_p, a)),
                 "es_gaussian": _f(TR.gaussian_es(r_p, a))}
                for a in (0.05, 0.01)
            ],
            "historical": _f(TR.historical_var(r_p, ALPHA)),
            "gaussian": _f(TR.gaussian_var(r_p, ALPHA)),
            "cornish_fisher": _f(TR.cornish_fisher_var(r_p, ALPHA)),
            "jarque_bera_p": _f(DG.jarque_bera(r_p)["p_value"]),
        },
        "covariance": {
            "estimators": list(ESTIMATORS),
            "labels": list(asset_returns.columns),
            "by_estimator": {
                key: {
                    "correlation": [[_f(v) for v in row] for row in
                                    _corr(matrix).values.tolist()],
                    "condition_number": _f(check_condition(matrix)),
                    "vol_ann": _f(np.sqrt(annualise_cov(
                        weights.values @ np.asarray(matrix) @ weights.values))),
                    "effective_bets": _f(STR.effective_bets(matrix)),
                    "risk_shares": [{"asset": a, "share": _f(s)} for a, s in
                                    euler_decomposition(weights, matrix)["pct_risk"].items()],
                } for key, matrix in covs.items()
            },
        },
        "construction": _construction(weights, cov, asset_returns),
        "structure": _structure(cov, aligned[names], weights),
        "diagnostics": _diagnostics(att.resid, aligned[names]),
    }


def _corr(cov: pd.DataFrame) -> pd.DataFrame:
    d = np.sqrt(np.diag(np.asarray(cov)))
    return pd.DataFrame(np.asarray(cov) / np.outer(d, d), index=cov.index, columns=cov.columns)


def _construction(weights: pd.Series, cov: pd.DataFrame, returns: pd.DataFrame) -> dict:
    books = {"Current": weights,
             "Min-Variance": OPT.min_variance(cov, long_only=True),
             "Risk-Parity": OPT.risk_parity(cov),
             "Max-Diversification": OPT.max_diversification(cov)}
    paths, portfolios = {}, []
    for name, w in books.items():
        d = euler_decomposition(w, cov)
        portfolios.append({
            "name": name,
            "weights": {k: _f(v) for k, v in w.items()},
            "risk_contrib": {k: _f(v) for k, v in d["pct_risk"].items()},
            "vol_ann": _f(np.sqrt(annualise_cov(d["portfolio_vol"].iloc[0] ** 2))),
            "diversification_ratio": _f(OPT.diversification_ratio(w, cov)),
            "max_risk_share": _f(d["pct_risk"].max()),
        })
        paths[name] = _series(RM.wealth(D.portfolio_return(returns, w)).iloc[::3])
    return {"portfolios": portfolios,
            "backtest": {"dates": _dates(returns.index[::3]), "paths": paths}}


def _structure(cov: pd.DataFrame, factors: pd.DataFrame, weights: pd.Series) -> dict:
    spectrum, loadings = STR.pca(cov)
    return {
        "pca_scree": [_f(v) for v in spectrum["explained"]],
        "pca_cumulative": [_f(v) for v in spectrum["cumulative"]],
        "components": list(spectrum.index),
        "loadings": {"assets": list(loadings.index),
                     "matrix": [[_f(v) for v in row] for row in loadings.values.tolist()]},
        "effective_bets": _f(STR.effective_bets(cov)),
        "n_assets": int(len(weights)),
        "vif": [{"factor": f, "vif": _f(v)} for f, v in STR.vif(factors).items()],
        "factor_correlation": {"labels": list(factors.columns),
                               "matrix": [[_f(v) for v in row]
                                          for row in factors.corr().values.tolist()]},
    }


def _diagnostics(resid: pd.Series, factors: pd.DataFrame) -> List[dict]:
    table = DG.diagnostics_table(resid, factors)
    return [{"name": name, "stat": _f(row["statistic"]), "pvalue": _f(row["p_value"]),
             "verdict": row["conclusion"], "licenses": row["licenses"]}
            for name, row in table.iterrows()]
