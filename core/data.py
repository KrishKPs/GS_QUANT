"""Data layer: prices -> returns, Ken French factors, aligned excess returns.

Pure functions, no caching and no Streamlit (cache at the UI boundary).
"""
from __future__ import annotations

import warnings
from typing import List, Sequence

import numpy as np
import pandas as pd

from . import gsq
from .conventions import normalise_weights

FACTOR_COLS = ["MKT", "SMB", "HML", "RMW", "MOM", "RF"]
_FF5 = "F-F_Research_Data_5_Factors_2x3_daily"
_MOM = "F-F_Momentum_Factor_daily"
_MIN_COVERAGE = 0.9  # drop a ticker with <90% of the best ticker's history


def load_prices(tickers: Sequence[str], start, end) -> pd.DataFrame:
    """Adjusted close prices, columns = tickers, dropna-aligned.

    Tickers with short history (a mid-window IPO) are dropped with a warning
    rather than truncating the whole sample.
    """
    import yfinance as yf

    tickers = list(dict.fromkeys(tickers))
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No price data returned for {tickers}.")
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if px.shape[1] == 1 and len(tickers) == 1:
        px.columns = tickers
    px = px.dropna(axis=1, how="all")

    missing = [t for t in tickers if t not in px.columns]
    if missing:
        warnings.warn(f"No data for: {', '.join(missing)}")
    short = px.columns[px.count() < _MIN_COVERAGE * px.count().max()].tolist()
    if short:
        warnings.warn(f"Dropped short-history tickers: {', '.join(short)}")
        px = px.drop(columns=short)
    if px.empty or px.shape[1] == 0:
        raise ValueError("No usable price series after cleaning.")
    px = px.dropna()
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    px.index.name = "date"
    return px.sort_index()


def to_returns(prices: pd.DataFrame, log: bool = False,
               winsorise: bool = False, limit: float = 3.0) -> pd.DataFrame:
    """Returns via `gs_quant.timeseries.returns`. Simple by default.

    Simple returns aggregate across assets, log returns aggregate across time —
    be consistent if you flip this. Winsorisation is an explicit opt-in
    robustness toggle (gs-quant's `winsorize`), never silent.
    """
    r = gsq.returns(prices, log=log).dropna(how="all")
    return r.apply(lambda c: gsq.winsorize(c, limit)) if winsorise else r


def align_weights(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Restrict weights to the surviving return columns and renormalise to 1."""
    w = pd.Series(weights, dtype=float).reindex(returns.columns).dropna()
    if w.empty:
        raise ValueError("No overlap between weights and return columns.")
    if len(w) < len(weights):
        dropped = set(pd.Series(weights).index) - set(w.index)
        warnings.warn(f"Renormalised weights after dropping: {', '.join(sorted(dropped))}")
    return normalise_weights(w)


def portfolio_return(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Weight-normalised blend of asset returns (fixed weights, no drift)."""
    w = align_weights(returns, weights)
    return returns[w.index].mul(w, axis=1).sum(axis=1).rename("PORT")


def load_factors(start, end) -> pd.DataFrame:
    """Ken French daily factors, decimal (not percent), DatetimeIndex.

    Columns: MKT SMB HML RMW MOM RF. MKT is already the excess market return,
    so the portfolio side must be excess too (see `align`).
    """
    from pandas_datareader.data import DataReader

    def _get(name: str) -> pd.DataFrame:
        df = DataReader(name, "famafrench", start, end)[0]
        df.columns = [c.strip() for c in df.columns]
        df.index = pd.DatetimeIndex(df.index.to_timestamp() if hasattr(df.index, "to_timestamp") else df.index)
        return df / 100.0

    ff = _get(_FF5).rename(columns={"Mkt-RF": "MKT"})
    mom = _get(_MOM).rename(columns={"Mom": "MOM"})
    out = ff.join(mom[["MOM"]], how="inner")  # momentum file spans different dates
    return out[FACTOR_COLS].sort_index()


def align(portfolio: pd.Series, factors: pd.DataFrame) -> pd.DataFrame:
    """Inner-join on dates and add PORT_EXCESS = PORT - RF.

    Always regress excess-on-excess: MKT is already an excess return.
    """
    df = pd.concat([portfolio.rename("PORT"), factors], axis=1, join="inner").dropna()
    df["PORT_EXCESS"] = df["PORT"] - df["RF"]
    return df


def factor_names(aligned: pd.DataFrame) -> List[str]:
    """Regressor columns of an aligned frame (everything but RF and PORT*)."""
    return [c for c in aligned.columns if c not in ("RF", "PORT", "PORT_EXCESS")]
