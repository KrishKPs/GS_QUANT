"""Quant Portfolio Analytics — Streamlit UI only. Every formula lives in core/.

Built on gs-quant (`gs_quant.timeseries`) for the return/risk primitives, with
HAC inference, Euler risk decomposition, shrinkage, Cornish-Fisher and the
optimisers layered on top. Nothing here forecasts anything.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import data as D
from core import diagnostics as DG
from core import optimise as OPT
from core import risk_metrics as RM
from core import structure as STR
from core import tail_risk as TR
from core.attribution import factor_regression, rolling_betas
from core.conventions import PERIODS_PER_YEAR, annualise_cov
from core.covariance import ESTIMATORS, check_condition
from core.risk_decomp import euler_decomposition, variance_decomposition

PALETTE = {  # identical factor colours in every chart
    "MKT": "#1f77b4", "SMB": "#ff7f0e", "HML": "#2ca02c",
    "RMW": "#d62728", "MOM": "#9467bd", "IDIOSYNCRATIC": "#7f7f7f",
}
NOT_A_FORECAST = "Measured on realised history. This is attribution, not prediction."

st.set_page_config(page_title="Portfolio Attribution", layout="wide")


@st.cache_data(show_spinner="Downloading prices…")
def get_prices(tickers, start, end):
    return D.load_prices(tickers, start, end)


@st.cache_data(show_spinner="Downloading Ken French factors…")
def get_factors(start, end):
    return D.load_factors(start, end)


def parse_portfolio(text: str) -> pd.Series:
    """'AAPL 0.3' or 'AAPL, 0.3' per line -> weights Series."""
    rows = {}
    for line in text.strip().splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) == 2:
            rows[parts[0].upper()] = float(parts[1])
    if not rows:
        raise ValueError("Enter one 'TICKER WEIGHT' per line.")
    return pd.Series(rows, name="weight")


def bar(series: pd.Series, title: str, pct: bool = False, colours: bool = False):
    fig = go.Figure(go.Bar(
        x=list(series.index), y=series.values,
        marker_color=[PALETTE.get(i, "#4c78a8") for i in series.index] if colours else "#4c78a8",
    ))
    fig.update_layout(title=title, height=340, margin=dict(t=45, b=10),
                      yaxis_tickformat=".1%" if pct else None)
    return fig


with st.sidebar:
    st.header("Portfolio")
    holdings = st.text_area("TICKER WEIGHT per line",
                            "AAPL 0.25\nMSFT 0.20\nJPM 0.20\nXOM 0.20\nJNJ 0.15", height=160)
    start = st.date_input("Start", dt.date(2018, 1, 1))
    end = st.date_input("End", dt.date.today())
    estimator = st.selectbox("Covariance estimator", list(ESTIMATORS), index=1)
    alpha_lvl = st.select_slider("VaR confidence", [0.90, 0.95, 0.99], value=0.95)
    winsorise = st.checkbox("Winsorise returns (3σ, gs-quant)", value=False,
                            help="Explicit robustness toggle — off by default, never silent.")
    run = st.button("Analyse", type="primary")

st.title("Portfolio Attribution & Risk Decomposition")
st.caption(
    "Where the return came from and where the risk actually lives. Powered by "
    "**gs-quant** `timeseries` primitives; closed-form maths and classical statistics only."
)

if not run:
    st.info("Set holdings in the sidebar and hit **Analyse**.")
    st.stop()

# ---------------------------------------------------------------- pipeline
weights = parse_portfolio(holdings)
prices = get_prices(list(weights.index), start, end)
returns = D.to_returns(prices, winsorise=winsorise)
weights = D.align_weights(returns, weights)
port = D.portfolio_return(returns, weights)
aligned = D.align(port, get_factors(start, end))
names = D.factor_names(aligned)
r_p, r_x = aligned["PORT"], aligned["PORT_EXCESS"]

att = factor_regression(r_x, aligned[names])
cov = ESTIMATORS[estimator](returns.loc[aligned.index])
euler = euler_decomposition(weights, cov)
var_dec = variance_decomposition(att.betas, aligned[names].cov(), att.resid.var(ddof=1))
tail_alpha = round(1 - alpha_lvl, 2)
n_eff = STR.effective_bets(cov)

tabs = st.tabs(["Overview", "Attribution", "Risk & Performance", "Tail Risk",
                "Covariance", "Construction", "Structure", "Diagnostics"])

# ---------------------------------------------------------------- 1 overview
with tabs[0]:
    st.caption(f"Headline numbers. {NOT_A_FORECAST}")
    c = st.columns(6)
    c[0].metric("Annualised vol", f"{RM.volatility(r_p):.2%}")
    c[1].metric("Sharpe (ex-post)", f"{RM.sharpe(r_x):.2f}")
    c[2].metric("Annualised α", f"{att.alpha_ann:.2%}", f"t = {att.alpha_tstat:.2f}")
    c[3].metric("R²", f"{att.r2:.1%}")
    c[4].metric(f"VaR {alpha_lvl:.0%} (1d)", f"{TR.cornish_fisher_var(r_p, tail_alpha):.2%}")
    c[5].metric("Effective bets", f"{n_eff:.2f}", f"of {len(weights)} names")

    top = euler["pct_risk"].idxmax()
    st.warning(
        f"**Hidden concentration:** {top} carries {euler.loc[top, 'pct_risk']:.0%} of "
        f"portfolio volatility on a {weights[top]:.0%} weight, and the book's "
        f"{len(weights)} names amount to {n_eff:.1f} effective independent bets. "
        "Weights are not risk."
    )
    left, right = st.columns(2)
    left.plotly_chart(bar(euler["pct_risk"], "Share of portfolio volatility", pct=True))
    right.plotly_chart(bar(var_dec["share"], "Share of portfolio variance", pct=True, colours=True))
    st.plotly_chart(px.line(RM.wealth(r_p), title="Realised wealth path")
                    .update_layout(height=300, showlegend=False))

# ---------------------------------------------------------------- 2 attribution
with tabs[1]:
    st.caption(
        f"Betas are bet size; risk contributions are how much each bet drives variance. "
        f"HAC (Newey–West, lag {att.hac_lag}) t-stats — |t| ≥ 2 is reliable. {NOT_A_FORECAST}"
    )
    s = att.summary
    fig = go.Figure(go.Bar(x=list(s.index), y=s["beta"],
                           marker_color=[PALETTE.get(f, "#333") for f in s.index],
                           text=[f"t={t:.1f}" for t in s["t_stat"]], textposition="outside"))
    fig.update_layout(title=f"Factor exposures — R² {att.r2:.1%}, adj R² {att.adj_r2:.1%}",
                      yaxis_title="beta", height=380)
    st.plotly_chart(fig)
    st.dataframe(s.style.format({"beta": "{:.3f}", "t_stat": "{:.2f}", "p_value": "{:.3f}"}),
                 width="stretch")

    a, b = st.columns(2)
    with a:
        st.subheader("Variance decomposition")
        st.caption("Σ CCV_i = βᵀΣ_Fβ; cross-terms carry factor correlation. Shares sum to 100%.")
        st.plotly_chart(bar(var_dec["share"], "% of total variance", pct=True, colours=True))
    with b:
        st.subheader("Volatility (Euler) decomposition")
        st.caption("Σ CCR_i = σ_p exactly — σ_p is homogeneous of degree 1 in w.")
        st.plotly_chart(bar(euler["pct_risk"], "% of portfolio vol", pct=True))
        st.dataframe(euler[["weight", "mcr", "ccr", "pct_risk"]].style.format("{:.4f}"),
                     width="stretch")

    if att.n_obs > 300:
        st.subheader("Rolling betas (252d)")
        st.caption("Exposures drift — one full-sample beta hides that.")
        st.plotly_chart(px.line(rolling_betas(r_x, aligned[names]), color_discrete_map=PALETTE)
                        .update_layout(height=340))

# ---------------------------------------------------------------- 3 risk & perf
with tabs[2]:
    st.caption(f"Ex-post performance statistics. {NOT_A_FORECAST}")
    mkt_beta = float(att.betas.get("MKT", np.nan))
    summary = RM.performance_summary(r_p, excess=r_x, benchmark=aligned["MKT"],
                                     market_beta=mkt_beta)
    c = st.columns(5)
    c[0].metric("Sharpe", f"{summary['sharpe']:.2f}")
    c[1].metric("Sortino", f"{summary['sortino']:.2f}")
    c[2].metric("Information ratio", f"{summary['information_ratio']:.2f}")
    c[3].metric("Calmar", f"{summary['calmar']:.2f}")
    c[4].metric("Max drawdown", f"{summary['max_drawdown']:.2%}")
    st.dataframe(summary.to_frame().style.format("{:.4f}"), width="stretch")
    st.caption("Sharpe is annualised ×√252, never ×252 — and it assumes normality, "
               f"which the skew ({summary['skew']:.2f}) and excess kurtosis "
               f"({summary['excess_kurtosis']:.2f}) below argue against.")

    a, b = st.columns(2)
    a.plotly_chart(px.area(RM.drawdown(r_p), title="Drawdown")
                   .update_layout(height=320, showlegend=False, yaxis_tickformat=".0%"))
    b.plotly_chart(px.line(RM.rolling_volatility(r_p), title="Rolling 126d volatility (gs-quant)")
                   .update_layout(height=320, showlegend=False, yaxis_tickformat=".0%"))

# ---------------------------------------------------------------- 4 tail risk
with tabs[3]:
    st.caption("Losses shown positive, 1-day horizon. Quantiles and moments of the "
               "realised distribution — no model is fitted.")
    table = TR.tail_table(r_p, alphas=(0.01, 0.05))
    st.dataframe(table.style.format("{:.2%}"), width="stretch")

    jb = DG.jarque_bera(r_p)
    st.info(
        f"Jarque–Bera p = {jb['p_value']:.3g} → returns are "
        f"**{'not ' if not jb['normal'] else ''}normal**. "
        + ("Trust the historical and Cornish–Fisher numbers over the Gaussian ones: "
           "Gaussian VaR understates fat left tails." if not jb["normal"] else
           "Gaussian VaR is defensible here.")
        + " ES is coherent (sub-additive) where VaR is not."
    )
    hist = go.Figure(go.Histogram(x=r_p, nbinsx=120, marker_color="#4c78a8"))
    for label, colour in [("VaR_historical", "#2ca02c"), ("VaR_gaussian", "#ff7f0e"),
                          ("VaR_cornish_fisher", "#d62728"), ("ES_historical", "#9467bd")]:
        hist.add_vline(x=-table.loc[label, f"{alpha_lvl:.0%}"], line_dash="dash",
                       line_color=colour, annotation_text=label.replace("_", " "),
                       annotation_position="top")
    hist.update_layout(title=f"Daily return distribution with {alpha_lvl:.0%} thresholds",
                       height=420, showlegend=False)
    st.plotly_chart(hist)

# ---------------------------------------------------------------- 5 covariance
with tabs[4]:
    st.caption("The covariance matrix drives risk decomposition, VaR and every optimiser, "
               "so how it is estimated matters. These are estimators of present structure.")
    rows, decomps = [], {}
    for label, fn in ESTIMATORS.items():
        est = fn(returns.loc[aligned.index])
        d = euler_decomposition(weights, est)
        decomps[label] = d["pct_risk"]
        rows.append({"estimator": label,
                     "portfolio vol (ann.)": float(np.sqrt(annualise_cov(
                         weights.values @ np.asarray(est) @ weights.values))),
                     "condition number": check_condition(est),
                     "effective bets": STR.effective_bets(est)})
    st.dataframe(pd.DataFrame(rows).set_index("estimator").style.format(
        {"portfolio vol (ann.)": "{:.2%}", "condition number": "{:.3g}",
         "effective bets": "{:.2f}"}), width="stretch")
    st.caption("Ledoit–Wolf trades a little bias for a large variance reduction and "
               "guarantees invertibility; EWMA (λ=0.94) weights recent structure more heavily.")
    st.plotly_chart(px.bar(pd.DataFrame(decomps), barmode="group",
                           title="Risk shares under each estimator")
                    .update_layout(height=360, yaxis_tickformat=".0%"))
    st.plotly_chart(px.imshow(returns.corr(), text_auto=".2f", color_continuous_scale="RdBu_r",
                              zmin=-1, zmax=1, title="Asset correlation")
                    .update_layout(height=420))

# ---------------------------------------------------------------- 6 construction
with tabs[5]:
    st.caption("Deterministic constructions — given Σ these are convex programs, not "
               "forecasts. No expected return enters (except the clearly labelled frontier).")
    books = {"Current": weights,
             "Min variance": OPT.min_variance(cov, long_only=True),
             "Risk parity": OPT.risk_parity(cov),
             "Max diversification": OPT.max_diversification(cov)}
    stats_rows, share_rows = [], {}
    for label, w in books.items():
        d = euler_decomposition(w, cov)
        share_rows[label] = d["pct_risk"]
        stats_rows.append({"portfolio": label,
                           "ann. vol": float(np.sqrt(annualise_cov(d["portfolio_vol"].iloc[0] ** 2))),
                           "diversification ratio": OPT.diversification_ratio(w, cov),
                           "effective positions": float(1 / (w ** 2).sum()),
                           "max risk share": float(d["pct_risk"].max())})
    st.dataframe(pd.DataFrame(stats_rows).set_index("portfolio").style.format(
        {"ann. vol": "{:.2%}", "diversification ratio": "{:.2f}",
         "effective positions": "{:.2f}", "max risk share": "{:.1%}"}),
        width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(px.bar(pd.DataFrame(books), barmode="group", title="Weights")
                   .update_layout(height=360, yaxis_tickformat=".0%"))
    b.plotly_chart(px.bar(pd.DataFrame(share_rows), barmode="group",
                          title="Risk contributions (Euler)")
                   .update_layout(height=360, yaxis_tickformat=".0%"))
    st.caption("Risk parity equalises the Euler contributions from §1.4 — that is exactly "
               "why the decomposition has to sum to σ_p for the problem to be well posed.")

    with st.expander("Ex-post efficient frontier (descriptive only)"):
        st.warning("Uses **historical mean** returns, a weak proxy for expected return. "
                   "This draws the realised trade-off curve; it forecasts nothing.")
        ef = OPT.efficient_frontier(returns.loc[aligned.index].mean(), cov, 25)
        if not ef.empty:
            st.plotly_chart(px.line(ef, x="vol", y="target_return", markers=True)
                            .update_layout(height=340))

# ---------------------------------------------------------------- 7 structure
with tabs[6]:
    st.caption("Pure linear algebra on structure that already exists in the sample.")
    spectrum, loadings = STR.pca(cov)
    c = st.columns(3)
    c[0].metric("Effective bets (N_eff)", f"{n_eff:.2f}", f"of {len(weights)} names")
    c[1].metric("PC1 explains", f"{spectrum['explained'].iloc[0]:.1%}")
    c[2].metric("PCs for 90%", int((spectrum["cumulative"] < 0.9).sum() + 1))
    a, b = st.columns(2)
    scree = go.Figure([go.Bar(x=list(spectrum.index), y=spectrum["explained"], name="explained"),
                       go.Scatter(x=list(spectrum.index), y=spectrum["cumulative"],
                                  name="cumulative", yaxis="y")])
    scree.update_layout(title="PCA scree", height=340, yaxis_tickformat=".0%")
    a.plotly_chart(scree)
    b.plotly_chart(px.imshow(loadings, text_auto=".2f", color_continuous_scale="RdBu_r",
                             title="Eigenvector loadings").update_layout(height=340))
    st.caption(f"1 ≤ N_eff ≤ n by construction. {len(weights)} names, {n_eff:.1f} real bets.")

    a, b = st.columns(2)
    v = STR.vif(aligned[names])
    a.plotly_chart(bar(v, "Factor VIF (>5 destabilises betas)"))
    b.plotly_chart(px.imshow(aligned[names].corr(), text_auto=".2f",
                             color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                             title="Factor correlation").update_layout(height=340))
    if (v > 5).any():
        st.warning(f"High VIF on {', '.join(v[v > 5].index)} — read those betas as a pair, "
                   "not individually.")

# ---------------------------------------------------------------- 8 diagnostics
with tabs[7]:
    st.caption("Each test's result licenses a modelling choice made elsewhere in the app.")
    st.dataframe(DG.diagnostics_table(att.resid, aligned[names]).style.format(
        {"statistic": "{:.2f}", "p_value": "{:.4f}"}), width="stretch")
    st.markdown(
        f"""
- **Ljung–Box** on residuals → autocorrelation → HAC (Newey–West) errors at lag {att.hac_lag}
  in the Attribution tab, instead of overconfident plain OLS t-stats.
- **Jarque–Bera** on residuals → non-normality → Cornish–Fisher VaR and Expected Shortfall
  in the Tail Risk tab, instead of Gaussian VaR alone.
- **ADF** on each factor → stationarity → the factor series are legitimate regressors.
- **Sample size** — {att.n_obs} observations (the engine refuses fewer than 60).

**Biggest limitation:** everything is estimated on a historical window, so it is
backward-looking. Exposures drift (see rolling betas) and short windows make every
estimate noisier. Stated openly rather than hidden.
"""
    )
