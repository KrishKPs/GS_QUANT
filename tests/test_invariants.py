"""Mathematical guarantees that must hold for *any* input."""
import numpy as np
import pandas as pd
import pytest

from core.attribution import factor_regression
from core.covariance import nearest_psd, sample_cov
from core.risk_decomp import euler_decomposition, portfolio_vol, variance_decomposition

ATOL, RTOL = 1e-8, 1e-6


def random_returns(rng, n=6, T=500):
    names = [f"A{i}" for i in range(n)]
    load = rng.normal(0, 1, (n, 3))
    common = rng.normal(0, 0.01, (T, 3))
    r = common @ load.T + rng.normal(0, 0.005, (T, n))
    return pd.DataFrame(r, index=pd.bdate_range("2018-01-01", periods=T), columns=names)


@pytest.mark.parametrize("seed", range(5))
def test_euler_contributions_sum_to_portfolio_vol(seed):
    rng = np.random.default_rng(seed)
    r = random_returns(rng)
    w = pd.Series(rng.normal(0, 1, r.shape[1]), index=r.columns)
    w /= w.sum()
    cov = sample_cov(r)
    d = euler_decomposition(w, cov)
    assert d["ccr"].sum() == pytest.approx(portfolio_vol(w, cov), abs=ATOL, rel=RTOL)
    assert d["pct_risk"].sum() == pytest.approx(1.0, abs=ATOL, rel=RTOL)


@pytest.mark.parametrize("seed", range(5))
def test_variance_shares_sum_to_one(seed):
    rng = np.random.default_rng(seed)
    F = pd.DataFrame(rng.normal(0, 0.01, (400, 4)),
                     index=pd.bdate_range("2018-01-01", periods=400),
                     columns=["MKT", "SMB", "HML", "MOM"])
    beta = pd.Series(rng.normal(0, 0.8, 4), index=F.columns)
    y = pd.Series(F.values @ beta.values + rng.normal(0, 0.004, 400), index=F.index)
    att = factor_regression(y, F)
    d = variance_decomposition(att.betas, F.cov(), att.resid.var(ddof=1))
    systematic = float(att.betas.values @ nearest_psd(F.cov()).values @ att.betas.values)
    ccv = d["variance_contribution"].drop("IDIOSYNCRATIC")
    assert ccv.sum() == pytest.approx(systematic, abs=ATOL, rel=RTOL)
    assert d["share"].sum() == pytest.approx(1.0, abs=ATOL, rel=RTOL)


def test_ols_variance_splits_into_fitted_plus_residual():
    rng = np.random.default_rng(7)
    F = pd.DataFrame(rng.normal(0, 0.01, (300, 2)),
                     index=pd.bdate_range("2019-01-01", periods=300), columns=["MKT", "SMB"])
    y = pd.Series(F["MKT"] * 0.9 + rng.normal(0, 0.003, 300), index=F.index)
    att = factor_regression(y, F)
    total = np.var(y.values, ddof=0)
    split = np.var(att.fitted.values, ddof=0) + np.var(att.resid.values, ddof=0)
    assert split == pytest.approx(total, abs=ATOL, rel=RTOL)
    assert att.resid.mean() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_nearest_psd_has_no_negative_eigenvalues(seed):
    rng = np.random.default_rng(seed)
    m = rng.normal(0, 1, (8, 8))
    m = (m + m.T) / 2  # symmetric but indefinite
    assert np.linalg.eigvalsh(np.asarray(nearest_psd(m))).min() >= -ATOL


def test_nearest_psd_is_identity_on_a_psd_matrix():
    rng = np.random.default_rng(3)
    cov = sample_cov(random_returns(rng))
    assert np.allclose(np.asarray(nearest_psd(cov)), np.asarray(cov), atol=ATOL)


def test_cornish_fisher_reduces_to_gaussian_without_skew_or_kurtosis():
    from core.tail_risk import cornish_fisher_z
    from scipy import stats
    for a in (0.01, 0.05, 0.10):
        assert cornish_fisher_z(a, 0.0, 0.0) == pytest.approx(stats.norm.ppf(a), abs=ATOL)


def test_expected_shortfall_is_at_least_var():
    from core.tail_risk import gaussian_es, gaussian_var, historical_es, historical_var
    rng = np.random.default_rng(11)
    r = pd.Series(rng.standard_t(4, 2000) * 0.01)
    for a in (0.01, 0.05):
        assert historical_es(r, a) >= historical_var(r, a)
        assert gaussian_es(r, a) >= gaussian_var(r, a)


@pytest.mark.parametrize("seed", range(3))
def test_min_variance_beats_every_random_long_only_portfolio(seed):
    from core.optimise import min_variance
    rng = np.random.default_rng(seed)
    cov = sample_cov(random_returns(rng, n=7))
    w = min_variance(cov, long_only=True)
    best = float(w.values @ np.asarray(cov) @ w.values)
    for _ in range(300):
        x = rng.random(len(w))
        x /= x.sum()
        assert best <= float(x @ np.asarray(cov) @ x) + 1e-12


@pytest.mark.parametrize("seed", range(3))
def test_risk_parity_equalises_component_risk(seed):
    from core.optimise import risk_parity
    rng = np.random.default_rng(seed)
    cov = sample_cov(random_returns(rng, n=6))
    ccr = euler_decomposition(risk_parity(cov), cov)["ccr"]
    assert ccr.std() / ccr.mean() < 1e-6
    assert (ccr > 0).all()


@pytest.mark.parametrize("seed", range(3))
def test_effective_bets_between_one_and_n(seed):
    from core.structure import effective_bets
    rng = np.random.default_rng(seed)
    r = random_returns(rng, n=9)
    n_eff = effective_bets(sample_cov(r))
    assert 1.0 - ATOL <= n_eff <= r.shape[1] + ATOL


def test_effective_bets_hits_the_bounds():
    from core.structure import effective_bets
    n = 5
    identical = pd.DataFrame(np.ones((n, n)) * 0.04)          # one shared bet
    independent = pd.DataFrame(np.eye(n) * 0.04)              # n equal bets
    assert effective_bets(identical) == pytest.approx(1.0, abs=1e-8)
    assert effective_bets(independent) == pytest.approx(n, abs=1e-8)


@pytest.mark.parametrize("seed", range(3))
def test_ledoit_wolf_is_symmetric_psd_and_invertible(seed):
    from core.covariance import ledoit_wolf_cov
    rng = np.random.default_rng(seed)
    cov = ledoit_wolf_cov(random_returns(rng, n=12, T=80))  # n close to T: sample is fragile
    a = np.asarray(cov)
    assert np.allclose(a, a.T, atol=ATOL)
    assert np.linalg.eigvalsh(a).min() > 0
    assert np.allclose(a @ np.linalg.inv(a), np.eye(len(a)), atol=1e-6)


def test_max_diversification_beats_equal_weight_on_the_diversification_ratio():
    from core.optimise import diversification_ratio, max_diversification
    rng = np.random.default_rng(5)
    cov = sample_cov(random_returns(rng, n=6))
    w_eq = pd.Series(np.ones(len(cov)) / len(cov), index=cov.index)
    assert diversification_ratio(max_diversification(cov), cov) >= diversification_ratio(w_eq, cov)


def test_ewma_and_sample_covariance_are_psd():
    from core.covariance import ewma_cov
    rng = np.random.default_rng(2)
    r = random_returns(rng)
    for cov in (ewma_cov(r), sample_cov(r)):
        assert np.linalg.eigvalsh(np.asarray(cov)).min() >= -ATOL
