"""Synthetic data with analytic ground truth: does the estimator recover it?"""
import numpy as np
import pandas as pd
import pytest

from core.attribution import factor_regression
from core.conventions import PERIODS_PER_YEAR

TRUE_BETAS = {"MKT": 1.20, "SMB": -0.30, "HML": 0.45}
TRUE_ALPHA_DAILY = 0.0002


@pytest.fixture
def planted():
    rng = np.random.default_rng(0)  # seed: randomness only in the test fixture
    T = 2000
    idx = pd.bdate_range("2015-01-01", periods=T)
    F = pd.DataFrame(rng.normal(0, 0.01, (T, 3)), index=idx, columns=TRUE_BETAS)
    eps = rng.normal(0, 0.002, T)
    y = TRUE_ALPHA_DAILY + F.values @ np.array(list(TRUE_BETAS.values())) + eps
    return pd.Series(y, index=idx, name="PORT_EXCESS"), F


def test_recovers_planted_betas(planted):
    y, F = planted
    att = factor_regression(y, F)
    for name, truth in TRUE_BETAS.items():
        assert att.betas[name] == pytest.approx(truth, abs=0.02)
        assert abs(att.tstats[name]) >= 2  # planted exposures are real


def test_recovers_planted_alpha(planted):
    y, F = planted
    att = factor_regression(y, F)
    assert att.alpha_daily == pytest.approx(TRUE_ALPHA_DAILY, abs=2e-4)
    assert att.alpha_ann == pytest.approx(att.alpha_daily * PERIODS_PER_YEAR)


def test_r2_high_when_noise_is_small(planted):
    y, F = planted
    assert factor_regression(y, F).r2 > 0.95


def test_short_sample_is_refused(planted):
    y, F = planted
    with pytest.raises(ValueError, match="at least 60"):
        factor_regression(y.iloc[:30], F.iloc[:30])


def test_gs_quant_volatility_matches_the_textbook_formula():
    """gs-quant is the primary engine; check it against sigma*sqrt(252) directly."""
    from core import gsq
    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(0, 0.01, 1000), index=pd.bdate_range("2019-01-01", periods=1000))
    assert gsq.volatility(r) == pytest.approx(float(r.std(ddof=1) * np.sqrt(252)), rel=5e-3)


def test_gs_quant_and_statsmodels_agree_on_betas(planted):
    """The Attribution object asserts this internally; pin it as a test too."""
    import statsmodels.api as sm
    from core import gsq
    y, F = planted
    gs_betas, gs_alpha, gs_r2, _ = gsq.ols(y, F)
    sm_fit = sm.OLS(y, sm.add_constant(F)).fit()
    assert np.allclose(gs_betas.values, sm_fit.params[list(F.columns)].values, atol=1e-10)
    assert gs_alpha == pytest.approx(sm_fit.params["const"], abs=1e-12)
    assert gs_r2 == pytest.approx(sm_fit.rsquared, abs=1e-10)


def test_drawdown_on_a_known_path():
    """100 -> 120 -> 60 -> 90: max drawdown is -50%, spent 2 periods under water."""
    from core.risk_metrics import calmar, drawdown_duration, max_drawdown, ulcer_index
    r = pd.Series([0.2, -0.5, 0.5], index=pd.bdate_range("2020-01-01", periods=3))
    assert max_drawdown(r) == pytest.approx(-0.5)
    assert drawdown_duration(r) == 2
    assert ulcer_index(r) == pytest.approx(np.sqrt(np.mean([0.0, 0.5 ** 2, 0.25 ** 2])))
    assert calmar(r) == pytest.approx((0.9 ** (252 / 3) - 1) / 0.5)


def test_sharpe_annualises_with_sqrt_252_not_252():
    from core.risk_metrics import sharpe
    rng = np.random.default_rng(9)
    r = pd.Series(rng.normal(0.0004, 0.01, 800))
    daily = r.mean() / r.std(ddof=1)
    assert sharpe(r) == pytest.approx(daily * np.sqrt(252))


def test_min_variance_closed_form_matches_the_solver():
    from core.optimise import min_variance
    rng = np.random.default_rng(6)
    x = rng.normal(0, 0.01, (400, 4))
    cov = pd.DataFrame(np.cov(x.T), index=list("ABCD"), columns=list("ABCD"))
    closed = min_variance(cov)                      # Sigma^-1 1 / (1' Sigma^-1 1)
    assert closed.sum() == pytest.approx(1.0)
    v_closed = float(closed @ cov @ closed)
    v_long = float(min_variance(cov, True) @ cov @ min_variance(cov, True))
    assert v_closed <= v_long + 1e-12               # unconstrained is never worse
