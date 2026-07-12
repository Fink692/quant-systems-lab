import numpy as np
import pandas as pd
import pytest

from quantlab.risk.macro import fit_macro_factor_model, macro_surprise_factors
from quantlab.workflows.demo_suite import run_full_demo


def test_macro_surprise_factors_standardize_indicator_changes():
    dates = pd.date_range("2020-01-01", periods=6)
    indicators = pd.DataFrame(
        {
            "growth": [100.0, 101.0, 100.5, 102.0, 103.0, 102.5],
            "rates": [2.0, 2.1, 2.0, 2.2, 2.4, 2.3],
        },
        index=dates,
    )

    factors = macro_surprise_factors(indicators, transform="diff", standardize=True)

    assert factors.shape == (5, 2)
    assert np.allclose(factors.mean(axis=0), 0.0)
    assert np.allclose(factors.std(axis=0, ddof=1), 1.0)


def test_macro_factor_model_recovers_exposures_and_stress_pnl():
    dates = pd.date_range("2021-01-01", periods=80, freq="B")
    rng = np.random.default_rng(12)
    macro_factors = pd.DataFrame(
        rng.normal(size=(80, 3)),
        index=dates,
        columns=["growth", "inflation", "rates"],
    )
    betas = pd.DataFrame(
        {
            "growth": [0.8, -0.4, 0.2, 0.0],
            "inflation": [-0.2, 0.5, 0.1, -0.3],
            "rates": [-0.6, 0.1, 0.4, 0.7],
        },
        index=["A", "B", "C", "D"],
    )
    asset_returns = macro_factors @ betas.T + 0.001 * rng.normal(size=(80, 4))

    result = fit_macro_factor_model(asset_returns, macro_factors, ridge=1e-8)
    weights = pd.Series([0.4, 0.3, 0.2, 0.1], index=asset_returns.columns)
    shocks = pd.Series({"growth": -1.0, "inflation": 0.5, "rates": 0.25})

    assert np.allclose(result.exposures.loc["A"], betas.loc["A"], atol=2e-3)
    assert result.r_squared.mean() > 0.999
    assert result.covariance_matrix().shape == (4, 4)
    assert np.isclose(
        result.stress_pnl(weights, shocks, portfolio_value=100.0), 100.0 * (result.portfolio_exposure(weights) @ shocks)
    )


def test_macro_factor_model_rejects_missing_shock_coverage():
    dates = pd.date_range("2022-01-01", periods=10)
    asset_returns = pd.DataFrame(np.random.default_rng(1).normal(size=(10, 2)), index=dates, columns=["A", "B"])
    macro_factors = pd.DataFrame(np.random.default_rng(2).normal(size=(10, 1)), index=dates, columns=["growth"])
    result = fit_macro_factor_model(asset_returns, macro_factors)

    with pytest.raises(ValueError, match="factor_shocks"):
        result.stress_pnl(pd.Series([0.5, 0.5], index=["A", "B"]), pd.Series({"rates": 1.0}))


def test_full_demo_exposes_macro_factor_metrics():
    result = run_full_demo(seed=33).as_dict()

    assert result["factor_risk"]["macro_factor_count"] == 3
    assert 0.0 <= result["factor_risk"]["macro_average_r_squared"] <= 1.0
    assert np.isfinite(result["factor_risk"]["macro_stress_pnl"])
