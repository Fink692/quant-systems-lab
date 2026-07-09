import numpy as np
import pandas as pd

from quantlab.credit.migration import (
    cumulative_default_probability,
    normalize_transition_matrix,
    simulate_rating_paths,
    transition_matrix_power,
)
from quantlab.data.synthetic import synthetic_cointegrated_prices, synthetic_factor_panel
from quantlab.options.black_scholes import black_scholes_price
from quantlab.options.variance_reduction import black_scholes_antithetic_price, black_scholes_control_variate_price
from quantlab.risk.covariance import ewma_covariance, ledoit_wolf_covariance, nearest_positive_semidefinite
from quantlab.stat_arb.dynamic_backtest import backtest_kalman_spread_strategy
from quantlab.workflows.demo_suite import run_full_demo


def test_covariance_estimators_return_psd_matrices():
    panel = synthetic_factor_panel(periods=80, assets=5, factors=2, seed=30)
    ewma = ewma_covariance(panel.asset_returns)
    lw = ledoit_wolf_covariance(panel.asset_returns)
    assert ewma.shape == (5, 5)
    assert lw.shape == (5, 5)
    assert np.min(np.linalg.eigvalsh(lw.to_numpy())) >= -1e-12

    bad = np.array([[1.0, 2.0], [2.0, 1.0]])
    repaired = nearest_positive_semidefinite(bad)
    assert np.min(np.linalg.eigvalsh(repaired)) >= -1e-10


def test_variance_reduced_black_scholes_estimators_are_close_to_analytic():
    analytic = black_scholes_price(100.0, 100.0, 1.0, 0.03, 0.2)
    antithetic = black_scholes_antithetic_price(100.0, 100.0, 1.0, 0.03, 0.2, paths=20_000, seed=5)
    control = black_scholes_control_variate_price(100.0, 100.0, 1.0, 0.03, 0.2, paths=20_000, seed=5)
    assert abs(antithetic.price - analytic) < 0.35
    assert abs(control.price - analytic) < 0.35
    assert control.standard_error < antithetic.standard_error


def test_dynamic_kalman_spread_backtest_runs():
    prices = synthetic_cointegrated_prices(periods=160, seed=44)
    result = backtest_kalman_spread_strategy(prices["PairB"], prices["PairA"], entry_z=1.5, exit_z=0.25, window=20)
    assert len(result.history) == 160
    assert np.isfinite(result.total_pnl)
    assert result.turnover >= 0.0
    assert {"spread", "signal", "hedge_ratio", "cumulative_pnl"}.issubset(result.history.columns)


def test_credit_migration_matrix_tools():
    matrix = pd.DataFrame(
        [[90.0, 8.0, 2.0], [5.0, 85.0, 10.0], [0.0, 0.0, 100.0]],
        index=["A", "B", "D"],
        columns=["A", "B", "D"],
    )
    normalized = normalize_transition_matrix(matrix)
    assert np.allclose(normalized.sum(axis=1), 1.0)
    two_year = transition_matrix_power(normalized, 2)
    assert np.allclose(two_year.sum(axis=1), 1.0)
    pd_2y = cumulative_default_probability(normalized, "A", 2, default_state="D")
    assert 0.0 < pd_2y < 1.0
    paths = simulate_rating_paths(normalized, "A", periods=3, paths=5, seed=2)
    assert paths.shape == (5, 4)


def test_full_demo_exposes_covariance_mc_migration_dynamic_metrics():
    result = run_full_demo(seed=13).as_dict()
    assert result["options"]["control_variate_standard_error"] > 0.0
    assert result["options"]["antithetic_standard_error"] > 0.0
    assert result["factor_risk"]["ledoit_wolf_trace"] > 0.0
    assert np.isfinite(result["stat_arb"]["dynamic_spread_pnl"])
    assert 0.0 <= result["credit"]["five_year_migration_pd"] <= 1.0
