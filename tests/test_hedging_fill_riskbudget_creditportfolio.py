import numpy as np

from quantlab.credit.portfolio import gaussian_copula_default_losses
from quantlab.market_making.fill_calibration import calibrate_fill_intensity
from quantlab.options.hedging import simulate_delta_hedge
from quantlab.portfolio.risk_budget import portfolio_risk_contributions, risk_budget_weights
from quantlab.workflows.demo_suite import run_full_demo


def test_delta_hedge_simulation_tracks_costs_and_pnl():
    spot_path = np.linspace(100.0, 105.0, 25)
    result = simulate_delta_hedge(spot_path, 100.0, 1.0, 0.03, 0.2, transaction_cost_bps=1.0)
    assert len(result.history) == len(spot_path)
    assert np.isfinite(result.final_pnl)
    assert result.total_transaction_cost >= 0.0
    assert result.history["delta"].iloc[-1] == 0.0


def test_fill_intensity_calibration_orders_probabilities_by_distance():
    distances = np.array([0.01, 0.02, 0.03, 0.10, 0.15, 0.20])
    horizons = np.ones_like(distances)
    fills = np.array([True, True, True, False, False, False])
    result = calibrate_fill_intensity(distances, horizons, fills)
    assert result.success
    assert result.base_intensity > 0.0
    assert result.distance_decay > 0.0
    assert result.fill_probability(0.01, 1.0) > result.fill_probability(0.20, 1.0)


def test_risk_budget_weights_match_target_contributions():
    covariance = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.0], [0.0, 0.0, 0.16]])
    target = np.array([0.2, 0.3, 0.5])
    weights = risk_budget_weights(covariance, target)
    contributions = portfolio_risk_contributions(weights, covariance)
    assert abs(weights.sum() - 1.0) < 1e-8
    assert np.max(np.abs(contributions - target)) < 1e-3


def test_gaussian_copula_credit_portfolio_losses_are_ordered():
    result = gaussian_copula_default_losses(
        default_probabilities=np.array([0.01, 0.03, 0.05]),
        exposures=np.array([1_000_000.0, 500_000.0, 250_000.0]),
        recovery_rates=0.4,
        asset_correlation=0.25,
        simulations=3_000,
        confidence=0.95,
        seed=7,
    )
    assert result.expected_loss >= 0.0
    assert result.value_at_risk >= result.expected_loss
    assert result.expected_shortfall >= result.value_at_risk
    assert result.default_counts.shape == result.losses.shape


def test_full_demo_exposes_hedging_fill_risk_budget_and_credit_portfolio():
    result = run_full_demo(seed=15).as_dict()
    assert "delta_hedge_pnl" in result["options"]
    assert result["market_making"]["calibrated_fill_intensity"] > 0.0
    assert result["portfolio"]["risk_budget_max_error"] < 1e-2
    assert result["credit"]["portfolio_expected_shortfall"] >= result["credit"]["portfolio_expected_loss"]
