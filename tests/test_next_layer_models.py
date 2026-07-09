import numpy as np
import pandas as pd

from quantlab.credit.curve import bootstrap_hazard_curve
from quantlab.credit.pricing import cds_par_spread, risky_zero_coupon_price
from quantlab.data.synthetic import synthetic_cointegrated_prices
from quantlab.market_making.queue import simulate_queue_position
from quantlab.options.svi import SVIParams, calibrate_svi_slice, svi_implied_volatility
from quantlab.portfolio.stress import historical_stress_scenarios, stress_test_portfolio
from quantlab.stat_arb.johansen import basket_spread, johansen_hedge_vector, johansen_test
from quantlab.systemic.firesale import simulate_fire_sale
from quantlab.workflows.demo_suite import run_full_demo


def test_svi_calibration_recovers_smooth_total_variance_slice():
    maturity = 1.25
    log_moneyness = np.linspace(-0.35, 0.35, 15)
    true = SVIParams(a=0.02, b=0.18, rho=-0.35, m=0.02, sigma=0.25)
    vols = svi_implied_volatility(log_moneyness, maturity, true)
    result = calibrate_svi_slice(log_moneyness, vols, maturity)
    assert result.success
    assert result.objective_value < 1e-10
    fitted = svi_implied_volatility(log_moneyness, maturity, result.params)
    assert np.max(np.abs(fitted - vols)) < 1e-5


def test_johansen_basket_spread_is_finite():
    prices = synthetic_cointegrated_prices(periods=180, seed=13)[["PairA", "PairB"]]
    result = johansen_test(prices)
    hedge = johansen_hedge_vector(prices, normalize_asset="PairA")
    spread = basket_spread(prices, hedge)
    assert result.eigenvectors.shape == (2, 2)
    assert hedge.loc["PairA"] == 1.0
    assert np.isfinite(spread.std(ddof=1))


def test_credit_pricing_tools_are_bounded_and_consistent():
    curve = bootstrap_hazard_curve(np.array([1.0, 3.0, 5.0]), np.array([0.012, 0.018, 0.024]), recovery_rate=0.4)
    risky = risky_zero_coupon_price(5.0, 0.03, curve)
    risk_free = np.exp(-0.03 * 5.0)
    spread = cds_par_spread(5.0, 0.03, curve)
    assert 0.0 < risky < risk_free
    assert spread > 0.0


def test_queue_position_simulation_and_portfolio_stress():
    queue = simulate_queue_position(
        initial_ahead=10.0,
        order_size=3.0,
        market_order_intensity=80.0,
        cancellation_intensity=20.0,
        horizon=1.0,
        seed=2,
    )
    assert queue.filled
    assert queue.fill_time is not None
    assert queue.history["remaining"].iloc[-1] == 0.0

    returns = pd.DataFrame(
        {
            "A": [-0.01, 0.02, -0.05, 0.01],
            "B": [-0.02, 0.01, -0.03, 0.02],
        }
    )
    scenarios = historical_stress_scenarios(returns, quantile=0.25)
    stress = stress_test_portfolio(pd.Series({"A": 0.6, "B": 0.4}), scenarios, portfolio_value=1_000_000.0)
    assert stress.worst_scenario in scenarios.index
    assert stress.scenario_pnl.min() < 0.0


def test_fire_sale_feedback_can_create_defaults():
    result = simulate_fire_sale(
        holdings=np.array([[100.0, 50.0], [90.0, 90.0]]),
        capital=np.array([10.0, 20.0]),
        initial_shock=np.array([-0.1, -0.05]),
        impact=np.array([0.001, 0.001]),
        liquidation_fraction=0.5,
    )
    assert result.rounds >= 1
    assert result.defaulted.any()
    assert np.all(result.prices >= 0.0)


def test_full_demo_exposes_next_layer_metrics():
    result = run_full_demo(seed=6).as_dict()
    assert result["options"]["svi_objective"] >= 0.0
    assert "queue_filled" in result["market_making"]
    assert result["stat_arb"]["johansen_spread_std"] > 0.0
    assert result["credit"]["cds_par_spread"] > 0.0
    assert "firesale_defaults" in result["systemic"]
