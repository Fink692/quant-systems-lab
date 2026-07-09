import numpy as np
import pandas as pd

from quantlab.credit.curve import bootstrap_hazard_curve
from quantlab.credit.pricing import risky_coupon_bond_price
from quantlab.credit.sensitivity import coupon_bond_spread_sensitivity, shift_hazard_curve_by_spread
from quantlab.market_making.avellaneda_stoikov import AvellanedaStoikovParams
from quantlab.market_making.execution import ExecutionModelParams
from quantlab.market_making.path_simulator import simulate_latency_market_maker_on_path
from quantlab.systemic.scenarios import run_systemic_stress_scenarios
from quantlab.workflows.demo_suite import run_full_demo


def test_latency_market_maker_replays_observed_path_with_fills():
    result = simulate_latency_market_maker_on_path(
        np.linspace(100.0, 101.0, 12),
        AvellanedaStoikovParams(risk_aversion=0.05, volatility=0.1, order_book_liquidity=1.0, horizon=1.0),
        ExecutionModelParams(fill_intensity=1_000.0, order_book_liquidity=1.0, latency=0.01),
        dt=0.1,
        latency_steps=1,
        seed=3,
    )
    assert len(result.history) == 11
    assert result.fill_rate > 0.0
    assert np.isfinite(result.final_pnl)
    assert "arrival_mid" in result.history.columns


def test_risky_coupon_bond_and_spread_sensitivity_are_ordered():
    curve = bootstrap_hazard_curve(np.array([1.0, 3.0, 5.0]), np.array([0.01, 0.015, 0.02]), recovery_rate=0.4)
    base_price = risky_coupon_bond_price(0.04, 5.0, 0.03, curve, payment_frequency=2)
    up_curve = shift_hazard_curve_by_spread(curve, 10.0)
    down_curve = shift_hazard_curve_by_spread(curve, -10.0)
    up_price = risky_coupon_bond_price(0.04, 5.0, 0.03, up_curve)
    down_price = risky_coupon_bond_price(0.04, 5.0, 0.03, down_curve)
    sensitivity = coupon_bond_spread_sensitivity(0.04, 5.0, 0.03, curve)

    assert up_price < base_price < down_price
    assert sensitivity.spread_duration > 0.0
    assert sensitivity.spread_dv01 > 0.0


def test_systemic_stress_scenarios_aggregate_expected_shortfall():
    result = run_systemic_stress_scenarios(
        holdings=np.array([[100.0, 0.0], [0.0, 100.0]]),
        scenario_returns=pd.DataFrame(
            [[-0.10, -0.10], [-0.30, -0.05]],
            index=["mild", "shock"],
            columns=["asset_a", "asset_b"],
        ),
        capital=np.array([20.0, 20.0]),
        scenario_probabilities=pd.Series([0.8, 0.2], index=["mild", "shock"]),
    )
    assert result.worst_scenario == "shock"
    assert result.scenario_results.loc["shock", "default_count"] == 1
    assert abs(result.expected_shortfall - 2.0) < 1e-12
    assert result.institution_shortfalls.loc["shock", "institution_0"] == 10.0


def test_full_demo_exposes_path_credit_and_systemic_scenario_metrics():
    result = run_full_demo(seed=20).as_dict()
    assert result["market_making"]["path_fill_rate"] > 0.0
    assert result["credit"]["risky_coupon_bond_price"] > result["credit"]["risky_bond_price"]
    assert result["credit"]["coupon_bond_spread_dv01"] > 0.0
    assert result["systemic"]["scenario_expected_shortfall"] > 0.0
