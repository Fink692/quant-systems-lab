import numpy as np
import pandas as pd

from quantlab.market_making.attribution import attribute_market_making_pnl
from quantlab.portfolio.cdar import cdar_minimizing_weights
from quantlab.portfolio.drawdown import portfolio_drawdown_summary
from quantlab.workflows.demo_suite import run_full_demo


def test_cdar_minimizing_weights_are_feasible_and_reduce_path_risk():
    returns = pd.DataFrame(
        {
            "steady": [0.01, 0.01, -0.005, 0.008, 0.006, 0.004],
            "crashy": [0.05, -0.20, 0.04, 0.03, -0.10, 0.02],
            "flat": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        }
    )
    result = cdar_minimizing_weights(returns, confidence=0.8)
    assert result.success
    assert abs(result.weights.sum() - 1.0) < 1e-10
    assert (result.weights >= -1e-12).all()
    equal_weight_cdar = portfolio_drawdown_summary(returns, np.full(3, 1.0 / 3.0), confidence=0.8).conditional_drawdown_at_risk
    assert result.weights["crashy"] < 0.05
    assert result.drawdown_summary.conditional_drawdown_at_risk <= equal_weight_cdar
    assert result.drawdown_summary.conditional_drawdown_at_risk >= 0.0


def test_cdar_target_return_constraint_tilts_weights():
    returns = pd.DataFrame(
        {
            "low": [0.001, 0.001, 0.001, 0.001],
            "high": [0.02, -0.01, 0.02, -0.01],
        }
    )
    result = cdar_minimizing_weights(returns, confidence=0.75, target_return=0.004)
    realized_mean = float(returns.mean().to_numpy() @ result.weights.to_numpy())
    assert realized_mean >= 0.004 - 1e-10
    assert result.weights["high"] > 0.0


def test_market_making_pnl_attribution_reconciles_final_pnl():
    history = pd.DataFrame(
        {
            "quote_mid": [100.0, 101.0],
            "bid": [99.9, 100.9],
            "ask": [100.1, 101.1],
            "bid_filled": [True, False],
            "ask_filled": [False, True],
            "slippage": [-0.02, -0.03],
            "pnl": [0.5, 1.25],
        }
    )
    attribution = attribute_market_making_pnl(history)
    assert attribution.fill_count == 2
    assert abs(attribution.spread_capture - 0.2) < 1e-12
    assert abs(attribution.explained_pnl - attribution.final_pnl) < 1e-12


def test_full_demo_exposes_cdar_and_pnl_attribution_metrics():
    result = run_full_demo(seed=21).as_dict()
    assert "cdar_objective" in result["portfolio"]
    assert result["portfolio"]["cdar_objective"] >= 0.0
    assert result["market_making"]["path_spread_capture"] >= 0.0
    assert "path_inventory_mark_to_market" in result["market_making"]
