import numpy as np
import pandas as pd

from quantlab.credit.kmv import default_point, distance_to_default
from quantlab.market_making.inventory import inventory_diagnostics, inventory_skew_quote_adjustment
from quantlab.options.portfolio import OptionPosition, option_book_greeks, option_book_value, stress_option_book
from quantlab.portfolio.cvar_attribution import portfolio_cvar_contributions
from quantlab.workflows.demo_suite import run_full_demo


def test_option_book_value_greeks_and_stress():
    positions = [
        OptionPosition(quantity=2.0, strike=100.0, maturity=1.0, volatility=0.2, option_type="call"),
        OptionPosition(quantity=-1.0, strike=95.0, maturity=1.0, volatility=0.25, option_type="put"),
    ]
    value = option_book_value(100.0, 0.03, positions)
    greeks = option_book_greeks(100.0, 0.03, positions)
    stress = stress_option_book(100.0, 0.03, positions, np.array([-0.1, 0.0, 0.1]), np.array([-0.02, 0.0, 0.02]))
    assert np.isfinite(value)
    assert {"delta", "gamma", "vega", "theta", "rho"}.issubset(greeks.index)
    assert len(stress.scenario_values) == 9
    assert stress.base_value == value


def test_inventory_diagnostics_and_quote_skew():
    history = pd.DataFrame({"inventory": [0.0, 2.0, 6.0, -7.0], "pnl": [0.0, 1.0, -2.0, 3.0]})
    diagnostics = inventory_diagnostics(history, inventory_limit=5.0, penalty_coefficient=2.0)
    assert diagnostics.max_inventory == 7.0
    assert diagnostics.inventory_penalty > 0.0
    bid_adjustment, ask_adjustment = inventory_skew_quote_adjustment(5.0, inventory_limit=10.0, tick_size=0.01)
    assert bid_adjustment < 0.0
    assert ask_adjustment < 0.0


def test_cvar_contributions_sum_to_empirical_cvar():
    returns = pd.DataFrame(
        {
            "A": [-0.10, -0.02, 0.01, 0.02],
            "B": [-0.05, -0.01, 0.00, 0.01],
        }
    )
    weights = np.array([0.6, 0.4])
    contributions = portfolio_cvar_contributions(returns, weights, confidence=0.75)
    portfolio_returns = returns.to_numpy() @ weights
    empirical_cvar = -portfolio_returns[portfolio_returns <= np.quantile(portfolio_returns, 0.25)].mean()
    assert abs(contributions.sum() - empirical_cvar) < 1e-12


def test_distance_to_default_and_default_point():
    point = default_point(50.0, 100.0)
    result = distance_to_default(150.0, point, asset_volatility=0.25, drift=0.03)
    assert point == 100.0
    assert result.distance_to_default > 0.0
    assert 0.0 < result.expected_default_frequency < 1.0


def test_full_demo_exposes_option_inventory_cvar_kmv_metrics():
    result = run_full_demo(seed=16).as_dict()
    assert "option_stress_worst_pnl" in result["options"]
    assert "inventory_penalty" in result["market_making"]
    assert "largest_cvar_contribution" in result["portfolio"]
    assert result["credit"]["distance_to_default"] > 0.0
