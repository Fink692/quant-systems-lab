import numpy as np
import pandas as pd

from quantlab.data.synthetic import synthetic_factor_panel
from quantlab.options.black_scholes import black_scholes_price
from quantlab.options.density import breeden_litzenberger_density, density_from_price_surface
from quantlab.risk.style_factors import build_style_exposures, estimate_style_factor_returns
from quantlab.rl.walk_forward import walk_forward_q_learning
from quantlab.systemic.capital import capital_adequacy, systemic_capital_surcharge
from quantlab.workflows.demo_suite import run_full_demo


def test_risk_neutral_density_from_call_slice():
    strikes = np.linspace(60.0, 140.0, 41)
    prices = np.array([black_scholes_price(100.0, strike, 1.0, 0.03, 0.2) for strike in strikes])
    density = breeden_litzenberger_density(strikes, prices, maturity=1.0, rate=0.03)
    assert abs(density.mass - 1.0) < 1e-10
    assert 95.0 < density.mean_strike < 110.0
    assert np.all(density.density >= 0.0)

    grid = np.vstack([prices, prices * 1.05])
    frame = density_from_price_surface(np.array([1.0, 1.5]), strikes, grid, rate=0.03)
    assert {"maturity", "strike", "density"}.issubset(frame.columns)
    assert len(frame) == 2 * len(strikes)


def test_style_factor_exposures_and_cross_sectional_returns():
    panel = synthetic_factor_panel(periods=40, assets=6, factors=2, seed=50)
    fundamentals = pd.DataFrame(
        {
            "market_cap": np.linspace(100.0, 1_000.0, 6),
            "book_to_market": np.linspace(0.4, 1.2, 6),
            "momentum": np.linspace(-0.2, 0.3, 6),
            "volatility": np.linspace(0.1, 0.4, 6),
        },
        index=panel.asset_returns.columns,
    )
    exposures = build_style_exposures(fundamentals)
    result = estimate_style_factor_returns(panel.asset_returns, exposures)
    assert exposures.shape == (6, 4)
    assert result.factor_returns.shape == (40, 4)
    assert result.residuals.shape == (40, 6)
    assert np.allclose(exposures.mean(axis=0), 0.0)


def test_walk_forward_q_learning_and_capital_tools():
    prices = np.linspace(100.0, 125.0, 80)
    wf = walk_forward_q_learning(prices, train_size=30, test_size=20, episodes=5, seed=3)
    assert len(wf.folds) >= 1
    assert np.isfinite(wf.mean_test_return)

    capital = pd.Series([8.0, 12.0], index=["BankA", "BankB"])
    rwa = pd.Series([100.0, 100.0], index=["BankA", "BankB"])
    adequacy = capital_adequacy(capital, rwa, minimum_ratio=0.10)
    assert adequacy.total_shortfall == 2.0
    assert adequacy.capital_shortfall.loc["BankA"] == 2.0

    surcharge = systemic_capital_surcharge(pd.Series([0.2, 0.4], index=["BankA", "BankB"]))
    assert surcharge.loc["BankB"] > surcharge.loc["BankA"]


def test_full_demo_exposes_density_style_walkforward_capital_metrics():
    result = run_full_demo(seed=14).as_dict()
    assert abs(result["options"]["density_mass"] - 1.0) < 1e-8
    assert result["factor_risk"]["style_factor_count"] == 4
    assert np.isfinite(result["rl_trading"]["walk_forward_q_return"])
    assert result["systemic"]["capital_shortfall"] >= 0.0
