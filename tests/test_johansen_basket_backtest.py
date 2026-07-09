import numpy as np
import pandas as pd

from quantlab.stat_arb.basket_backtest import backtest_johansen_basket_strategy
from quantlab.workflows.demo_suite import run_full_demo


def test_johansen_basket_backtest_trades_multi_asset_spread():
    rng = np.random.default_rng(41)
    dates = pd.date_range("2020-01-01", periods=180, freq="B")
    common = np.cumsum(rng.normal(0.0, 1.0, size=len(dates)))
    prices = pd.DataFrame(
        {
            "A": 50.0 + common + rng.normal(0.0, 0.15, size=len(dates)),
            "B": 35.0 + 1.3 * common + rng.normal(0.0, 0.15, size=len(dates)),
            "C": 20.0 + 0.6 * common + rng.normal(0.0, 0.15, size=len(dates)),
        },
        index=dates,
    )

    result = backtest_johansen_basket_strategy(
        prices,
        entry_z=1.4,
        exit_z=0.25,
        window=30,
        gross_exposure=1.0,
        transaction_cost=0.001,
    )

    assert result.history.shape[0] == len(prices)
    assert result.asset_positions.shape == prices.shape
    assert abs(result.hedge_weights.abs().sum() - 1.0) < 1e-12
    assert result.turnover >= 0.0
    assert result.max_drawdown >= 0.0
    assert np.isfinite(result.total_pnl)
    assert set(np.unique(result.history["signal"])).issubset({-1.0, 0.0, 1.0})


def test_johansen_basket_backtest_respects_gross_exposure():
    rng = np.random.default_rng(42)
    common = np.cumsum(rng.normal(size=120))
    prices = pd.DataFrame(
        {
            "A": 100.0 + common + rng.normal(scale=0.1, size=120),
            "B": 80.0 + 0.8 * common + rng.normal(scale=0.1, size=120),
            "C": 60.0 + 1.1 * common + rng.normal(scale=0.1, size=120),
        }
    )

    result = backtest_johansen_basket_strategy(prices, window=25, gross_exposure=2.0)

    assert abs(result.hedge_weights.abs().sum() - 2.0) < 1e-12


def test_full_demo_exposes_johansen_basket_backtest_metrics():
    result = run_full_demo(seed=35).as_dict()

    assert "johansen_basket_pnl" in result["stat_arb"]
    assert result["stat_arb"]["johansen_basket_turnover"] >= 0.0
    assert result["stat_arb"]["johansen_basket_max_drawdown"] >= 0.0
