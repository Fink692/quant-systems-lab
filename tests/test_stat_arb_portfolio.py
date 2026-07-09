import numpy as np

from quantlab.data.synthetic import synthetic_cointegrated_prices
from quantlab.stat_arb.network import pairwise_cointegration_network
from quantlab.stat_arb.portfolio import allocate_pair_capital, backtest_pair_portfolio
from quantlab.stat_arb.selection import rank_cointegrated_pairs
from quantlab.workflows.demo_suite import run_full_demo


def test_allocate_pair_capital_uses_scores_and_preserves_gross():
    prices = synthetic_cointegrated_prices(periods=160, seed=31)
    network = pairwise_cointegration_network(prices, p_value_threshold=0.05)
    candidates = rank_cointegrated_pairs(prices, network, max_p_value=0.05, top_n=2)
    allocations = allocate_pair_capital(candidates, total_gross=2.0)

    assert len(allocations) == len(candidates)
    assert abs(allocations.sum() - 2.0) < 1e-12
    assert (allocations > 0.0).all()
    assert allocations.iloc[0] >= allocations.iloc[-1]


def test_pair_portfolio_backtest_aggregates_pair_pnl_and_turnover():
    prices = synthetic_cointegrated_prices(periods=180, seed=32)
    network = pairwise_cointegration_network(prices, p_value_threshold=0.05)
    candidates = rank_cointegrated_pairs(prices, network, max_p_value=0.05, top_n=2)
    result = backtest_pair_portfolio(prices, candidates, entry_z=1.5, exit_z=0.25, window=30, transaction_cost=0.001)

    assert len(result.history) == len(prices.dropna())
    assert result.pair_pnl.shape[1] == len(candidates)
    assert abs(result.allocations.sum() - 1.0) < 1e-12
    assert np.allclose(result.history["pnl"], result.pair_pnl.sum(axis=1))
    assert result.turnover >= 0.0
    assert result.max_drawdown >= 0.0


def test_full_demo_exposes_pair_portfolio_metrics():
    result = run_full_demo(seed=25).as_dict()
    assert "pair_portfolio_pnl" in result["stat_arb"]
    assert "pair_portfolio_turnover" in result["stat_arb"]
    assert result["stat_arb"]["pair_portfolio_turnover"] >= 0.0
    assert result["stat_arb"]["pair_portfolio_max_drawdown"] >= 0.0
