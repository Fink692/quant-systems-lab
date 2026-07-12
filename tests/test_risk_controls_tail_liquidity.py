import numpy as np
import pandas as pd

from quantlab.credit.tranches import tranche_loss_distribution
from quantlab.market_making.latency import latency_slippage_report
from quantlab.portfolio.drawdown import conditional_drawdown_at_risk, portfolio_drawdown_summary
from quantlab.rl.evaluation import constant_weight_policy
from quantlab.rl.risk_controls import RiskLimits, apply_risk_limits, risk_limited_policy, volatility_target_weight
from quantlab.rl.trading_env import TradingState
from quantlab.systemic.liquidity import simulate_liquidity_spiral
from quantlab.workflows.demo_suite import run_full_demo


def test_conditional_drawdown_and_portfolio_summary():
    summary = conditional_drawdown_at_risk(np.array([100.0, 110.0, 90.0, 95.0, 80.0, 120.0]), confidence=0.8)
    assert summary.max_drawdown > 0.25
    assert summary.conditional_drawdown_at_risk <= summary.max_drawdown
    assert summary.time_under_water == 3

    returns = pd.DataFrame({"A": [0.02, -0.10, 0.05], "B": [0.01, -0.02, 0.01]})
    portfolio = portfolio_drawdown_summary(returns, np.array([0.75, 0.25]), confidence=0.75)
    assert portfolio.max_drawdown > 0.0
    assert len(portfolio.equity) == len(returns)


def test_risk_limits_and_volatility_targeting():
    state = TradingState(time_index=3, price=100.0, position=0.0, cash=96.0, equity=96.0, peak_equity=100.0)
    decision = apply_risk_limits(state, proposed_weight=1.0, limits=RiskLimits(max_leverage=1.0, max_drawdown=0.05))
    assert 0.0 < decision.approved_weight < 1.0
    assert decision.limited

    breached = TradingState(time_index=4, price=100.0, position=0.0, cash=90.0, equity=90.0, peak_equity=100.0)
    limited_policy = risk_limited_policy(
        constant_weight_policy(1.0), RiskLimits(max_leverage=1.0, max_drawdown=0.05, de_risk_weight=0.0)
    )
    assert limited_policy(breached) == 0.0
    assert volatility_target_weight(1.0, realized_volatility=0.24, target_volatility=0.12, max_leverage=2.0) == 0.5


def test_latency_slippage_report_detects_adverse_fills():
    report = latency_slippage_report(
        quote_mid_prices=np.array([100.0, 100.0, 100.0]),
        arrival_mid_prices=np.array([99.98, 100.03, 100.02]),
        quote_prices=np.array([99.99, 100.01, 99.99]),
        sides=np.array(["bid", "ask", "bid"]),
        quantities=np.array([1.0, 2.0, 1.0]),
    )
    assert report.total_slippage < 0.0
    assert report.adverse_fill_rate == 2.0 / 3.0
    assert report.realized_edge.shape == (3,)


def test_tranche_loss_distribution_maps_portfolio_losses_to_attachment_band():
    tranche = tranche_loss_distribution(
        np.array([0.0, 50.0, 100.0, 200.0]),
        attachment=0.05,
        detachment=0.15,
        portfolio_notional=1_000.0,
        confidence=0.75,
    )
    assert np.allclose(tranche.loss_rates, np.array([0.0, 0.0, 0.5, 1.0]))
    assert tranche.expected_loss_rate == 0.375
    assert tranche.expected_shortfall == 1.0


def test_liquidity_spiral_deleverages_and_impacts_prices():
    result = simulate_liquidity_spiral(
        holdings=np.array([[100.0, 0.0], [0.0, 100.0]]),
        capital=np.array([40.0, 40.0]),
        initial_returns=np.array([-0.2, -0.1]),
        market_depth=np.array([1_000.0, 1_000.0]),
        leverage_limit=3.0,
        liquidation_speed=0.5,
    )
    assert result.rounds > 0
    assert result.total_liquidation.sum() > 0.0
    assert np.all(result.prices <= np.array([0.8, 0.9]))


def test_full_demo_exposes_tail_risk_and_liquidity_metrics():
    result = run_full_demo(seed=17).as_dict()
    assert "latency_total_slippage" in result["market_making"]
    assert "risk_limited_max_drawdown" in result["rl_trading"]
    assert "conditional_drawdown_at_risk" in result["portfolio"]
    assert "mezzanine_tranche_expected_loss" in result["credit"]
    assert "liquidity_spiral_min_price" in result["systemic"]
