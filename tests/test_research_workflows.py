import numpy as np

from quantlab.options.calibration import OptionQuote
from quantlab.options.diagnostics import evaluate_option_pricer, pricing_error_report
from quantlab.options.surface_arbitrage import detect_surface_arbitrage
from quantlab.options.surface_repair import repair_call_price_surface
from quantlab.portfolio.frontier import efficient_frontier
from quantlab.risk.backtesting import kupiec_var_backtest
from quantlab.rl.policy_search import grid_search_constant_weight_policy
from quantlab.workflows.demo_suite import run_full_demo


def test_option_pricing_diagnostics_summarize_errors():
    quotes = [
        OptionQuote(strike=95.0, maturity=1.0, price=8.0, weight=2.0),
        OptionQuote(strike=100.0, maturity=1.0, price=5.0, weight=1.0),
        OptionQuote(strike=105.0, maturity=1.0, price=3.0, weight=1.0),
    ]
    report = pricing_error_report(quotes, np.array([8.1, 4.9, 3.2]))
    assert len(report.residuals) == 3
    assert report.rmse > 0.0
    assert report.max_abs_error == 0.20000000000000018

    evaluated = evaluate_option_pricer(quotes, lambda quote: quote.price)
    assert evaluated.rmse == 0.0


def test_surface_repair_removes_basic_static_arbitrage():
    maturities = np.array([0.5, 1.0])
    strikes = np.array([90.0, 100.0, 110.0])
    prices = np.array([[15.0, 14.0, 8.0], [14.0, 13.0, 7.0]])
    assert detect_surface_arbitrage(maturities, strikes, prices)
    result = repair_call_price_surface(maturities, strikes, prices)
    assert result.success
    assert result.violations_before
    assert result.violations_after == []
    assert np.all(result.repaired_prices >= 0.0)


def test_efficient_frontier_builds_target_portfolios():
    mu = np.array([0.02, 0.05, 0.08])
    cov = np.diag([0.04, 0.05, 0.09])
    result = efficient_frontier(mu, cov, np.array([0.03, 0.05, 0.07]), asset_names=["A", "B", "C"])
    assert len(result.points) == 3
    assert result.weights.shape == (3, 3)
    assert np.allclose(result.weights.sum(axis=1), 1.0)


def test_kupiec_backtest_and_policy_search():
    rng = np.random.default_rng(123)
    returns = rng.normal(0.0, 0.01, size=500)
    var_estimates = np.full_like(returns, 0.0165)
    backtest = kupiec_var_backtest(returns, var_estimates, confidence=0.95)
    assert backtest.observations == 500
    assert 0.0 <= backtest.exception_rate <= 1.0
    assert 0.0 <= backtest.kupiec_p_value <= 1.0

    prices = np.linspace(100.0, 120.0, 50)
    search = grid_search_constant_weight_policy(prices, candidate_weights=np.array([-1.0, 0.0, 1.0]))
    assert search.best_weight == 1.0
    assert set(search.results.columns) >= {"weight", "score", "total_return", "max_drawdown", "sharpe"}


def test_full_demo_exposes_research_workflow_metrics():
    result = run_full_demo(seed=8).as_dict()
    assert result["options"]["pricing_rmse"] >= 0.0
    assert "best_constant_weight" in result["rl_trading"]
    assert result["portfolio"]["frontier_points"] > 0
    assert 0.0 <= result["portfolio"]["var_exception_rate"] <= 1.0
    assert result["surface_arbitrage"]["violations_after_repair"] == 0
