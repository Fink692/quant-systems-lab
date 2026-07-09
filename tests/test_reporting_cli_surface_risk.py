import json

import numpy as np

from quantlab.cli import main
from quantlab.data.synthetic import synthetic_factor_panel, synthetic_option_chain
from quantlab.options.surface import build_volatility_surface_from_chain
from quantlab.portfolio.backtest import rolling_rebalance_backtest, static_weight_backtest
from quantlab.portfolio.optimization import min_variance_weights
from quantlab.reporting.markdown import render_demo_markdown, save_demo_report
from quantlab.risk.var import component_var, gaussian_var, historical_cvar, historical_var
from quantlab.workflows.demo_suite import run_full_demo


def test_volatility_surface_query_and_diagnostics():
    surface = build_volatility_surface_from_chain(synthetic_option_chain())
    atm_vol = surface.implied_volatility(1.0, 100.0)
    atm_price = surface.price(1.0, 100.0, "call")
    local_vol = surface.local_volatility()
    assert atm_vol > 0.0
    assert atm_price > 0.0
    assert local_vol.shape == surface.implied_volatilities.shape
    assert np.nanmean(local_vol) > 0.0
    assert len(surface.arbitrage_violations()) == 0


def test_var_metrics_and_portfolio_backtests():
    panel = synthetic_factor_panel(periods=90, assets=4, factors=2, seed=31)
    weights = np.full(panel.asset_returns.shape[1], 1.0 / panel.asset_returns.shape[1])
    portfolio_returns = panel.asset_returns.to_numpy() @ weights
    assert historical_var(portfolio_returns, confidence=0.95) > 0.0
    assert historical_cvar(portfolio_returns, confidence=0.95) >= historical_var(portfolio_returns, confidence=0.95)
    assert gaussian_var(float(portfolio_returns.mean()), float(portfolio_returns.std(ddof=1))) > 0.0

    covariance = panel.asset_returns.cov().to_numpy()
    components = component_var(weights, covariance)
    assert components.shape == weights.shape
    assert components.sum() > 0.0

    min_var = min_variance_weights(covariance)
    static = static_weight_backtest(panel.asset_returns, min_var, transaction_cost_bps=1.0)
    rolling = rolling_rebalance_backtest(
        panel.asset_returns,
        weight_function=lambda window: min_variance_weights(window.cov().to_numpy() + np.eye(window.shape[1]) * 1e-10),
        lookback=20,
        rebalance_frequency=10,
        transaction_cost_bps=1.0,
    )
    assert len(static.history) == len(panel.asset_returns) + 1
    assert len(rolling.weights) == len(panel.asset_returns) + 1
    assert np.isfinite(static.total_return)
    assert rolling.max_drawdown >= 0.0


def test_markdown_report_render_and_save(tmp_path):
    result = run_full_demo(seed=9)
    markdown = render_demo_markdown(result)
    assert markdown.startswith("# Quant Systems Lab Demo Report")
    assert "## Options" in markdown
    path = save_demo_report(result, tmp_path / "reports" / "demo.md")
    assert path.exists()
    assert path.read_text(encoding="utf-8") == markdown


def test_new_cli_commands(capsys, tmp_path):
    for command in (["surface-demo"], ["risk-demo", "--seed", "4"], ["portfolio-demo", "--seed", "4"]):
        assert main(command) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload

    output = tmp_path / "demo-report.md"
    assert main(["demo-report", "--seed", "4", "--output", str(output)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sections"] == 10
    assert output.exists()
