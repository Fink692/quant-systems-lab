import numpy as np
import pandas as pd
from scipy.stats import norm

from quantlab.credit.structural import calibrate_merton_asset_parameters, merton_equity_value
from quantlab.data.synthetic import synthetic_factor_panel
from quantlab.market_making.execution import ExecutionModelParams, expected_execution_value, fill_probability
from quantlab.options.black_scholes import black_scholes_price
from quantlab.options.greeks import black_scholes_greeks
from quantlab.options.local_volatility import dupire_local_volatility
from quantlab.portfolio.robust import resampled_efficient_weights, robust_mean_variance_weights
from quantlab.risk.attribution import factor_risk_attribution
from quantlab.risk.factor_model import fit_factor_model
from quantlab.rl.risk_metrics import performance_summary, risk_adjusted_reward
from quantlab.rough_vol.pricing import rough_bergomi_option_price
from quantlab.rough_vol.rough_bergomi import RoughBergomiParams
from quantlab.stat_arb.backtest import backtest_spread_strategy
from quantlab.systemic.clearing import eisenberg_noe_clearing


def test_black_scholes_greeks_match_finite_difference_delta():
    greeks = black_scholes_greeks(100.0, 101.0, 1.0, 0.03, 0.2, option_type="call")
    bump = 1e-3
    up = black_scholes_price(100.0 + bump, 101.0, 1.0, 0.03, 0.2)
    down = black_scholes_price(100.0 - bump, 101.0, 1.0, 0.03, 0.2)
    finite_delta = (up - down) / (2.0 * bump)
    assert abs(greeks.delta - finite_delta) < 1e-5
    assert greeks.gamma > 0.0
    assert greeks.vega > 0.0


def test_dupire_local_volatility_recovers_flat_black_scholes_surface():
    maturities = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    strikes = np.linspace(75.0, 125.0, 11)
    prices = np.array(
        [
            [black_scholes_price(100.0, strike, maturity, 0.02, 0.22) for strike in strikes]
            for maturity in maturities
        ]
    )
    local_vol = dupire_local_volatility(maturities, strikes, prices, rate=0.02)
    interior = local_vol[1:-1, 2:-2]
    assert np.nanmean(np.abs(interior - 0.22)) < 0.04


def test_execution_model_latency_and_adverse_selection():
    params = ExecutionModelParams(fill_intensity=3.0, order_book_liquidity=1.5, latency=0.1, adverse_selection_bps=1.0)
    near = fill_probability(quote_distance=0.01, horizon=1.0, params=params)
    far = fill_probability(quote_distance=1.0, horizon=1.0, params=params)
    blocked = fill_probability(quote_distance=0.01, horizon=0.05, params=params)
    assert near > far > 0.0
    assert blocked == 0.0
    assert expected_execution_value(100.0, 99.9, "bid", 1.0, params) > 0.0


def test_risk_attribution_and_robust_portfolio_weights():
    panel = synthetic_factor_panel(periods=80, assets=5, factors=2, seed=12)
    model = fit_factor_model(panel.asset_returns, panel.factor_returns)
    weights = pd.Series(np.full(5, 0.2), index=panel.asset_returns.columns)
    attribution = factor_risk_attribution(weights, model.exposures, model.factor_covariance, model.specific_variance)
    assert abs(attribution.total_variance - attribution.factor_variance - attribution.specific_variance) < 1e-12
    assert abs(attribution.asset_contributions.sum() - attribution.total_variance) < 1e-12

    returns = panel.asset_returns.to_numpy()
    robust = robust_mean_variance_weights(returns.mean(axis=0), np.cov(returns, rowvar=False), risk_aversion=5.0)
    resampled = resampled_efficient_weights(returns, n_resamples=5, risk_aversion=5.0, seed=1)
    assert abs(robust.sum() - 1.0) < 1e-8
    assert abs(resampled.sum() - 1.0) < 1e-8
    assert np.all(robust >= -1e-10)


def test_rl_metrics_and_rough_vol_pricing():
    equity = np.array([100.0, 101.0, 99.0, 104.0])
    summary = performance_summary(equity)
    assert abs(summary["total_return"] - 0.04) < 1e-12
    assert summary["max_drawdown"] > 0.0
    assert risk_adjusted_reward(0.02, drawdown=0.01, turnover=0.5, drawdown_penalty=1.0, turnover_penalty=0.01) == 0.005

    price = rough_bergomi_option_price(
        100.0,
        100.0,
        0.25,
        0.03,
        RoughBergomiParams(hurst=0.12, eta=1.0, rho=-0.4, xi0=0.04),
        paths=256,
        steps=16,
        seed=10,
    )
    assert price > 0.0


def test_spread_backtest_merton_calibration_and_clearing():
    spread = np.array([0.0, 1.0, 0.2, -0.8, -0.1])
    positions = np.array([0.0, -1.0, -1.0, 1.0, 1.0])
    backtest = backtest_spread_strategy(spread, positions, transaction_cost=0.001)
    assert len(backtest.history) == len(spread)
    assert np.isfinite(backtest.total_pnl)

    asset_value = 120.0
    debt = 90.0
    maturity = 1.0
    rate = 0.03
    asset_volatility = 0.25
    equity_value = merton_equity_value(asset_value, debt, maturity, rate, asset_volatility)
    d1 = (np.log(asset_value / debt) + (rate + 0.5 * asset_volatility**2) * maturity) / (
        asset_volatility * np.sqrt(maturity)
    )
    equity_volatility = norm.cdf(d1) * asset_value * asset_volatility / equity_value
    calibrated = calibrate_merton_asset_parameters(equity_value, equity_volatility, debt, maturity, rate)
    assert calibrated.success
    assert abs(calibrated.asset_value - asset_value) < 1e-5
    assert abs(calibrated.asset_volatility - asset_volatility) < 1e-5

    clearing = eisenberg_noe_clearing(np.array([[0.0, 100.0], [0.0, 0.0]]), np.array([40.0, 0.0]))
    assert clearing.converged
    assert clearing.defaulted.tolist() == [True, False]
    assert abs(clearing.payments[0] - 40.0) < 1e-10
