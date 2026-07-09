import numpy as np
import pandas as pd

from quantlab.credit.curve import bootstrap_hazard_curve
from quantlab.market_making.avellaneda_stoikov import AvellanedaStoikovParams
from quantlab.market_making.simulator import simulate_market_maker
from quantlab.portfolio.black_litterman import black_litterman_posterior
from quantlab.risk.statistical_factors import fit_pca_factor_model
from quantlab.rl.evaluation import constant_weight_policy, run_policy, walk_forward_splits
from quantlab.rl.trading_env import TradingEnv
from quantlab.rough_vol.calibration import estimate_hurst_from_variogram
from quantlab.stat_arb.network import mean_reversion_signal, pairwise_cointegration_network
from quantlab.systemic.stress import exposure_centrality, external_asset_stress


def test_market_making_simulator_produces_history():
    result = simulate_market_maker(
        100.0,
        AvellanedaStoikovParams(risk_aversion=0.1, volatility=0.2, order_book_liquidity=1.0, horizon=1.0),
        steps=40,
        seed=4,
    )
    assert len(result.history) == 40
    assert {"mid", "bid", "ask", "inventory", "pnl"}.issubset(result.history.columns)
    assert np.isfinite(result.final_pnl)


def test_trading_policy_evaluation_and_walk_forward_splits():
    prices = np.array([100.0, 101.0, 102.0, 101.5, 103.0])
    result = run_policy(TradingEnv(prices, transaction_cost_bps=0.5), constant_weight_policy(0.6))
    assert len(result.history) == len(prices)
    assert np.isfinite(result.total_return)
    assert result.max_drawdown >= 0.0
    splits = list(walk_forward_splits(length=20, train_size=8, test_size=4, step_size=4))
    assert len(splits) == 3
    assert splits[0] == (slice(0, 8), slice(8, 12))


def test_pca_factor_model_and_black_litterman_shapes():
    rng = np.random.default_rng(22)
    returns = pd.DataFrame(rng.normal(size=(60, 4)), columns=["A", "B", "C", "D"])
    pca = fit_pca_factor_model(returns, n_factors=2)
    assert pca.factor_returns.shape == (60, 2)
    assert pca.loadings.shape == (4, 2)

    covariance = returns.cov().to_numpy()
    posterior = black_litterman_posterior(
        covariance,
        market_weights=np.full(4, 0.25),
        views_matrix=np.array([[1.0, -1.0, 0.0, 0.0]]),
        views=np.array([0.01]),
    )
    assert posterior.posterior_returns.shape == (4,)
    assert posterior.posterior_covariance.shape == (4, 4)


def test_roughness_estimator_returns_bounded_hurst():
    rng = np.random.default_rng(99)
    series = np.cumsum(rng.normal(size=500))
    estimate = estimate_hurst_from_variogram(series, max_lag=15)
    assert 0.2 < estimate.hurst < 0.8
    assert estimate.r_squared > 0.5


def test_cointegration_network_and_mean_reversion_signal():
    rng = np.random.default_rng(101)
    x = np.cumsum(rng.normal(size=160))
    prices = pd.DataFrame(
        {
            "A": 2.0 + 1.2 * x + rng.normal(scale=0.15, size=160),
            "B": x,
            "C": np.cumsum(rng.normal(size=160)),
        }
    )
    network = pairwise_cointegration_network(prices, p_value_threshold=0.05)
    assert bool(network.adjacency.loc["A", "B"])
    spread = prices["A"].to_numpy() - 1.2 * prices["B"].to_numpy()
    signal = mean_reversion_signal(spread, entry_z=1.5, exit_z=0.25, window=20)
    assert signal.shape == spread.shape
    assert set(np.unique(signal)).issubset({-1.0, 0.0, 1.0})


def test_credit_curve_and_systemic_stress_tools():
    curve = bootstrap_hazard_curve(np.array([1.0, 3.0, 5.0]), np.array([0.01, 0.015, 0.02]), recovery_rate=0.4)
    assert curve.survival(5.0) < curve.survival(1.0)
    assert curve.spread(2.0) == 0.015

    stress = external_asset_stress(
        holdings=np.array([[100.0, 50.0], [25.0, 150.0]]),
        asset_returns=np.array([-0.3, -0.05]),
        capital=np.array([20.0, 40.0]),
    )
    assert stress.defaulted.tolist() == [True, False]
    centrality = exposure_centrality(np.array([[0.0, 5.0], [2.0, 0.0]]), names=["BankA", "BankB"])
    assert list(centrality.index) == ["BankA", "BankB"]
