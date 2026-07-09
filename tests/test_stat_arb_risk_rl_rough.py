import numpy as np
import pandas as pd

from quantlab.risk.factor_model import fit_factor_model, shrink_covariance
from quantlab.rl.trading_env import TradingEnv
from quantlab.rough_vol.rough_bergomi import RoughBergomiParams, simulate_rough_bergomi
from quantlab.stat_arb.cointegration import engle_granger, estimate_ou, zscore


def test_factor_model_shapes_and_covariance():
    rng = np.random.default_rng(7)
    factors = pd.DataFrame(rng.normal(size=(80, 2)), columns=["value", "momentum"])
    assets = pd.DataFrame(
        {
            "A": 0.2 + 1.5 * factors["value"] - 0.2 * factors["momentum"] + rng.normal(scale=0.05, size=80),
            "B": -0.1 + 0.4 * factors["value"] + 0.8 * factors["momentum"] + rng.normal(scale=0.05, size=80),
        }
    )
    result = fit_factor_model(assets, factors)
    assert result.exposures.shape == (2, 2)
    assert result.covariance_matrix().shape == (2, 2)
    shrunk = shrink_covariance(assets, shrinkage=0.5)
    assert shrunk.shape == (2, 2)


def test_stat_arb_estimators_return_spread_diagnostics():
    rng = np.random.default_rng(11)
    x = np.cumsum(rng.normal(size=120))
    y = 2.0 + 1.3 * x + rng.normal(scale=0.2, size=120)
    result = engle_granger(y, x)
    assert abs(result.hedge_ratio - 1.3) < 0.1
    ou = estimate_ou(result.spread)
    assert ou["theta"] > 0
    scores = zscore(result.spread, window=20)
    assert np.isnan(scores[:19]).all()
    assert np.isfinite(scores[19:]).all()


def test_trading_env_steps_and_charges_costs():
    env = TradingEnv(np.array([100.0, 101.0, 99.0]), transaction_cost_bps=2.0)
    state, reward, done, info = env.step(0.5)
    assert state.time_index == 1
    assert not done
    assert info["transaction_cost"] > 0
    assert np.isfinite(reward)


def test_rough_bergomi_simulation_shapes():
    spots, variances = simulate_rough_bergomi(
        100.0,
        maturity=0.25,
        steps=8,
        paths=4,
        params=RoughBergomiParams(hurst=0.1, eta=1.2, rho=-0.5, xi0=0.04),
        seed=3,
    )
    assert spots.shape == (4, 9)
    assert variances.shape == (4, 9)
    assert np.all(spots > 0)
