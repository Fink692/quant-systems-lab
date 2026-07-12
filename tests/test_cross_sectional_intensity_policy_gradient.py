import numpy as np
import pandas as pd

from quantlab.credit.intensity import fit_logistic_hazard
from quantlab.risk.cross_sectional import (
    build_sector_exposures,
    estimate_cross_sectional_factor_returns,
    factor_mimicking_portfolios,
    neutralize_portfolio_exposures,
)
from quantlab.rl.policy_gradient import (
    PolicyGradientRiskConstraints,
    train_constrained_policy_gradient,
    train_softmax_policy_gradient,
)
from quantlab.rl.trading_env import TradingEnv
from quantlab.workflows.demo_suite import run_full_demo


def test_cross_sectional_factor_returns_and_sector_exposures():
    assets = ["A", "B", "C", "D", "E", "F"]
    exposures = pd.DataFrame(
        {
            "value": [-1.0, -0.5, 0.0, 0.4, 0.8, 1.2],
            "momentum": [1.0, 0.5, -0.2, -0.4, 0.1, -0.9],
        },
        index=assets,
    )
    dates = pd.date_range("2022-01-01", periods=12)
    value_returns = np.linspace(-0.01, 0.01, len(dates))
    momentum_returns = np.linspace(0.02, -0.005, len(dates))
    returns = pd.DataFrame(index=dates, columns=assets, dtype=float)
    for idx, date in enumerate(dates):
        returns.loc[date] = 0.001 + exposures.to_numpy() @ np.array([value_returns[idx], momentum_returns[idx]])

    result = estimate_cross_sectional_factor_returns(returns, exposures)
    assert result.factor_returns.shape == (len(dates), 2)
    assert np.max(np.abs(result.residuals.to_numpy())) < 1e-12
    assert abs(result.factor_returns["value"].iloc[-1] - value_returns[-1]) < 1e-12

    sectors = build_sector_exposures(pd.Series(["tech", "tech", "bank", "bank", "energy", "energy"], index=assets))
    assert sectors.shape == (6, 3)
    assert np.allclose(sectors.sum(axis=1), 1.0)


def test_neutralized_and_mimicking_portfolios_hit_target_exposures():
    exposures = pd.DataFrame(
        {
            "size": [-1.0, -0.5, 0.2, 0.8],
            "value": [0.4, -0.2, 0.3, -0.5],
        },
        index=["A", "B", "C", "D"],
    )
    weights = pd.Series([0.4, 0.3, 0.2, 0.1], index=exposures.index)
    neutralized = neutralize_portfolio_exposures(weights, exposures)
    assert abs(neutralized.sum() - weights.sum()) < 1e-12
    assert np.linalg.norm((exposures.T @ neutralized).to_numpy()) < 1e-12

    mimicking = factor_mimicking_portfolios(exposures)
    exposure_matrix = mimicking.to_numpy() @ exposures.to_numpy()
    assert np.allclose(exposure_matrix, np.eye(2), atol=1e-10)


def test_logistic_hazard_orders_low_and_high_risk_borrowers():
    covariates = pd.DataFrame(
        {
            "leverage": [0.25, 0.35, 0.45, 0.75, 0.90, 1.10],
            "spread": [0.006, 0.008, 0.012, 0.025, 0.035, 0.055],
        }
    )
    fit = fit_logistic_hazard(covariates, np.array([0, 0, 0, 1, 1, 1]), l2_penalty=1.0)
    low_high = pd.DataFrame({"leverage": [0.30, 1.05], "spread": [0.007, 0.050]})
    hazards = fit.predict_hazard(low_high)
    assert hazards.iloc[1] > hazards.iloc[0]
    assert 0.0 < fit.survival_probability(low_high) < 1.0


def test_softmax_policy_gradient_trains_discrete_policy_surface():
    prices = 100.0 * np.exp(np.linspace(0.0, 0.08, 40))
    result = train_softmax_policy_gradient(prices, episodes=6, learning_rate=0.02, seed=5)
    state = TradingEnv(prices).reset()
    probabilities = result.action_probabilities(state)
    assert result.theta.shape == (3, 3)
    assert len(result.episode_rewards) == 6
    assert np.all(probabilities >= 0.0)
    assert abs(probabilities.sum() - 1.0) < 1e-12
    assert result.policy(state) in {-1.0, 0.0, 1.0}


def test_constrained_policy_gradient_updates_lagrangian_penalties():
    prices = np.array([100.0, 98.0, 96.0, 94.0, 92.0, 90.0, 89.0, 88.0])
    result = train_constrained_policy_gradient(
        prices,
        candidate_weights=np.array([1.0]),
        constraints=PolicyGradientRiskConstraints(max_drawdown=0.005, max_turnover=0.10),
        episodes=4,
        learning_rate=0.02,
        penalty_learning_rate=5.0,
        seed=11,
    )
    state = TradingEnv(prices).reset()
    probabilities = result.action_probabilities(state)
    assert result.theta.shape == (3, 1)
    assert len(result.constraint_history) == 4
    assert probabilities.shape == (1,)
    assert probabilities[0] == 1.0
    assert result.lagrange_multipliers["drawdown"] > 0.0
    assert result.lagrange_multipliers["turnover"] > 0.0
    assert result.adjusted_episode_rewards.iloc[-1] < result.episode_rewards.iloc[-1]


def test_full_demo_exposes_cross_sectional_intensity_and_policy_gradient_metrics():
    result = run_full_demo(seed=18).as_dict()
    assert result["factor_risk"]["cross_sectional_factor_count"] >= 2
    assert result["factor_risk"]["neutralized_style_exposure_norm"] < 1e-8
    assert "policy_gradient_last_reward" in result["rl_trading"]
    assert "constrained_pg_last_adjusted_reward" in result["rl_trading"]
    assert result["rl_trading"]["constrained_pg_drawdown_lambda"] >= 0.0
    assert result["rl_trading"]["constrained_pg_turnover_violation"] >= 0.0
    assert 0.0 < result["credit"]["logistic_hazard_stressed"] < 1.0
