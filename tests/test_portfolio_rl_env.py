import numpy as np
import pandas as pd

from quantlab.rl.portfolio_env import PortfolioTradingEnv, constant_mix_policy, momentum_rotation_policy, run_portfolio_policy
from quantlab.workflows.demo_suite import run_full_demo


def test_portfolio_trading_env_runs_constant_mix_policy():
    prices = pd.DataFrame(
        {
            "A": [100.0, 101.0, 102.0, 103.0],
            "B": [100.0, 99.0, 98.0, 97.0],
        }
    )
    result = run_portfolio_policy(
        PortfolioTradingEnv(prices, transaction_cost_bps=1.0),
        constant_mix_policy(pd.Series({"A": 0.6, "B": 0.4})),
    )
    assert len(result.history) == len(prices)
    assert len(result.weights) == len(prices)
    assert abs(result.weights.iloc[1].sum() - 1.0) < 1e-12
    assert result.history["transaction_cost"].iloc[1] > 0.0
    assert np.isfinite(result.total_return)


def test_portfolio_env_normalizes_levered_actions_and_reports_risk():
    prices = pd.DataFrame({"A": [100.0, 95.0, 90.0], "B": [100.0, 102.0, 104.0]})
    env = PortfolioTradingEnv(prices, transaction_cost_bps=0.0, drawdown_penalty=0.1, volatility_window=2)
    state = env.reset()
    next_state, reward, done, info = env.step(pd.Series({"A": 2.0, "B": 1.0}))
    assert not done
    assert abs(next_state.weights.sum() - 1.0) < 1e-12
    assert next_state.weights["A"] > next_state.weights["B"]
    assert info["turnover"] == 1.0
    assert np.isfinite(reward)


def test_momentum_rotation_policy_concentrates_after_lookback():
    prices = pd.DataFrame(
        {
            "A": [100.0, 101.0, 102.0, 103.0, 104.0],
            "B": [100.0, 99.0, 98.0, 97.0, 96.0],
            "C": [100.0, 100.5, 101.0, 101.5, 102.0],
        }
    )
    result = run_portfolio_policy(
        PortfolioTradingEnv(prices, transaction_cost_bps=0.0),
        momentum_rotation_policy(lookback=2, top_n=1),
    )
    concentrated = result.weights.iloc[-1]
    assert concentrated["A"] == 1.0
    assert concentrated[["B", "C"]].sum() == 0.0


def test_full_demo_exposes_multi_asset_rl_metrics():
    result = run_full_demo(seed=23).as_dict()
    assert "portfolio_constant_return" in result["rl_trading"]
    assert "portfolio_momentum_return" in result["rl_trading"]
    assert result["rl_trading"]["portfolio_momentum_turnover"] >= 0.0
    assert result["rl_trading"]["portfolio_momentum_max_drawdown"] >= 0.0
