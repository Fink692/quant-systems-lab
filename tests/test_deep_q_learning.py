import numpy as np
import pytest

from quantlab.rl.deep_q import train_deep_q_learning
from quantlab.rl.trading_env import TradingEnv
from quantlab.workflows.demo_suite import run_full_demo


def test_deep_q_learning_trains_neural_action_values():
    prices = 100.0 * np.exp(np.linspace(0.0, 0.08, 50))
    result = train_deep_q_learning(
        prices,
        episodes=5,
        hidden_units=5,
        learning_rate=0.01,
        epsilon=0.2,
        batch_size=8,
        seed=3,
    )
    state = TradingEnv(prices).reset()
    q_values = result.q_values(state)

    assert result.input_weights.shape == (3, 5)
    assert result.output_weights.shape == (5, 3)
    assert len(result.episode_rewards) == 5
    assert len(result.training_losses) > 0
    assert np.all(np.isfinite(q_values))
    assert np.isfinite(result.training_losses.iloc[-1])
    assert result.policy(state) in {-1.0, 0.0, 1.0}


def test_deep_q_learning_validates_candidate_weight_bounds():
    with pytest.raises(ValueError, match="bounded"):
        train_deep_q_learning(
            np.linspace(100.0, 105.0, 10),
            candidate_weights=np.array([-1.5, 0.0, 1.0]),
            episodes=2,
        )


def test_full_demo_exposes_deep_q_metrics():
    result = run_full_demo(seed=29).as_dict()

    assert "deep_q_last_reward" in result["rl_trading"]
    assert result["rl_trading"]["deep_q_last_loss"] >= 0.0
    assert result["rl_trading"]["deep_q_initial_action"] in {-1.0, 0.0, 1.0}
