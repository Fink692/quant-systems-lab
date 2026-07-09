from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.rl.trading_env import TradingEnv, TradingState


@dataclass(frozen=True)
class DeepQLearningResult:
    input_weights: np.ndarray
    hidden_bias: np.ndarray
    output_weights: np.ndarray
    output_bias: np.ndarray
    candidate_weights: np.ndarray
    feature_scale: float
    episode_rewards: pd.Series
    training_losses: pd.Series

    def q_values(self, state: TradingState, previous_price: float | None = None) -> np.ndarray:
        features = _state_features(state, previous_price, self.feature_scale)
        _hidden, q_values = _forward(features, self.input_weights, self.hidden_bias, self.output_weights, self.output_bias)
        return q_values

    def policy(self, state: TradingState, previous_price: float | None = None) -> float:
        q_values = self.q_values(state, previous_price)
        return float(self.candidate_weights[int(np.argmax(q_values))])


def train_deep_q_learning(
    prices: np.ndarray,
    candidate_weights: np.ndarray | None = None,
    episodes: int = 60,
    hidden_units: int = 8,
    learning_rate: float = 0.01,
    discount_factor: float = 0.95,
    epsilon: float = 0.2,
    epsilon_decay: float = 0.98,
    min_epsilon: float = 0.02,
    transaction_cost_bps: float = 1.0,
    drawdown_penalty: float = 0.0,
    replay_capacity: int = 512,
    batch_size: int = 16,
    seed: int | None = None,
) -> DeepQLearningResult:
    """Train a compact neural Q-learning trading agent with replay."""
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1 or len(prices) < 5 or np.any(prices <= 0):
        raise ValueError("prices must be a positive one-dimensional array with at least five values")
    if min(episodes, hidden_units, learning_rate, discount_factor, replay_capacity, batch_size) <= 0:
        raise ValueError("training hyperparameters must be positive")
    if discount_factor > 1 or not 0 <= epsilon <= 1 or not 0 <= min_epsilon <= 1 or not 0 < epsilon_decay <= 1:
        raise ValueError("invalid exploration or discount parameters")

    actions = np.array([-1.0, 0.0, 1.0]) if candidate_weights is None else np.asarray(candidate_weights, dtype=float)
    if actions.ndim != 1 or len(actions) == 0:
        raise ValueError("candidate_weights must be one-dimensional and non-empty")
    if np.any(np.abs(actions) > 1.0):
        raise ValueError("candidate_weights must be bounded by leverage one")

    log_returns = np.diff(np.log(prices))
    feature_scale = max(float(np.std(log_returns, ddof=1)), 1e-6)
    rng = np.random.default_rng(seed)
    input_dim = 3
    w1 = rng.normal(0.0, 1.0 / np.sqrt(input_dim), size=(input_dim, hidden_units))
    b1 = np.zeros(hidden_units, dtype=float)
    w2 = rng.normal(0.0, 1.0 / np.sqrt(hidden_units), size=(hidden_units, len(actions)))
    b2 = np.zeros(len(actions), dtype=float)

    replay: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []
    episode_rewards: list[float] = []
    losses: list[float] = []

    for episode in range(episodes):
        env = TradingEnv(prices, transaction_cost_bps=transaction_cost_bps, drawdown_penalty=drawdown_penalty)
        state = env.reset()
        previous_price: float | None = None
        done = False
        total_reward = 0.0
        current_epsilon = max(min_epsilon, epsilon * epsilon_decay**episode)

        while not done:
            features = _state_features(state, previous_price, feature_scale)
            _hidden, q_values = _forward(features, w1, b1, w2, b2)
            if rng.random() < current_epsilon:
                action_index = int(rng.integers(0, len(actions)))
            else:
                action_index = int(np.argmax(q_values))

            next_state, reward, done, _info = env.step(float(actions[action_index]))
            next_features = _state_features(next_state, state.price, feature_scale)
            replay.append((features, action_index, float(reward), next_features, bool(done)))
            if len(replay) > replay_capacity:
                replay.pop(0)

            sample_size = min(batch_size, len(replay))
            sample_indices = rng.choice(len(replay), size=sample_size, replace=False)
            batch_loss = 0.0
            for idx in sample_indices:
                transition = replay[int(idx)]
                loss = _update_network(transition, w1, b1, w2, b2, learning_rate, discount_factor)
                batch_loss += loss
            losses.append(batch_loss / sample_size)

            previous_price = state.price
            state = next_state
            total_reward += reward

        episode_rewards.append(float(total_reward))

    return DeepQLearningResult(
        input_weights=w1,
        hidden_bias=b1,
        output_weights=w2,
        output_bias=b2,
        candidate_weights=actions,
        feature_scale=feature_scale,
        episode_rewards=pd.Series(episode_rewards, name="episode_reward"),
        training_losses=pd.Series(losses, name="training_loss"),
    )


def _state_features(state: TradingState, previous_price: float | None, feature_scale: float) -> np.ndarray:
    if previous_price is None or previous_price <= 0:
        momentum = 0.0
    else:
        momentum = np.log(state.price / previous_price) / feature_scale
    position_weight = 0.0 if state.equity == 0 else state.position * state.price / state.equity
    drawdown = 0.0 if state.peak_equity <= 0 else max(1.0 - state.equity / state.peak_equity, 0.0)
    return np.array([momentum, np.clip(position_weight, -1.0, 1.0), drawdown], dtype=float)


def _forward(
    features: np.ndarray,
    input_weights: np.ndarray,
    hidden_bias: np.ndarray,
    output_weights: np.ndarray,
    output_bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.tanh(features @ input_weights + hidden_bias)
    q_values = hidden @ output_weights + output_bias
    return hidden, q_values


def _update_network(
    transition: tuple[np.ndarray, int, float, np.ndarray, bool],
    input_weights: np.ndarray,
    hidden_bias: np.ndarray,
    output_weights: np.ndarray,
    output_bias: np.ndarray,
    learning_rate: float,
    discount_factor: float,
) -> float:
    features, action_index, reward, next_features, done = transition
    hidden, q_values = _forward(features, input_weights, hidden_bias, output_weights, output_bias)
    _next_hidden, next_q_values = _forward(next_features, input_weights, hidden_bias, output_weights, output_bias)
    target = reward if done else reward + discount_factor * float(np.max(next_q_values))
    td_error = float(np.clip(q_values[action_index] - target, -10.0, 10.0))

    grad_q = np.zeros_like(q_values)
    grad_q[action_index] = td_error
    grad_w2 = np.outer(hidden, grad_q)
    grad_b2 = grad_q
    grad_hidden = output_weights @ grad_q
    grad_hidden_pre_activation = grad_hidden * (1.0 - hidden**2)
    grad_w1 = np.outer(features, grad_hidden_pre_activation)
    grad_b1 = grad_hidden_pre_activation

    input_weights -= learning_rate * np.clip(grad_w1, -5.0, 5.0)
    hidden_bias -= learning_rate * np.clip(grad_b1, -5.0, 5.0)
    output_weights -= learning_rate * np.clip(grad_w2, -5.0, 5.0)
    output_bias -= learning_rate * np.clip(grad_b2, -5.0, 5.0)
    return 0.5 * td_error**2
