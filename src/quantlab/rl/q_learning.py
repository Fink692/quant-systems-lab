from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.rl.trading_env import TradingEnv, TradingState


@dataclass(frozen=True)
class QLearningResult:
    q_table: np.ndarray
    candidate_weights: np.ndarray
    bin_edges: np.ndarray
    episode_rewards: pd.Series

    def policy(self, state: TradingState, previous_price: float | None = None) -> float:
        if previous_price is None or previous_price <= 0:
            momentum = 0.0
        else:
            momentum = state.price / previous_price - 1.0
        state_index = _momentum_state(momentum, self.bin_edges)
        return float(self.candidate_weights[int(np.argmax(self.q_table[state_index]))])


def train_tabular_q_learning(
    prices: np.ndarray,
    candidate_weights: np.ndarray | None = None,
    episodes: int = 100,
    learning_rate: float = 0.2,
    discount_factor: float = 0.95,
    epsilon: float = 0.2,
    transaction_cost_bps: float = 1.0,
    seed: int | None = None,
) -> QLearningResult:
    """Train a compact tabular Q-learning trading baseline on momentum states."""
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1 or len(prices) < 5 or np.any(prices <= 0):
        raise ValueError("prices must be a positive one-dimensional array with at least five values")
    if min(episodes, learning_rate, discount_factor) <= 0 or not 0 <= epsilon <= 1:
        raise ValueError("invalid Q-learning hyperparameters")
    actions = np.array([-1.0, 0.0, 1.0]) if candidate_weights is None else np.asarray(candidate_weights, dtype=float)
    if actions.ndim != 1 or len(actions) == 0:
        raise ValueError("candidate_weights must be one-dimensional and non-empty")
    returns = prices[1:] / prices[:-1] - 1.0
    volatility = np.std(returns, ddof=1)
    edge = max(float(volatility), 1e-6)
    bin_edges = np.array([-edge, edge])
    q_table = np.zeros((3, len(actions)), dtype=float)
    rng = np.random.default_rng(seed)
    episode_rewards = []

    for episode in range(episodes):
        env = TradingEnv(prices, transaction_cost_bps=transaction_cost_bps)
        state = env.reset()
        previous_price = state.price
        state_index = 1
        done = False
        total_reward = 0.0
        while not done:
            if rng.random() < epsilon:
                action_index = int(rng.integers(0, len(actions)))
            else:
                action_index = int(np.argmax(q_table[state_index]))
            next_state, reward, done, _info = env.step(float(actions[action_index]))
            momentum = next_state.price / previous_price - 1.0
            next_state_index = _momentum_state(momentum, bin_edges)
            target = reward + (0.0 if done else discount_factor * float(np.max(q_table[next_state_index])))
            q_table[state_index, action_index] += learning_rate * (target - q_table[state_index, action_index])
            previous_price = next_state.price
            state_index = next_state_index
            total_reward += reward
        episode_rewards.append(total_reward)

    return QLearningResult(q_table, actions, bin_edges, pd.Series(episode_rewards, name="episode_reward"))


def _momentum_state(momentum: float, bin_edges: np.ndarray) -> int:
    return int(np.digitize([momentum], bin_edges)[0])
