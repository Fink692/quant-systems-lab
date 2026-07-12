from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.rl.evaluation import walk_forward_splits
from quantlab.rl.q_learning import train_tabular_q_learning
from quantlab.rl.trading_env import TradingEnv


@dataclass(frozen=True)
class WalkForwardQLearningResult:
    folds: pd.DataFrame

    @property
    def mean_test_return(self) -> float:
        return float(self.folds["test_return"].mean())


def walk_forward_q_learning(
    prices: np.ndarray,
    train_size: int,
    test_size: int,
    candidate_weights: np.ndarray | None = None,
    episodes: int = 50,
    transaction_cost_bps: float = 1.0,
    seed: int | None = None,
) -> WalkForwardQLearningResult:
    """Train tabular Q-learning on rolling windows and evaluate out-of-sample."""
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1 or len(prices) < train_size + test_size:
        raise ValueError("prices must be one-dimensional and long enough for one split")
    rows = []
    for fold, (train_slice, test_slice) in enumerate(walk_forward_splits(len(prices), train_size, test_size)):
        train_prices = prices[train_slice]
        test_prices = prices[test_slice]
        model = train_tabular_q_learning(
            train_prices,
            candidate_weights=candidate_weights,
            episodes=episodes,
            transaction_cost_bps=transaction_cost_bps,
            seed=None if seed is None else seed + fold,
        )
        test_return = _evaluate_q_policy(
            test_prices, model.q_table, model.candidate_weights, model.bin_edges, transaction_cost_bps
        )
        rows.append(
            {
                "fold": fold,
                "train_start": train_slice.start,
                "train_stop": train_slice.stop,
                "test_start": test_slice.start,
                "test_stop": test_slice.stop,
                "test_return": test_return,
                "last_train_reward": float(model.episode_rewards.iloc[-1]),
            }
        )
    return WalkForwardQLearningResult(pd.DataFrame(rows))


def _evaluate_q_policy(
    prices: np.ndarray,
    q_table: np.ndarray,
    candidate_weights: np.ndarray,
    bin_edges: np.ndarray,
    transaction_cost_bps: float,
) -> float:
    env = TradingEnv(prices, transaction_cost_bps=transaction_cost_bps)
    state = env.reset()
    previous_price = state.price
    initial_equity = state.equity
    done = False
    while not done:
        momentum = state.price / previous_price - 1.0 if previous_price > 0 else 0.0
        state_index = int(np.digitize([momentum], bin_edges)[0])
        action = float(candidate_weights[int(np.argmax(q_table[state_index]))])
        previous_price = state.price
        state, _reward, done, _info = env.step(action)
    return float(state.equity / initial_equity - 1.0)
