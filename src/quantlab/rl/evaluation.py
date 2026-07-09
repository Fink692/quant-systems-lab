from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.rl.trading_env import TradingEnv, TradingState

Policy = Callable[[TradingState], float]


@dataclass(frozen=True)
class BacktestResult:
    history: pd.DataFrame

    @property
    def total_return(self) -> float:
        equity = self.history["equity"]
        return float(equity.iloc[-1] / equity.iloc[0] - 1.0)

    @property
    def max_drawdown(self) -> float:
        equity = self.history["equity"]
        return float((1.0 - equity / equity.cummax()).max())

    @property
    def sharpe(self) -> float:
        returns = self.history["equity"].pct_change().dropna()
        if returns.std(ddof=1) == 0:
            return 0.0
        return float(np.sqrt(252.0) * returns.mean() / returns.std(ddof=1))


def constant_weight_policy(weight: float) -> Policy:
    return lambda _state: float(np.clip(weight, -1.0, 1.0))


def run_policy(env: TradingEnv, policy: Policy) -> BacktestResult:
    rows: list[dict[str, float]] = []
    state = env.reset()
    rows.append(_row_from_state(state, reward=0.0, transaction_cost=0.0, drawdown=0.0))
    done = False
    while not done:
        state, reward, done, info = env.step(policy(state))
        rows.append(
            _row_from_state(
                state,
                reward=reward,
                transaction_cost=info["transaction_cost"],
                drawdown=info["drawdown"],
            )
        )
    return BacktestResult(pd.DataFrame(rows))


def walk_forward_splits(length: int, train_size: int, test_size: int, step_size: int | None = None) -> Iterator[tuple[slice, slice]]:
    if min(length, train_size, test_size) <= 0:
        raise ValueError("length, train_size, and test_size must be positive")
    step = test_size if step_size is None else step_size
    if step <= 0:
        raise ValueError("step_size must be positive")
    start = 0
    while start + train_size + test_size <= length:
        yield slice(start, start + train_size), slice(start + train_size, start + train_size + test_size)
        start += step


def _row_from_state(state: TradingState, reward: float, transaction_cost: float, drawdown: float) -> dict[str, float]:
    return {
        "time_index": float(state.time_index),
        "price": float(state.price),
        "position": float(state.position),
        "cash": float(state.cash),
        "equity": float(state.equity),
        "reward": float(reward),
        "transaction_cost": float(transaction_cost),
        "drawdown": float(drawdown),
    }
