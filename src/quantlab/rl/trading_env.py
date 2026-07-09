from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TradingState:
    time_index: int
    price: float
    position: float
    cash: float
    equity: float
    peak_equity: float


class TradingEnv:
    """Small dependency-free trading environment for policy experiments."""

    def __init__(
        self,
        prices: np.ndarray,
        initial_cash: float = 1_000_000.0,
        transaction_cost_bps: float = 1.0,
        drawdown_penalty: float = 0.0,
    ) -> None:
        prices = np.asarray(prices, dtype=float)
        if prices.ndim != 1 or len(prices) < 2:
            raise ValueError("prices must be a one-dimensional array with at least two values")
        if np.any(prices <= 0):
            raise ValueError("prices must be positive")
        self.prices = prices
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost_bps / 10_000.0
        self.drawdown_penalty = drawdown_penalty
        self.reset()

    def reset(self) -> TradingState:
        self.time_index = 0
        self.cash = float(self.initial_cash)
        self.position = 0.0
        self.peak_equity = float(self.initial_cash)
        return self.state

    @property
    def state(self) -> TradingState:
        price = float(self.prices[self.time_index])
        equity = self.cash + self.position * price
        return TradingState(self.time_index, price, self.position, self.cash, equity, self.peak_equity)

    def step(self, target_weight: float) -> tuple[TradingState, float, bool, dict[str, float]]:
        """Trade to target risky-asset weight, then advance one price step."""
        target_weight = float(np.clip(target_weight, -1.0, 1.0))
        old_state = self.state
        old_equity = old_state.equity
        target_position = target_weight * old_equity / old_state.price
        trade_quantity = target_position - self.position
        trade_value = trade_quantity * old_state.price
        costs = abs(trade_value) * self.transaction_cost
        self.cash -= trade_value + costs
        self.position = target_position

        self.time_index += 1
        new_state = self.state
        self.peak_equity = max(self.peak_equity, new_state.equity)
        drawdown = 1.0 - new_state.equity / self.peak_equity if self.peak_equity > 0 else 0.0
        reward = (new_state.equity - old_equity) / old_equity - self.drawdown_penalty * drawdown
        done = self.time_index == len(self.prices) - 1
        info = {"transaction_cost": float(costs), "drawdown": float(drawdown)}
        return self.state, float(reward), done, info
