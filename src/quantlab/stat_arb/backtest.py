from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SpreadBacktestResult:
    history: pd.DataFrame

    @property
    def total_pnl(self) -> float:
        return float(self.history["cumulative_pnl"].iloc[-1])

    @property
    def hit_rate(self) -> float:
        pnl = self.history["pnl"].iloc[1:]
        if len(pnl) == 0:
            return 0.0
        return float((pnl > 0.0).mean())


def backtest_spread_strategy(
    spread: np.ndarray,
    positions: np.ndarray,
    transaction_cost: float = 0.0,
) -> SpreadBacktestResult:
    """Backtest a unit-notional mean-reversion spread strategy."""
    spread = np.asarray(spread, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if spread.ndim != 1 or positions.shape != spread.shape:
        raise ValueError("spread and positions must be one-dimensional arrays with the same shape")
    if transaction_cost < 0:
        raise ValueError("transaction_cost must be non-negative")

    pnl = np.zeros_like(spread)
    turnover = np.zeros_like(spread)
    for idx in range(1, len(spread)):
        pnl[idx] = positions[idx - 1] * (spread[idx] - spread[idx - 1])
        turnover[idx] = abs(positions[idx] - positions[idx - 1])
        pnl[idx] -= transaction_cost * turnover[idx]
    history = pd.DataFrame(
        {
            "spread": spread,
            "position": positions,
            "turnover": turnover,
            "pnl": pnl,
            "cumulative_pnl": np.cumsum(pnl),
        }
    )
    return SpreadBacktestResult(history)
