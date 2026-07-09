from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.stat_arb.kalman import kalman_dynamic_hedge_ratio
from quantlab.stat_arb.network import mean_reversion_signal


@dataclass(frozen=True)
class DynamicHedgeBacktestResult:
    history: pd.DataFrame

    @property
    def total_pnl(self) -> float:
        return float(self.history["cumulative_pnl"].iloc[-1])

    @property
    def turnover(self) -> float:
        return float(self.history["turnover"].sum())


def backtest_kalman_spread_strategy(
    y: pd.Series,
    x: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    window: int = 60,
    transaction_cost: float = 0.0,
) -> DynamicHedgeBacktestResult:
    """Backtest a dynamic Kalman-hedged spread with z-score signals."""
    if transaction_cost < 0:
        raise ValueError("transaction_cost must be non-negative")
    kalman = kalman_dynamic_hedge_ratio(y, x)
    signal = mean_reversion_signal(kalman.spread.to_numpy(), entry_z=entry_z, exit_z=exit_z, window=window)
    spread = kalman.spread.to_numpy(dtype=float)
    pnl = np.zeros_like(spread)
    turnover = np.zeros_like(spread)
    for idx in range(1, len(spread)):
        pnl[idx] = signal[idx - 1] * (spread[idx] - spread[idx - 1])
        turnover[idx] = abs(signal[idx] - signal[idx - 1])
        pnl[idx] -= transaction_cost * turnover[idx]
    history = pd.DataFrame(
        {
            "spread": spread,
            "signal": signal,
            "hedge_ratio": kalman.states["hedge_ratio"].to_numpy(),
            "turnover": turnover,
            "pnl": pnl,
            "cumulative_pnl": np.cumsum(pnl),
        },
        index=kalman.spread.index,
    )
    return DynamicHedgeBacktestResult(history)
