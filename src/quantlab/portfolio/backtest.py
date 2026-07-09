from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

WeightFunction = Callable[[pd.DataFrame], np.ndarray]


@dataclass(frozen=True)
class PortfolioBacktestResult:
    history: pd.DataFrame
    weights: pd.DataFrame

    @property
    def total_return(self) -> float:
        return float(self.history["equity"].iloc[-1] / self.history["equity"].iloc[0] - 1.0)

    @property
    def max_drawdown(self) -> float:
        equity = self.history["equity"]
        return float((1.0 - equity / equity.cummax()).max())


def static_weight_backtest(
    returns: pd.DataFrame,
    weights: np.ndarray,
    initial_equity: float = 1.0,
    transaction_cost_bps: float = 0.0,
) -> PortfolioBacktestResult:
    """Backtest fixed portfolio weights over a return panel."""
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (returns.shape[1],):
        raise ValueError("weights length must match return columns")
    return rolling_rebalance_backtest(
        returns,
        weight_function=lambda _window: weights,
        lookback=1,
        rebalance_frequency=max(len(returns), 1),
        initial_equity=initial_equity,
        transaction_cost_bps=transaction_cost_bps,
    )


def rolling_rebalance_backtest(
    returns: pd.DataFrame,
    weight_function: WeightFunction,
    lookback: int = 60,
    rebalance_frequency: int = 21,
    initial_equity: float = 1.0,
    transaction_cost_bps: float = 0.0,
) -> PortfolioBacktestResult:
    """Run a rolling-window portfolio strategy with transaction costs."""
    if returns.empty:
        raise ValueError("returns cannot be empty")
    if lookback < 1 or rebalance_frequency < 1 or initial_equity <= 0 or transaction_cost_bps < 0:
        raise ValueError("invalid backtest parameters")
    costs = transaction_cost_bps / 10_000.0
    current_weights = np.full(returns.shape[1], 1.0 / returns.shape[1])
    equity = float(initial_equity)
    history_rows: list[dict[str, float]] = [{"equity": equity, "portfolio_return": 0.0, "transaction_cost": 0.0}]
    weight_rows = [pd.Series(current_weights, index=returns.columns, name=returns.index[0])]

    for idx, (date, row) in enumerate(returns.iterrows()):
        transaction_cost = 0.0
        if idx > 0 and idx >= lookback and (idx - lookback) % rebalance_frequency == 0:
            target = np.asarray(weight_function(returns.iloc[idx - lookback : idx]), dtype=float)
            if target.shape != current_weights.shape:
                raise ValueError("weight_function returned inconsistent shape")
            if not np.isfinite(target).all():
                raise ValueError("weight_function returned non-finite weights")
            target = target / target.sum()
            turnover = np.sum(np.abs(target - current_weights))
            transaction_cost = equity * turnover * costs
            equity -= transaction_cost
            current_weights = target

        portfolio_return = float(current_weights @ row.to_numpy(dtype=float))
        equity *= 1.0 + portfolio_return
        history_rows.append(
            {
                "equity": float(equity),
                "portfolio_return": portfolio_return,
                "transaction_cost": float(transaction_cost),
            }
        )
        weight_rows.append(pd.Series(current_weights, index=returns.columns, name=date))

    history = pd.DataFrame(history_rows, index=[returns.index[0]] + list(returns.index))
    weights_df = pd.DataFrame(weight_rows)
    return PortfolioBacktestResult(history=history, weights=weights_df)
