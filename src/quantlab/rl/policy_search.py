from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.rl.evaluation import constant_weight_policy, run_policy
from quantlab.rl.trading_env import TradingEnv


@dataclass(frozen=True)
class PolicySearchResult:
    best_weight: float
    results: pd.DataFrame


def grid_search_constant_weight_policy(
    prices: np.ndarray,
    candidate_weights: np.ndarray | None = None,
    transaction_cost_bps: float = 1.0,
    drawdown_penalty: float = 0.0,
    objective: str = "sharpe",
) -> PolicySearchResult:
    """Search constant target weights as a simple constrained RL baseline."""
    if objective not in {"sharpe", "total_return", "drawdown_adjusted"}:
        raise ValueError("objective must be 'sharpe', 'total_return', or 'drawdown_adjusted'")
    weights = np.linspace(-1.0, 1.0, 21) if candidate_weights is None else np.asarray(candidate_weights, dtype=float)
    rows = []
    for weight in weights:
        env = TradingEnv(prices, transaction_cost_bps=transaction_cost_bps, drawdown_penalty=drawdown_penalty)
        backtest = run_policy(env, constant_weight_policy(float(weight)))
        score = {
            "sharpe": backtest.sharpe,
            "total_return": backtest.total_return,
            "drawdown_adjusted": backtest.total_return - backtest.max_drawdown,
        }[objective]
        rows.append(
            {
                "weight": float(weight),
                "score": float(score),
                "total_return": backtest.total_return,
                "max_drawdown": backtest.max_drawdown,
                "sharpe": backtest.sharpe,
            }
        )
    results = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return PolicySearchResult(best_weight=float(results.loc[0, "weight"]), results=results)
