from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DrawdownSummary:
    equity: pd.Series
    drawdown: pd.Series
    max_drawdown: float
    average_drawdown: float
    conditional_drawdown_at_risk: float
    time_under_water: int


def conditional_drawdown_at_risk(equity_curve: np.ndarray | pd.Series, confidence: float = 0.95) -> DrawdownSummary:
    """Summarize path risk using drawdowns and empirical CDaR."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    equity = pd.Series(equity_curve, dtype=float).reset_index(drop=True)
    if len(equity) < 2 or (equity <= 0).any():
        raise ValueError("equity_curve must contain at least two positive values")

    running_peak = equity.cummax()
    drawdown = (1.0 - equity / running_peak).rename("drawdown")
    threshold = float(drawdown.quantile(confidence))
    tail = drawdown[drawdown >= threshold]
    cdar = float(tail.mean()) if len(tail) else threshold
    return DrawdownSummary(
        equity=equity.rename("equity"),
        drawdown=drawdown,
        max_drawdown=float(drawdown.max()),
        average_drawdown=float(drawdown.mean()),
        conditional_drawdown_at_risk=cdar,
        time_under_water=int((drawdown > 0.0).sum()),
    )


def portfolio_drawdown_summary(
    returns: pd.DataFrame | np.ndarray,
    weights: np.ndarray,
    confidence: float = 0.95,
    initial_value: float = 1.0,
) -> DrawdownSummary:
    """Build an equity curve from weighted returns and compute drawdown risk."""
    if initial_value <= 0:
        raise ValueError("initial_value must be positive")
    scenarios = np.asarray(returns, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if scenarios.ndim != 2 or weights.shape != (scenarios.shape[1],):
        raise ValueError("returns must be scenario-by-asset and weights must match asset count")
    portfolio_returns = scenarios @ weights
    if np.any(1.0 + portfolio_returns <= 0):
        raise ValueError("portfolio returns imply non-positive equity")
    equity = initial_value * np.cumprod(1.0 + portfolio_returns)
    return conditional_drawdown_at_risk(equity, confidence=confidence)
