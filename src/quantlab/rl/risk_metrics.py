from __future__ import annotations

import numpy as np


def risk_adjusted_reward(
    period_return: float,
    drawdown: float,
    turnover: float = 0.0,
    drawdown_penalty: float = 1.0,
    turnover_penalty: float = 0.0,
) -> float:
    """Reward helper for constrained trading policies."""
    if drawdown < 0 or turnover < 0:
        raise ValueError("drawdown and turnover must be non-negative")
    return float(period_return - drawdown_penalty * drawdown - turnover_penalty * turnover)


def performance_summary(equity_curve: np.ndarray, periods_per_year: int = 252) -> dict[str, float]:
    """Compute common trading performance metrics from an equity curve."""
    equity = np.asarray(equity_curve, dtype=float)
    if equity.ndim != 1 or len(equity) < 2:
        raise ValueError("equity_curve must contain at least two values")
    if np.any(equity <= 0):
        raise ValueError("equity_curve must be positive")
    returns = equity[1:] / equity[:-1] - 1.0
    cumulative_max = np.maximum.accumulate(equity)
    drawdowns = 1.0 - equity / cumulative_max
    volatility = np.std(returns, ddof=1) * np.sqrt(periods_per_year) if len(returns) > 1 else 0.0
    downside = returns[returns < 0]
    downside_vol = np.std(downside, ddof=1) * np.sqrt(periods_per_year) if len(downside) > 1 else 0.0
    annual_return = (equity[-1] / equity[0]) ** (periods_per_year / (len(equity) - 1)) - 1.0
    sharpe = 0.0 if volatility == 0.0 else annual_return / volatility
    sortino = 0.0 if downside_vol == 0.0 else annual_return / downside_vol
    return {
        "total_return": float(equity[-1] / equity[0] - 1.0),
        "annual_return": float(annual_return),
        "annual_volatility": float(volatility),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(np.max(drawdowns)),
    }
