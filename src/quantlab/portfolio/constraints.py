from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def turnover_constrained_mean_variance_weights(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    previous_weights: np.ndarray,
    max_turnover: float,
    risk_aversion: float = 1.0,
    long_only: bool = True,
) -> np.ndarray:
    """Mean-variance optimization with an L1 turnover cap."""
    mu = np.asarray(expected_returns, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    previous = np.asarray(previous_weights, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or mu.shape != (cov.shape[0],) or previous.shape != mu.shape:
        raise ValueError("expected_returns, covariance, and previous_weights shapes are inconsistent")
    if max_turnover < 0 or risk_aversion <= 0:
        raise ValueError("max_turnover must be non-negative and risk_aversion positive")
    if abs(previous.sum() - 1.0) > 1e-8:
        raise ValueError("previous_weights must sum to 1")

    def objective(weights: np.ndarray) -> float:
        return float(0.5 * risk_aversion * weights @ cov @ weights - mu @ weights)

    constraints = [
        {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
        {"type": "ineq", "fun": lambda weights: max_turnover - np.sum(np.abs(weights - previous))},
    ]
    bounds = [(0.0, 1.0)] * len(mu) if long_only else None
    result = minimize(objective, previous.copy(), bounds=bounds, constraints=constraints, method="SLSQP")
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def apply_weight_bounds(weights: np.ndarray, lower: float = 0.0, upper: float = 1.0) -> np.ndarray:
    """Project weights to simple bounds and renormalize."""
    if lower > upper:
        raise ValueError("lower cannot exceed upper")
    clipped = np.clip(np.asarray(weights, dtype=float), lower, upper)
    total = clipped.sum()
    if total == 0:
        raise ValueError("bounded weights sum to zero")
    return clipped / total
