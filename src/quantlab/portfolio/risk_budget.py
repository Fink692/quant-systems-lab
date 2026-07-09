from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def portfolio_risk_contributions(weights: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Return fractional portfolio variance contributions by asset."""
    weights = np.asarray(weights, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1] or weights.shape != (covariance.shape[0],):
        raise ValueError("weights and covariance shapes are inconsistent")
    variance = float(weights @ covariance @ weights)
    if variance <= 0:
        raise ValueError("portfolio variance must be positive")
    return weights * (covariance @ weights) / variance


def risk_budget_weights(
    covariance: np.ndarray,
    target_budget: np.ndarray,
    long_only: bool = True,
) -> np.ndarray:
    """Find weights whose variance contributions match a target risk budget."""
    covariance = np.asarray(covariance, dtype=float)
    target = np.asarray(target_budget, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1] or target.shape != (covariance.shape[0],):
        raise ValueError("covariance and target_budget shapes are inconsistent")
    if np.any(target < 0) or target.sum() <= 0:
        raise ValueError("target_budget must be non-negative with positive sum")
    target = target / target.sum()
    n_assets = covariance.shape[0]

    def objective(weights: np.ndarray) -> float:
        contributions = portfolio_risk_contributions(weights, covariance)
        return float(np.sum((contributions - target) ** 2))

    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}
    bounds = [(1e-10, 1.0)] * n_assets if long_only else None
    result = minimize(objective, np.full(n_assets, 1.0 / n_assets), bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(result.message)
    return result.x
