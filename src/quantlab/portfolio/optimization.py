from __future__ import annotations

import numpy as np
from scipy.optimize import linprog, minimize


def _as_covariance(covariance: np.ndarray) -> np.ndarray:
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be a square matrix")
    return cov


def min_variance_weights(covariance: np.ndarray, long_only: bool = True) -> np.ndarray:
    cov = _as_covariance(covariance)
    n_assets = cov.shape[0]
    if not long_only:
        inv = np.linalg.pinv(cov)
        ones = np.ones(n_assets)
        return (inv @ ones) / (ones @ inv @ ones)

    objective = lambda weights: float(weights @ cov @ weights)
    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}
    result = minimize(objective, np.full(n_assets, 1.0 / n_assets), bounds=[(0.0, 1.0)] * n_assets, constraints=constraints)
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def mean_variance_weights(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    risk_aversion: float = 1.0,
    long_only: bool = True,
) -> np.ndarray:
    cov = _as_covariance(covariance)
    mu = np.asarray(expected_returns, dtype=float)
    if mu.shape != (cov.shape[0],):
        raise ValueError("expected_returns shape must match covariance")
    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive")

    objective = lambda weights: float(0.5 * risk_aversion * weights @ cov @ weights - mu @ weights)
    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}
    bounds = [(0.0, 1.0)] * len(mu) if long_only else None
    result = minimize(objective, np.full(len(mu), 1.0 / len(mu)), bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def risk_parity_weights(covariance: np.ndarray, long_only: bool = True) -> np.ndarray:
    cov = _as_covariance(covariance)
    n_assets = cov.shape[0]

    def objective(weights: np.ndarray) -> float:
        portfolio_var = weights @ cov @ weights
        marginal = cov @ weights
        contributions = weights * marginal / portfolio_var
        target = np.full(n_assets, 1.0 / n_assets)
        return float(np.sum((contributions - target) ** 2))

    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}
    bounds = [(1e-10, 1.0)] * n_assets if long_only else None
    result = minimize(objective, np.full(n_assets, 1.0 / n_assets), bounds=bounds, constraints=constraints)
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def cvar_weights(returns: np.ndarray, alpha: float = 0.95, long_only: bool = True) -> np.ndarray:
    """Minimize empirical CVaR with a linear program."""
    scenarios = np.asarray(returns, dtype=float)
    if scenarios.ndim != 2:
        raise ValueError("returns must be a scenario-by-asset matrix")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    n_scenarios, n_assets = scenarios.shape
    n_variables = n_assets + 1 + n_scenarios
    weight_slice = slice(0, n_assets)
    eta_index = n_assets
    u_slice = slice(n_assets + 1, n_variables)

    objective = np.zeros(n_variables)
    objective[eta_index] = 1.0
    objective[u_slice] = 1.0 / ((1.0 - alpha) * n_scenarios)

    a_eq = np.zeros((1, n_variables))
    a_eq[0, weight_slice] = 1.0
    b_eq = np.array([1.0])

    a_ub = np.zeros((n_scenarios, n_variables))
    b_ub = np.zeros(n_scenarios)
    for idx in range(n_scenarios):
        a_ub[idx, weight_slice] = -scenarios[idx]
        a_ub[idx, eta_index] = -1.0
        a_ub[idx, n_assets + 1 + idx] = -1.0

    bounds = [(0.0, 1.0)] * n_assets if long_only else [(None, None)] * n_assets
    bounds += [(None, None)] + [(0.0, None)] * n_scenarios

    result = linprog(objective, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    return result.x[weight_slice]
