from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class EllipsoidalRobustResult:
    weights: np.ndarray
    nominal_return: float
    worst_case_return: float
    variance: float
    uncertainty_penalty: float
    objective_value: float
    success: bool
    message: str


def robust_mean_variance_weights(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    risk_aversion: float = 1.0,
    uncertainty_penalty: float = 0.5,
    long_only: bool = True,
) -> np.ndarray:
    """Mean-variance optimizer with an L2 penalty for expected-return uncertainty."""
    mu = np.asarray(expected_returns, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or mu.shape != (cov.shape[0],):
        raise ValueError("expected_returns and covariance shapes are inconsistent")
    if risk_aversion <= 0 or uncertainty_penalty < 0:
        raise ValueError("risk_aversion must be positive and uncertainty_penalty non-negative")

    def objective(weights: np.ndarray) -> float:
        return float(
            0.5 * risk_aversion * weights @ cov @ weights
            - mu @ weights
            + uncertainty_penalty * np.linalg.norm(weights, ord=2)
        )

    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}
    bounds = [(0.0, 1.0)] * len(mu) if long_only else None
    result = minimize(objective, np.full(len(mu), 1.0 / len(mu)), constraints=constraints, bounds=bounds)
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def ellipsoidal_robust_mean_variance_weights(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    mean_uncertainty: np.ndarray,
    uncertainty_radius: float = 1.0,
    risk_aversion: float = 1.0,
    long_only: bool = True,
) -> EllipsoidalRobustResult:
    """Mean-variance optimizer with ellipsoidal expected-return uncertainty."""
    mu = np.asarray(expected_returns, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    uncertainty = np.asarray(mean_uncertainty, dtype=float)
    if uncertainty.ndim == 1:
        uncertainty = np.diag(uncertainty)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or mu.shape != (cov.shape[0],):
        raise ValueError("expected_returns and covariance shapes are inconsistent")
    if uncertainty.shape != cov.shape:
        raise ValueError("mean_uncertainty must match covariance shape")
    if risk_aversion <= 0 or uncertainty_radius < 0:
        raise ValueError("risk_aversion must be positive and uncertainty_radius non-negative")
    if not np.allclose(cov, cov.T) or not np.allclose(uncertainty, uncertainty.T):
        raise ValueError("covariance and mean_uncertainty must be symmetric")
    if np.min(np.linalg.eigvalsh(cov)) < -1e-10 or np.min(np.linalg.eigvalsh(uncertainty)) < -1e-10:
        raise ValueError("covariance and mean_uncertainty must be positive semidefinite")

    def uncertainty_penalty(weights: np.ndarray) -> float:
        return float(uncertainty_radius * np.sqrt(max(weights @ uncertainty @ weights, 0.0)))

    def objective(weights: np.ndarray) -> float:
        return float(0.5 * risk_aversion * weights @ cov @ weights - mu @ weights + uncertainty_penalty(weights))

    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}
    bounds = [(0.0, 1.0)] * len(mu) if long_only else None
    result = minimize(
        objective,
        np.full(len(mu), 1.0 / len(mu)),
        constraints=constraints,
        bounds=bounds,
        method="SLSQP",
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(result.message)
    weights = np.asarray(result.x, dtype=float)
    nominal_return = float(mu @ weights)
    penalty = uncertainty_penalty(weights)
    variance = float(weights @ cov @ weights)
    return EllipsoidalRobustResult(
        weights=weights,
        nominal_return=nominal_return,
        worst_case_return=float(nominal_return - penalty),
        variance=variance,
        uncertainty_penalty=penalty,
        objective_value=float(result.fun),
        success=bool(result.success),
        message=str(result.message),
    )


def resampled_efficient_weights(
    returns: np.ndarray,
    n_resamples: int = 100,
    risk_aversion: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Michaud-style resampled mean-variance weights using bootstrap samples."""
    scenarios = np.asarray(returns, dtype=float)
    if scenarios.ndim != 2:
        raise ValueError("returns must be a scenario-by-asset matrix")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    from quantlab.portfolio.optimization import mean_variance_weights

    rng = np.random.default_rng(seed)
    weights = []
    for _ in range(n_resamples):
        sample = scenarios[rng.integers(0, len(scenarios), size=len(scenarios))]
        covariance = np.cov(sample, rowvar=False)
        covariance += np.eye(covariance.shape[0]) * 1e-10
        weights.append(mean_variance_weights(sample.mean(axis=0), covariance, risk_aversion=risk_aversion))
    return np.mean(weights, axis=0)
