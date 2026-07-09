from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BlackLittermanResult:
    posterior_returns: np.ndarray
    posterior_covariance: np.ndarray
    equilibrium_returns: np.ndarray


def black_litterman_posterior(
    covariance: np.ndarray,
    market_weights: np.ndarray,
    views_matrix: np.ndarray,
    views: np.ndarray,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    omega: np.ndarray | None = None,
) -> BlackLittermanResult:
    """Compute Black-Litterman posterior returns and covariance."""
    sigma = np.asarray(covariance, dtype=float)
    weights = np.asarray(market_weights, dtype=float)
    p = np.asarray(views_matrix, dtype=float)
    q = np.asarray(views, dtype=float)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError("covariance must be square")
    if weights.shape != (sigma.shape[0],):
        raise ValueError("market_weights shape must match covariance")
    if p.shape[1] != sigma.shape[0] or q.shape != (p.shape[0],):
        raise ValueError("views dimensions are inconsistent")
    if risk_aversion <= 0 or tau <= 0:
        raise ValueError("risk_aversion and tau must be positive")

    pi = risk_aversion * sigma @ weights
    tau_sigma = tau * sigma
    omega_matrix = np.diag(np.diag(p @ tau_sigma @ p.T)) if omega is None else np.asarray(omega, dtype=float)
    middle = np.linalg.inv(p @ tau_sigma @ p.T + omega_matrix)
    posterior_mean = pi + tau_sigma @ p.T @ middle @ (q - p @ pi)
    posterior_cov = sigma + tau_sigma - tau_sigma @ p.T @ middle @ p @ tau_sigma
    return BlackLittermanResult(posterior_mean, posterior_cov, pi)
