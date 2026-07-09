from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def ewma_covariance(returns: pd.DataFrame | np.ndarray, lambda_: float = 0.94) -> pd.DataFrame | np.ndarray:
    """Exponentially weighted covariance estimate."""
    if not 0 < lambda_ < 1:
        raise ValueError("lambda_ must be in (0, 1)")
    is_frame = isinstance(returns, pd.DataFrame)
    values = returns.to_numpy(dtype=float) if is_frame else np.asarray(returns, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("returns must be a two-dimensional matrix with at least two rows")
    demeaned = values - values.mean(axis=0)
    weights = (1.0 - lambda_) * lambda_ ** np.arange(values.shape[0] - 1, -1, -1)
    weights = weights / weights.sum()
    cov = (demeaned * weights[:, None]).T @ demeaned
    if is_frame:
        return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
    return cov


def ledoit_wolf_covariance(returns: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
    """Ledoit-Wolf shrinkage covariance estimate."""
    is_frame = isinstance(returns, pd.DataFrame)
    values = returns.to_numpy(dtype=float) if is_frame else np.asarray(returns, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("returns must be a two-dimensional matrix with at least two rows")
    estimator = LedoitWolf().fit(values)
    cov = estimator.covariance_
    if is_frame:
        return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
    return cov


def nearest_positive_semidefinite(matrix: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Project a symmetric matrix to the nearest PSD matrix by eigenvalue clipping."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    symmetric = 0.5 * (values + values.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalues, epsilon)
    return (eigenvectors * clipped) @ eigenvectors.T
