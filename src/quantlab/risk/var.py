from __future__ import annotations

import numpy as np
from scipy.stats import norm


def historical_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Historical value at risk as a positive loss number."""
    values = _validate_returns(returns)
    _validate_confidence(confidence)
    return float(-np.quantile(values, 1.0 - confidence))


def historical_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Historical conditional value at risk as a positive expected tail loss."""
    values = _validate_returns(returns)
    var_threshold = np.quantile(values, 1.0 - confidence)
    tail = values[values <= var_threshold]
    return float(-np.mean(tail))


def gaussian_var(mean: float, volatility: float, confidence: float = 0.95, horizon: float = 1.0) -> float:
    """Parametric Gaussian VaR for arithmetic returns."""
    if volatility < 0 or horizon <= 0:
        raise ValueError("volatility must be non-negative and horizon positive")
    _validate_confidence(confidence)
    return float(-(mean * horizon + volatility * np.sqrt(horizon) * norm.ppf(1.0 - confidence)))


def component_var(weights: np.ndarray, covariance: np.ndarray, confidence: float = 0.95) -> np.ndarray:
    """Approximate component VaR under a Gaussian portfolio return model."""
    weights = np.asarray(weights, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1] or weights.shape != (covariance.shape[0],):
        raise ValueError("weights and covariance shapes are inconsistent")
    _validate_confidence(confidence)
    portfolio_vol = np.sqrt(weights @ covariance @ weights)
    if portfolio_vol == 0:
        return np.zeros_like(weights)
    z = norm.ppf(confidence)
    marginal = covariance @ weights / portfolio_vol
    return z * weights * marginal


def _validate_returns(returns: np.ndarray) -> np.ndarray:
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("returns must be a non-empty one-dimensional array")
    return values


def _validate_confidence(confidence: float) -> None:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
