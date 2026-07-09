from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.stattools import adfuller


@dataclass(frozen=True)
class CointegrationResult:
    hedge_ratio: float
    intercept: float
    adf_statistic: float
    p_value: float
    spread: np.ndarray


def engle_granger(y: np.ndarray, x: np.ndarray) -> CointegrationResult:
    """Run a two-series Engle-Granger cointegration test."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.shape != x.shape or y.ndim != 1:
        raise ValueError("y and x must be one-dimensional arrays with the same shape")
    design = np.column_stack([np.ones_like(x), x])
    intercept, hedge_ratio = np.linalg.lstsq(design, y, rcond=None)[0]
    spread = y - intercept - hedge_ratio * x
    adf_statistic, p_value, *_ = adfuller(spread, autolag="AIC")
    return CointegrationResult(float(hedge_ratio), float(intercept), float(adf_statistic), float(p_value), spread)


def estimate_ou(spread: np.ndarray, dt: float = 1.0) -> dict[str, float]:
    """Estimate Ornstein-Uhlenbeck parameters from a spread time series."""
    values = np.asarray(spread, dtype=float)
    if values.ndim != 1 or len(values) < 3:
        raise ValueError("spread must be a one-dimensional array with at least three observations")
    if dt <= 0:
        raise ValueError("dt must be positive")

    x_t = values[:-1]
    x_next = values[1:]
    design = np.column_stack([np.ones_like(x_t), x_t])
    intercept, phi = np.linalg.lstsq(design, x_next, rcond=None)[0]
    phi = float(np.clip(phi, 1e-8, 0.999999))
    theta = -np.log(phi) / dt
    mu = intercept / (1.0 - phi)
    residuals = x_next - intercept - phi * x_t
    sigma = np.std(residuals, ddof=1) * np.sqrt(2.0 * theta / (1.0 - phi**2))
    half_life = np.log(2.0) / theta
    return {"theta": float(theta), "mu": float(mu), "sigma": float(sigma), "half_life": float(half_life)}


def zscore(values: np.ndarray, window: int = 60) -> np.ndarray:
    """Rolling z-score with NaNs before the first full window."""
    values = np.asarray(values, dtype=float)
    if window < 2:
        raise ValueError("window must be at least 2")
    scores = np.full_like(values, np.nan, dtype=float)
    for idx in range(window - 1, len(values)):
        sample = values[idx - window + 1 : idx + 1]
        std = np.std(sample, ddof=1)
        scores[idx] = 0.0 if std == 0 else (values[idx] - np.mean(sample)) / std
    return scores
