from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class FillIntensityCalibration:
    base_intensity: float
    distance_decay: float
    log_likelihood: float
    success: bool
    message: str

    def fill_probability(self, quote_distance: float, horizon: float) -> float:
        if quote_distance < 0 or horizon < 0:
            raise ValueError("quote_distance and horizon must be non-negative")
        intensity = self.base_intensity * np.exp(-self.distance_decay * quote_distance)
        return float(1.0 - np.exp(-intensity * horizon))


def calibrate_fill_intensity(
    quote_distances: np.ndarray,
    horizons: np.ndarray,
    fills: np.ndarray,
    regularization: float = 1e-4,
) -> FillIntensityCalibration:
    """Fit a Poisson fill model p(fill)=1-exp(-lambda*exp(-k*d)*T)."""
    distances = np.asarray(quote_distances, dtype=float)
    times = np.asarray(horizons, dtype=float)
    observed = np.asarray(fills, dtype=bool)
    if distances.shape != times.shape or observed.shape != distances.shape or distances.ndim != 1:
        raise ValueError("quote_distances, horizons, and fills must be same-length one-dimensional arrays")
    if np.any(distances < 0) or np.any(times < 0):
        raise ValueError("quote distances and horizons must be non-negative")
    if regularization < 0:
        raise ValueError("regularization must be non-negative")

    def negative_log_likelihood(raw: np.ndarray) -> float:
        base = np.exp(raw[0])
        decay = np.exp(raw[1])
        probabilities = 1.0 - np.exp(-base * np.exp(-decay * distances) * times)
        probabilities = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
        log_likelihood = np.sum(observed * np.log(probabilities) + (~observed) * np.log(1.0 - probabilities))
        return float(-log_likelihood + regularization * np.sum(raw**2))

    result = minimize(negative_log_likelihood, np.log([1.0, 1.0]), method="Nelder-Mead")
    base = float(np.exp(result.x[0]))
    decay = float(np.exp(result.x[1]))
    return FillIntensityCalibration(
        base_intensity=base,
        distance_decay=decay,
        log_likelihood=float(-result.fun),
        success=bool(result.success),
        message=str(result.message),
    )
