from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExponentialHazardFit:
    hazard_rate: float
    log_likelihood: float
    observations: int
    defaults: int
    total_time: float

    def survival_probability(self, maturity: float) -> float:
        if maturity < 0:
            raise ValueError("maturity must be non-negative")
        return float(np.exp(-self.hazard_rate * maturity))

    def default_probability(self, maturity: float) -> float:
        return float(1.0 - self.survival_probability(maturity))


def fit_exponential_hazard(durations: np.ndarray, default_observed: np.ndarray) -> ExponentialHazardFit:
    """Maximum-likelihood exponential hazard fit with right-censored observations."""
    durations = np.asarray(durations, dtype=float)
    observed = np.asarray(default_observed, dtype=bool)
    if durations.ndim != 1 or observed.shape != durations.shape:
        raise ValueError("durations and default_observed must be one-dimensional arrays with the same shape")
    if np.any(durations <= 0):
        raise ValueError("durations must be positive")
    total_time = float(np.sum(durations))
    defaults = int(np.sum(observed))
    hazard = defaults / total_time if defaults > 0 else 0.0
    if hazard == 0.0:
        log_likelihood = 0.0
    else:
        log_likelihood = defaults * np.log(hazard) - hazard * total_time
    return ExponentialHazardFit(
        hazard_rate=float(hazard),
        log_likelihood=float(log_likelihood),
        observations=int(len(durations)),
        defaults=defaults,
        total_time=total_time,
    )
