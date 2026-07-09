from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CIRIntensityParams:
    kappa: float
    theta: float
    sigma: float
    lambda0: float

    def validate(self) -> None:
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")
        if self.theta < 0 or self.lambda0 < 0:
            raise ValueError("theta and lambda0 must be non-negative")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")

    @property
    def feller_ratio(self) -> float:
        if self.sigma == 0.0:
            return np.inf
        return float(2.0 * self.kappa * self.theta / self.sigma**2)


@dataclass(frozen=True)
class CIRIntensitySimulationResult:
    times: np.ndarray
    intensities: pd.DataFrame
    integrated_hazard: pd.Series
    survival_probabilities: pd.Series
    default_times: pd.Series
    default_indicators: pd.Series

    @property
    def default_probability(self) -> float:
        return float(self.default_indicators.mean())

    @property
    def mean_survival_probability(self) -> float:
        return float(self.survival_probabilities.mean())

    @property
    def mean_terminal_intensity(self) -> float:
        return float(self.intensities.iloc[-1].mean())


def simulate_cir_intensity(
    params: CIRIntensityParams,
    maturity: float,
    steps: int = 252,
    paths: int = 10_000,
    seed: int | None = None,
) -> CIRIntensitySimulationResult:
    """Simulate CIR stochastic default intensities and first-jump default times."""
    params.validate()
    if maturity <= 0 or steps < 1 or paths < 1:
        raise ValueError("maturity, steps, and paths must be positive")

    rng = np.random.default_rng(seed)
    dt = maturity / steps
    times = np.linspace(0.0, maturity, steps + 1)
    intensities = np.empty((steps + 1, paths), dtype=float)
    intensities[0, :] = params.lambda0
    integrated = np.zeros(paths, dtype=float)
    thresholds = rng.exponential(1.0, size=paths)
    default_times = np.full(paths, np.nan, dtype=float)
    defaulted = np.zeros(paths, dtype=bool)

    for step in range(1, steps + 1):
        previous = np.maximum(intensities[step - 1, :], 0.0)
        shocks = rng.normal(size=paths)
        next_values = previous + params.kappa * (params.theta - previous) * dt + params.sigma * np.sqrt(previous) * np.sqrt(dt) * shocks
        next_values = np.maximum(next_values, 0.0)
        intensities[step, :] = next_values
        integrated += 0.5 * (previous + next_values) * dt
        newly_defaulted = (~defaulted) & (integrated >= thresholds)
        if newly_defaulted.any():
            default_times[newly_defaulted] = times[step]
            defaulted[newly_defaulted] = True

    columns = [f"path_{idx}" for idx in range(paths)]
    index = pd.Index(times, name="time")
    path_index = pd.Index(columns, name="path")
    return CIRIntensitySimulationResult(
        times=times,
        intensities=pd.DataFrame(intensities, index=index, columns=columns),
        integrated_hazard=pd.Series(integrated, index=path_index, name="integrated_hazard"),
        survival_probabilities=pd.Series(np.exp(-integrated), index=path_index, name="survival_probability"),
        default_times=pd.Series(default_times, index=path_index, name="default_time"),
        default_indicators=pd.Series(defaulted, index=path_index, name="defaulted"),
    )
