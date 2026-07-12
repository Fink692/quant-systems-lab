from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HazardCurve:
    maturities: np.ndarray
    hazard_rates: np.ndarray
    recovery_rate: float

    def survival(self, maturity: float) -> float:
        if maturity < 0:
            raise ValueError("maturity must be non-negative")
        elapsed = 0.0
        cumulative_hazard = 0.0
        for knot, hazard in zip(self.maturities, self.hazard_rates):
            interval_end = min(maturity, knot)
            if interval_end > elapsed:
                cumulative_hazard += hazard * (interval_end - elapsed)
                elapsed = interval_end
            if elapsed >= maturity:
                break
        if maturity > elapsed:
            cumulative_hazard += self.hazard_rates[-1] * (maturity - elapsed)
        return float(np.exp(-cumulative_hazard))

    def spread(self, maturity: float) -> float:
        idx = int(np.searchsorted(self.maturities, maturity, side="left"))
        idx = min(idx, len(self.hazard_rates) - 1)
        return float(self.hazard_rates[idx] * (1.0 - self.recovery_rate))


def bootstrap_hazard_curve(
    maturities: np.ndarray, credit_spreads: np.ndarray, recovery_rate: float = 0.4
) -> HazardCurve:
    """Bootstrap a simple piecewise-constant hazard curve from flat CDS-style spreads."""
    maturities = np.asarray(maturities, dtype=float)
    spreads = np.asarray(credit_spreads, dtype=float)
    if maturities.ndim != 1 or spreads.shape != maturities.shape:
        raise ValueError("maturities and credit_spreads must be one-dimensional arrays with the same shape")
    if np.any(np.diff(maturities) <= 0) or np.any(maturities <= 0):
        raise ValueError("maturities must be strictly increasing and positive")
    if np.any(spreads < 0):
        raise ValueError("credit_spreads must be non-negative")
    if not 0 <= recovery_rate < 1:
        raise ValueError("recovery_rate must be in [0, 1)")
    hazards = spreads / (1.0 - recovery_rate)
    return HazardCurve(maturities=maturities, hazard_rates=hazards, recovery_rate=recovery_rate)
