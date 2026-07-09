from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ATMSkewPowerLawFit:
    hurst: float
    level: float
    skew_sign: float
    slope: float
    intercept: float
    r_squared: float

    def predict_skew(self, maturities: np.ndarray) -> np.ndarray:
        times = np.asarray(maturities, dtype=float)
        if np.any(times <= 0):
            raise ValueError("maturities must be positive")
        return self.skew_sign * self.level * times ** (self.hurst - 0.5)


def fit_atm_skew_power_law(maturities: np.ndarray, atm_skews: np.ndarray) -> ATMSkewPowerLawFit:
    """Estimate rough-volatility H from ATM skew scaling: skew(T) ~ T^(H - 1/2)."""
    times = np.asarray(maturities, dtype=float)
    skews = np.asarray(atm_skews, dtype=float)
    if times.ndim != 1 or skews.shape != times.shape or len(times) < 2:
        raise ValueError("maturities and atm_skews must be matching vectors with at least two points")
    if np.any(times <= 0) or np.any(skews == 0):
        raise ValueError("maturities must be positive and skews cannot be zero")
    sign = float(np.sign(np.median(skews)))
    if sign == 0.0:
        sign = 1.0
    x = np.log(times)
    y = np.log(np.abs(skews))
    slope, intercept = np.polyfit(x, y, deg=1)
    fitted = slope * x + intercept
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return ATMSkewPowerLawFit(
        hurst=float(np.clip(slope + 0.5, 0.0, 1.0)),
        level=float(np.exp(intercept)),
        skew_sign=sign,
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r_squared),
    )
