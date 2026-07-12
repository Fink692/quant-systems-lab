from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class SVIParams:
    """Raw SVI total variance parameters."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def validate(self) -> None:
        if self.b < 0:
            raise ValueError("b must be non-negative")
        if not -1 < self.rho < 1:
            raise ValueError("rho must be in (-1, 1)")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")


@dataclass(frozen=True)
class SVICalibrationResult:
    params: SVIParams
    objective_value: float
    residuals: np.ndarray
    success: bool
    message: str


def svi_total_variance(log_moneyness: np.ndarray, params: SVIParams) -> np.ndarray:
    """Raw SVI total implied variance w(k)."""
    params.validate()
    k = np.asarray(log_moneyness, dtype=float)
    centered = k - params.m
    return params.a + params.b * (params.rho * centered + np.sqrt(centered**2 + params.sigma**2))


def svi_implied_volatility(log_moneyness: np.ndarray, maturity: float, params: SVIParams) -> np.ndarray:
    if maturity <= 0:
        raise ValueError("maturity must be positive")
    total_variance = svi_total_variance(log_moneyness, params)
    if np.any(total_variance <= 0):
        raise ValueError("SVI total variance must be positive")
    return np.sqrt(total_variance / maturity)


def calibrate_svi_slice(
    log_moneyness: np.ndarray,
    implied_volatilities: np.ndarray,
    maturity: float,
    initial: SVIParams | None = None,
) -> SVICalibrationResult:
    """Calibrate a raw SVI slice to implied volatilities at one maturity."""
    k = np.asarray(log_moneyness, dtype=float)
    vols = np.asarray(implied_volatilities, dtype=float)
    if k.ndim != 1 or vols.shape != k.shape:
        raise ValueError("log_moneyness and implied_volatilities must be one-dimensional arrays with the same shape")
    if maturity <= 0 or np.any(vols <= 0):
        raise ValueError("maturity and implied volatilities must be positive")

    target_total_variance = vols**2 * maturity
    seed = initial or SVIParams(
        a=max(float(np.min(target_total_variance)) * 0.5, 1e-6),
        b=0.1,
        rho=0.0,
        m=float(k[np.argmin(target_total_variance)]),
        sigma=0.2,
    )

    def residuals(raw: np.ndarray) -> np.ndarray:
        params = SVIParams(a=raw[0], b=raw[1], rho=raw[2], m=raw[3], sigma=raw[4])
        return svi_total_variance(k, params) - target_total_variance

    result = least_squares(
        residuals,
        np.array([seed.a, seed.b, seed.rho, seed.m, seed.sigma], dtype=float),
        bounds=([-2.0, 0.0, -0.999, -5.0, 1e-6], [5.0, 5.0, 0.999, 5.0, 5.0]),
        xtol=1e-11,
        ftol=1e-11,
        gtol=1e-11,
        max_nfev=2_000,
    )
    params = SVIParams(
        a=float(result.x[0]),
        b=float(result.x[1]),
        rho=float(result.x[2]),
        m=float(result.x[3]),
        sigma=float(result.x[4]),
    )
    return SVICalibrationResult(
        params=params,
        objective_value=float(np.sum(result.fun**2)),
        residuals=result.fun,
        success=bool(result.success),
        message=str(result.message),
    )
