from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SSVIParams:
    rho: float
    eta: float
    gamma: float

    def validate(self) -> None:
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must be in (-1, 1)")
        if self.eta <= 0:
            raise ValueError("eta must be positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")


@dataclass(frozen=True)
class SSVIArbitrageCheck:
    calendar_monotone: bool
    butterfly_sufficient: bool
    max_calendar_decrease: float
    max_butterfly_metric: float
    max_slope_metric: float

    @property
    def passes(self) -> bool:
        return self.calendar_monotone and self.butterfly_sufficient


def ssvi_phi(theta: np.ndarray | float, params: SSVIParams) -> np.ndarray:
    params.validate()
    theta_array = np.asarray(theta, dtype=float)
    if np.any(theta_array <= 0):
        raise ValueError("theta must be positive")
    return params.eta * theta_array ** (-params.gamma)


def ssvi_total_variance(log_moneyness: np.ndarray | float, theta: float, params: SSVIParams) -> np.ndarray:
    """Evaluate Gatheral-Jacquier SSVI total implied variance."""
    params.validate()
    if theta <= 0:
        raise ValueError("theta must be positive")
    k = np.asarray(log_moneyness, dtype=float)
    phi = float(ssvi_phi(theta, params))
    inner = (phi * k + params.rho) ** 2 + 1.0 - params.rho**2
    return 0.5 * theta * (1.0 + params.rho * phi * k + np.sqrt(inner))


def ssvi_implied_volatility(log_moneyness: np.ndarray | float, maturity: float, theta: float, params: SSVIParams) -> np.ndarray:
    if maturity <= 0:
        raise ValueError("maturity must be positive")
    total_variance = ssvi_total_variance(log_moneyness, theta, params)
    return np.sqrt(np.maximum(total_variance / maturity, 0.0))


def ssvi_surface(
    maturities: np.ndarray,
    log_moneyness: np.ndarray,
    atm_total_variances: np.ndarray,
    params: SSVIParams,
) -> np.ndarray:
    """Build an SSVI implied-volatility grid indexed by maturity and log-moneyness."""
    maturities = np.asarray(maturities, dtype=float)
    theta = np.asarray(atm_total_variances, dtype=float)
    k = np.asarray(log_moneyness, dtype=float)
    if maturities.ndim != 1 or theta.shape != maturities.shape or k.ndim != 1:
        raise ValueError("maturities/theta must match and log_moneyness must be one-dimensional")
    if np.any(maturities <= 0) or np.any(theta <= 0):
        raise ValueError("maturities and atm_total_variances must be positive")
    return np.vstack([ssvi_implied_volatility(k, maturity, total_variance, params) for maturity, total_variance in zip(maturities, theta)])


def check_ssvi_no_arbitrage(atm_total_variances: np.ndarray, params: SSVIParams, tolerance: float = 1e-10) -> SSVIArbitrageCheck:
    """Check monotone ATM variance and common SSVI butterfly sufficient bounds."""
    params.validate()
    theta = np.asarray(atm_total_variances, dtype=float)
    if theta.ndim != 1 or len(theta) < 2 or np.any(theta <= 0):
        raise ValueError("atm_total_variances must be a positive vector with at least two entries")
    decreases = theta[:-1] - theta[1:]
    max_calendar_decrease = float(np.maximum(decreases, 0.0).max(initial=0.0))
    phi = ssvi_phi(theta, params)
    slope_metric = theta * phi * (1.0 + abs(params.rho))
    butterfly_metric = theta * phi**2 * (1.0 + abs(params.rho))
    return SSVIArbitrageCheck(
        calendar_monotone=bool(max_calendar_decrease <= tolerance),
        butterfly_sufficient=bool(np.max(slope_metric) < 4.0 + tolerance and np.max(butterfly_metric) <= 4.0 + tolerance),
        max_calendar_decrease=max_calendar_decrease,
        max_butterfly_metric=float(np.max(butterfly_metric)),
        max_slope_metric=float(np.max(slope_metric)),
    )
