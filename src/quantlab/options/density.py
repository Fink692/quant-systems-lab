from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskNeutralDensity:
    strikes: np.ndarray
    density: np.ndarray
    maturity: float

    @property
    def mass(self) -> float:
        return float(np.trapezoid(self.density, self.strikes))

    @property
    def mean_strike(self) -> float:
        mass = self.mass
        if mass <= 0:
            return float("nan")
        return float(np.trapezoid(self.strikes * self.density, self.strikes) / mass)


def breeden_litzenberger_density(
    strikes: np.ndarray,
    call_prices: np.ndarray,
    maturity: float,
    rate: float,
    clip_negative: bool = True,
) -> RiskNeutralDensity:
    """Estimate risk-neutral terminal density from call prices across strikes."""
    strikes = np.asarray(strikes, dtype=float)
    prices = np.asarray(call_prices, dtype=float)
    if strikes.ndim != 1 or prices.shape != strikes.shape:
        raise ValueError("strikes and call_prices must be one-dimensional arrays with the same shape")
    if len(strikes) < 3:
        raise ValueError("at least three strikes are required")
    if maturity <= 0 or np.any(np.diff(strikes) <= 0):
        raise ValueError("maturity must be positive and strikes strictly increasing")
    first_derivative = np.gradient(prices, strikes, edge_order=2)
    second_derivative = np.gradient(first_derivative, strikes, edge_order=2)
    density = np.exp(rate * maturity) * second_derivative
    if clip_negative:
        density = np.maximum(density, 0.0)
    mass = np.trapezoid(density, strikes)
    if mass > 0:
        density = density / mass
    return RiskNeutralDensity(strikes=strikes, density=density, maturity=float(maturity))


def density_from_price_surface(
    maturities: np.ndarray,
    strikes: np.ndarray,
    call_prices: np.ndarray,
    rate: float,
) -> pd.DataFrame:
    """Estimate one risk-neutral density slice per maturity from a call price grid."""
    maturities = np.asarray(maturities, dtype=float)
    strikes = np.asarray(strikes, dtype=float)
    prices = np.asarray(call_prices, dtype=float)
    if prices.shape != (len(maturities), len(strikes)):
        raise ValueError("call_prices shape must be (len(maturities), len(strikes))")
    rows = []
    for maturity, row in zip(maturities, prices):
        density = breeden_litzenberger_density(strikes, row, maturity, rate)
        for strike, value in zip(density.strikes, density.density):
            rows.append({"maturity": float(maturity), "strike": float(strike), "density": float(value)})
    return pd.DataFrame(rows)
