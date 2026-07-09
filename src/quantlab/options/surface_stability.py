from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantlab.options.black_scholes import black_scholes_price
from quantlab.options.local_volatility import dupire_local_volatility
from quantlab.options.surface import VolatilitySurface
from quantlab.options.surface_arbitrage import ArbitrageViolation, detect_surface_arbitrage


@dataclass(frozen=True)
class SurfaceStabilityReport:
    dense_maturities: np.ndarray
    dense_strikes: np.ndarray
    implied_volatilities: np.ndarray
    call_prices: np.ndarray
    local_volatilities: np.ndarray
    arbitrage_violations: list[ArbitrageViolation]
    invalid_implied_vol_count: int
    local_vol_nan_fraction: float
    max_implied_vol_step: float
    strike_curvature_rms: float
    maturity_curvature_rms: float

    @property
    def passes(self) -> bool:
        return self.invalid_implied_vol_count == 0 and len(self.arbitrage_violations) == 0 and self.local_vol_nan_fraction < 0.25


def diagnose_surface_interpolation_stability(
    surface: VolatilitySurface,
    maturity_points: int = 9,
    strike_points: int = 25,
) -> SurfaceStabilityReport:
    """Sample an interpolated volatility surface and diagnose numerical instability."""
    if maturity_points < 3 or strike_points < 3:
        raise ValueError("maturity_points and strike_points must be at least three")
    maturities = np.linspace(float(surface.maturities.min()), float(surface.maturities.max()), maturity_points)
    strikes = np.linspace(float(surface.strikes.min()), float(surface.strikes.max()), strike_points)
    vols = np.array([[surface.implied_volatility(maturity, strike) for strike in strikes] for maturity in maturities])
    invalid_mask = (~np.isfinite(vols)) | (vols <= 0.0)
    call_prices = np.full_like(vols, np.nan, dtype=float)
    local_vols = np.full_like(vols, np.nan, dtype=float)
    violations: list[ArbitrageViolation] = []

    if not invalid_mask.any():
        call_prices = np.array(
            [
                [
                    black_scholes_price(surface.spot, strike, maturity, surface.rate, vol, surface.dividend, "call")
                    for strike, vol in zip(strikes, row)
                ]
                for maturity, row in zip(maturities, vols)
            ],
            dtype=float,
        )
        violations = detect_surface_arbitrage(
            maturities,
            strikes,
            call_prices,
            spot=surface.spot,
            rate=surface.rate,
            dividend=surface.dividend,
        )
        local_vols = dupire_local_volatility(maturities, strikes, call_prices, surface.rate, surface.dividend)

    finite_local = np.isfinite(local_vols)
    local_nan_fraction = 1.0 - float(finite_local.mean())
    strike_steps = np.abs(np.diff(vols, axis=1))
    maturity_steps = np.abs(np.diff(vols, axis=0))
    strike_curvature = np.diff(vols, n=2, axis=1)
    maturity_curvature = np.diff(vols, n=2, axis=0)
    return SurfaceStabilityReport(
        dense_maturities=maturities,
        dense_strikes=strikes,
        implied_volatilities=vols,
        call_prices=call_prices,
        local_volatilities=local_vols,
        arbitrage_violations=violations,
        invalid_implied_vol_count=int(invalid_mask.sum()),
        local_vol_nan_fraction=local_nan_fraction,
        max_implied_vol_step=float(max(np.nanmax(strike_steps), np.nanmax(maturity_steps))),
        strike_curvature_rms=float(np.sqrt(np.nanmean(strike_curvature**2))),
        maturity_curvature_rms=float(np.sqrt(np.nanmean(maturity_curvature**2))),
    )
