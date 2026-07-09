from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import least_squares

from quantlab.options.bates import BatesParams, bates_price
from quantlab.options.heston import HestonParams, heston_price
from quantlab.options.sabr import SABRParams, sabr_implied_volatility

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class OptionQuote:
    strike: float
    maturity: float
    price: float
    option_type: OptionType = "call"
    weight: float = 1.0


@dataclass(frozen=True)
class CalibrationResult:
    parameters: dict[str, float]
    objective_value: float
    residuals: np.ndarray
    success: bool
    message: str


def calibrate_sabr_smile(
    forward: float,
    maturity: float,
    strikes: np.ndarray,
    implied_volatilities: np.ndarray,
    beta: float = 0.5,
    initial: tuple[float, float, float] = (0.2, -0.2, 0.5),
) -> CalibrationResult:
    """Calibrate alpha, rho, and nu to one SABR volatility smile with fixed beta."""
    strikes = np.asarray(strikes, dtype=float)
    market_vols = np.asarray(implied_volatilities, dtype=float)
    if strikes.shape != market_vols.shape or strikes.ndim != 1:
        raise ValueError("strikes and implied_volatilities must be one-dimensional arrays with the same shape")

    def residuals(raw: np.ndarray) -> np.ndarray:
        params = SABRParams(alpha=raw[0], beta=beta, rho=raw[1], nu=raw[2])
        model = np.array([sabr_implied_volatility(forward, strike, maturity, params) for strike in strikes])
        return model - market_vols

    result = least_squares(
        residuals,
        np.asarray(initial, dtype=float),
        bounds=([1e-6, -0.999, 0.0], [5.0, 0.999, 5.0]),
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    return CalibrationResult(
        parameters={"alpha": float(result.x[0]), "beta": float(beta), "rho": float(result.x[1]), "nu": float(result.x[2])},
        objective_value=float(np.sum(result.fun**2)),
        residuals=result.fun,
        success=bool(result.success),
        message=str(result.message),
    )


def calibrate_heston(
    quotes: list[OptionQuote],
    spot: float,
    rate: float,
    dividend: float = 0.0,
    initial: HestonParams = HestonParams(kappa=1.5, theta=0.04, sigma=0.4, rho=-0.4, v0=0.04),
    max_nfev: int = 80,
) -> CalibrationResult:
    """Least-squares Heston calibration against option prices."""
    if not quotes:
        raise ValueError("quotes cannot be empty")

    def residuals(raw: np.ndarray) -> np.ndarray:
        params = HestonParams(kappa=raw[0], theta=raw[1], sigma=raw[2], rho=raw[3], v0=raw[4])
        values = []
        for quote in quotes:
            model = heston_price(spot, quote.strike, quote.maturity, rate, params, dividend, quote.option_type)
            values.append(np.sqrt(quote.weight) * (model - quote.price))
        return np.asarray(values)

    result = least_squares(
        residuals,
        np.array([initial.kappa, initial.theta, initial.sigma, initial.rho, initial.v0], dtype=float),
        bounds=([1e-4, 1e-5, 1e-4, -0.999, 1e-5], [10.0, 2.0, 5.0, 0.999, 2.0]),
        max_nfev=max_nfev,
    )
    return CalibrationResult(
        parameters={
            "kappa": float(result.x[0]),
            "theta": float(result.x[1]),
            "sigma": float(result.x[2]),
            "rho": float(result.x[3]),
            "v0": float(result.x[4]),
        },
        objective_value=float(np.sum(result.fun**2)),
        residuals=result.fun,
        success=bool(result.success),
        message=str(result.message),
    )


def calibrate_bates(
    quotes: list[OptionQuote],
    spot: float,
    rate: float,
    heston_seed: HestonParams,
    dividend: float = 0.0,
    initial_jumps: tuple[float, float, float] = (0.05, -0.03, 0.15),
    max_nfev: int = 80,
) -> CalibrationResult:
    """Calibrate Bates jump parameters while holding Heston parameters fixed."""
    if not quotes:
        raise ValueError("quotes cannot be empty")

    def residuals(raw: np.ndarray) -> np.ndarray:
        params = BatesParams(heston=heston_seed, jump_intensity=raw[0], jump_mean=raw[1], jump_volatility=raw[2])
        values = []
        for quote in quotes:
            model = bates_price(spot, quote.strike, quote.maturity, rate, params, dividend, quote.option_type)
            values.append(np.sqrt(quote.weight) * (model - quote.price))
        return np.asarray(values)

    result = least_squares(
        residuals,
        np.asarray(initial_jumps, dtype=float),
        bounds=([0.0, -1.0, 0.0], [5.0, 1.0, 2.0]),
        max_nfev=max_nfev,
    )
    return CalibrationResult(
        parameters={
            "jump_intensity": float(result.x[0]),
            "jump_mean": float(result.x[1]),
            "jump_volatility": float(result.x[2]),
        },
        objective_value=float(np.sum(result.fun**2)),
        residuals=result.fun,
        success=bool(result.success),
        message=str(result.message),
    )
