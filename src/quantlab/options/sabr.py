from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from quantlab.options.black_scholes import black_scholes_price

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class SABRParams:
    alpha: float
    beta: float
    rho: float
    nu: float

    def validate(self) -> None:
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if not 0 <= self.beta <= 1:
            raise ValueError("beta must be in [0, 1]")
        if not -1 < self.rho < 1:
            raise ValueError("rho must be in (-1, 1)")
        if self.nu < 0:
            raise ValueError("nu must be non-negative")


def sabr_implied_volatility(
    forward: float,
    strike: float,
    maturity: float,
    params: SABRParams,
) -> float:
    """Hagan et al. lognormal SABR implied volatility approximation."""
    params.validate()
    if forward <= 0 or strike <= 0:
        raise ValueError("forward and strike must be positive")
    if maturity < 0:
        raise ValueError("maturity must be non-negative")

    alpha, beta, rho, nu = params.alpha, params.beta, params.rho, params.nu
    one_minus_beta = 1.0 - beta

    if abs(forward - strike) < 1e-12:
        f_beta = forward**one_minus_beta
        correction = (
            (one_minus_beta**2 / 24.0) * (alpha**2 / forward ** (2.0 * one_minus_beta))
            + (rho * beta * nu * alpha / 4.0) / f_beta
            + ((2.0 - 3.0 * rho**2) * nu**2 / 24.0)
        )
        return float((alpha / f_beta) * (1.0 + correction * maturity))

    log_fk = np.log(forward / strike)
    fk_beta = (forward * strike) ** (0.5 * one_minus_beta)
    z = (nu / alpha) * fk_beta * log_fk if nu > 0 else 0.0
    if abs(z) < 1e-12:
        z_over_xz = 1.0
    else:
        x_z = np.log((np.sqrt(1.0 - 2.0 * rho * z + z**2) + z - rho) / (1.0 - rho))
        z_over_xz = z / x_z

    denominator = fk_beta * (1.0 + (one_minus_beta**2 / 24.0) * log_fk**2 + (one_minus_beta**4 / 1920.0) * log_fk**4)
    correction = (
        (one_minus_beta**2 / 24.0) * (alpha**2 / (forward * strike) ** one_minus_beta)
        + (rho * beta * nu * alpha / 4.0) / fk_beta
        + ((2.0 - 3.0 * rho**2) * nu**2 / 24.0)
    )
    return float((alpha / denominator) * z_over_xz * (1.0 + correction * maturity))


def sabr_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    params: SABRParams,
    dividend: float = 0.0,
    option_type: OptionType = "call",
) -> float:
    """Price by feeding SABR's implied volatility into Black-Scholes-Merton."""
    forward = spot * np.exp((rate - dividend) * maturity)
    implied_vol = sabr_implied_volatility(forward, strike, maturity, params)
    return black_scholes_price(spot, strike, maturity, rate, implied_vol, dividend, option_type)
