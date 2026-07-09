from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from scipy.integrate import quad

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class HestonParams:
    """Risk-neutral Heston variance process parameters."""

    kappa: float
    theta: float
    sigma: float
    rho: float
    v0: float

    def validate(self) -> None:
        if self.kappa <= 0:
            raise ValueError("kappa must be positive")
        if self.theta <= 0:
            raise ValueError("theta must be positive")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if not -1 < self.rho < 1:
            raise ValueError("rho must be in (-1, 1)")
        if self.v0 <= 0:
            raise ValueError("v0 must be positive")


def heston_characteristic_function(
    u: complex,
    params: HestonParams,
    spot: float,
    maturity: float,
    rate: float,
    dividend: float = 0.0,
) -> complex:
    """Characteristic function of log S_T under the Heston model."""
    params.validate()
    if spot <= 0:
        raise ValueError("spot must be positive")
    if maturity < 0:
        raise ValueError("maturity must be non-negative")

    kappa, theta, sigma, rho, v0 = params.kappa, params.theta, params.sigma, params.rho, params.v0
    x0 = np.log(spot)
    iu = 1j * u
    d = np.sqrt((rho * sigma * iu - kappa) ** 2 + sigma**2 * (u**2 + iu))
    g = (kappa - rho * sigma * iu - d) / (kappa - rho * sigma * iu + d)
    exp_neg_dt = np.exp(-d * maturity)

    c = (rate - dividend) * iu * maturity
    c += (kappa * theta / sigma**2) * (
        (kappa - rho * sigma * iu - d) * maturity
        - 2.0 * np.log((1.0 - g * exp_neg_dt) / (1.0 - g))
    )
    d_term = ((kappa - rho * sigma * iu - d) / sigma**2) * ((1.0 - exp_neg_dt) / (1.0 - g * exp_neg_dt))
    return complex(np.exp(c + d_term * v0 + iu * x0))


def carr_madan_call_price(
    characteristic_function: Callable[[complex], complex],
    strike: float,
    maturity: float,
    rate: float,
    alpha: float = 1.5,
    integration_limit: float = 150.0,
    quad_limit: int = 250,
) -> float:
    """Price a call from a log-price characteristic function using Carr-Madan damping."""
    if strike <= 0:
        raise ValueError("strike must be positive")
    if maturity <= 0:
        raise ValueError("maturity must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    log_strike = np.log(strike)

    def integrand(u: float) -> float:
        z = u - 1j * (alpha + 1.0)
        numerator = np.exp(-1j * u * log_strike) * np.exp(-rate * maturity) * characteristic_function(z)
        denominator = alpha**2 + alpha - u**2 + 1j * (2.0 * alpha + 1.0) * u
        return float(np.real(numerator / denominator))

    integral, _ = quad(integrand, 0.0, integration_limit, limit=quad_limit, epsabs=1e-8, epsrel=1e-8)
    return float(np.exp(-alpha * log_strike) * integral / np.pi)


def heston_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    params: HestonParams,
    dividend: float = 0.0,
    option_type: OptionType = "call",
) -> float:
    """Price a European option under Heston using Fourier inversion."""
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    cf = lambda u: heston_characteristic_function(u, params, spot, maturity, rate, dividend)
    call = carr_madan_call_price(cf, strike, maturity, rate)
    if option_type == "call":
        return call
    discounted_spot = spot * np.exp(-dividend * maturity)
    discounted_strike = strike * np.exp(-rate * maturity)
    return float(call - discounted_spot + discounted_strike)
