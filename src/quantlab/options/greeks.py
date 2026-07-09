from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import norm

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class BlackScholesGreeks:
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def black_scholes_greeks(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend: float = 0.0,
    option_type: OptionType = "call",
) -> BlackScholesGreeks:
    """Analytic Black-Scholes-Merton Greeks for a European option."""
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if spot <= 0 or strike <= 0 or maturity <= 0 or volatility <= 0:
        raise ValueError("spot, strike, maturity, and volatility must be positive")

    sqrt_t = np.sqrt(maturity)
    d1 = (np.log(spot / strike) + (rate - dividend + 0.5 * volatility**2) * maturity) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    discounted_spot = spot * np.exp(-dividend * maturity)
    discounted_strike = strike * np.exp(-rate * maturity)

    if option_type == "call":
        price = discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2)
        delta = np.exp(-dividend * maturity) * norm.cdf(d1)
        theta = (
            -(discounted_spot * norm.pdf(d1) * volatility) / (2.0 * sqrt_t)
            - rate * discounted_strike * norm.cdf(d2)
            + dividend * discounted_spot * norm.cdf(d1)
        )
        rho = maturity * discounted_strike * norm.cdf(d2)
    else:
        price = discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1)
        delta = np.exp(-dividend * maturity) * (norm.cdf(d1) - 1.0)
        theta = (
            -(discounted_spot * norm.pdf(d1) * volatility) / (2.0 * sqrt_t)
            + rate * discounted_strike * norm.cdf(-d2)
            - dividend * discounted_spot * norm.cdf(-d1)
        )
        rho = -maturity * discounted_strike * norm.cdf(-d2)

    gamma = np.exp(-dividend * maturity) * norm.pdf(d1) / (spot * volatility * sqrt_t)
    vega = discounted_spot * norm.pdf(d1) * sqrt_t
    return BlackScholesGreeks(float(price), float(delta), float(gamma), float(vega), float(theta), float(rho))
