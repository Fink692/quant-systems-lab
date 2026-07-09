from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

OptionType = Literal["call", "put"]


def _validate_option_type(option_type: str) -> None:
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")


def black_scholes_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend: float = 0.0,
    option_type: OptionType = "call",
) -> float:
    """Price a European option with the Black-Scholes-Merton formula."""
    _validate_option_type(option_type)
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if maturity < 0:
        raise ValueError("maturity must be non-negative")
    if volatility < 0:
        raise ValueError("volatility must be non-negative")

    if maturity == 0 or volatility == 0:
        forward_intrinsic = spot * np.exp(-dividend * maturity) - strike * np.exp(-rate * maturity)
        return float(max(forward_intrinsic, 0.0) if option_type == "call" else max(-forward_intrinsic, 0.0))

    sqrt_t = np.sqrt(maturity)
    d1 = (np.log(spot / strike) + (rate - dividend + 0.5 * volatility**2) * maturity) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    discounted_spot = spot * np.exp(-dividend * maturity)
    discounted_strike = strike * np.exp(-rate * maturity)
    call = discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2)

    if option_type == "call":
        return float(call)
    return float(call - discounted_spot + discounted_strike)


def implied_volatility(
    option_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float = 0.0,
    option_type: OptionType = "call",
    lower: float = 1e-8,
    upper: float = 5.0,
) -> float:
    """Invert Black-Scholes-Merton with Brent's method."""
    _validate_option_type(option_type)
    if option_price < 0:
        raise ValueError("option_price must be non-negative")
    if maturity <= 0:
        raise ValueError("maturity must be positive for implied volatility")

    def objective(volatility: float) -> float:
        return black_scholes_price(spot, strike, maturity, rate, volatility, dividend, option_type) - option_price

    low_value = objective(lower)
    high_value = objective(upper)
    if low_value * high_value > 0:
        raise ValueError("option_price is outside the no-arbitrage range covered by the volatility bracket")

    return float(brentq(objective, lower, upper, xtol=1e-10, rtol=1e-10, maxiter=200))
