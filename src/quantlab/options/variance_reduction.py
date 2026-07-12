from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class MonteCarloEstimate:
    price: float
    standard_error: float
    paths: int


def black_scholes_antithetic_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend: float = 0.0,
    option_type: OptionType = "call",
    paths: int = 100_000,
    seed: int | None = None,
) -> MonteCarloEstimate:
    """Black-Scholes Monte Carlo with antithetic variates."""
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if min(spot, strike, maturity, volatility, paths) <= 0:
        raise ValueError("spot, strike, maturity, volatility, and paths must be positive")
    rng = np.random.default_rng(seed)
    half_paths = int(np.ceil(paths / 2))
    z = rng.normal(size=half_paths)
    z_full = np.concatenate([z, -z])[:paths]
    terminal = spot * np.exp(
        (rate - dividend - 0.5 * volatility**2) * maturity + volatility * np.sqrt(maturity) * z_full
    )
    payoff = np.maximum(terminal - strike, 0.0) if option_type == "call" else np.maximum(strike - terminal, 0.0)
    discounted = np.exp(-rate * maturity) * payoff
    return MonteCarloEstimate(float(np.mean(discounted)), float(np.std(discounted, ddof=1) / np.sqrt(paths)), paths)


def black_scholes_control_variate_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend: float = 0.0,
    option_type: OptionType = "call",
    paths: int = 100_000,
    seed: int | None = None,
) -> MonteCarloEstimate:
    """Black-Scholes Monte Carlo using discounted stock as a control variate."""
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if min(spot, strike, maturity, volatility, paths) <= 0:
        raise ValueError("spot, strike, maturity, volatility, and paths must be positive")
    rng = np.random.default_rng(seed)
    z = rng.normal(size=paths)
    terminal = spot * np.exp((rate - dividend - 0.5 * volatility**2) * maturity + volatility * np.sqrt(maturity) * z)
    payoff = np.maximum(terminal - strike, 0.0) if option_type == "call" else np.maximum(strike - terminal, 0.0)
    discounted_payoff = np.exp(-rate * maturity) * payoff
    control = np.exp(-rate * maturity) * terminal
    expected_control = spot * np.exp(-dividend * maturity)
    control_variance = np.var(control, ddof=1)
    beta = 0.0 if control_variance == 0.0 else np.cov(discounted_payoff, control, ddof=1)[0, 1] / control_variance
    adjusted = discounted_payoff - beta * (control - expected_control)
    return MonteCarloEstimate(float(np.mean(adjusted)), float(np.std(adjusted, ddof=1) / np.sqrt(paths)), paths)
