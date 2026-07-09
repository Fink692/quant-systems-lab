from __future__ import annotations

from typing import Literal

import numpy as np

from quantlab.options.heston import HestonParams

OptionType = Literal["call", "put"]


def heston_monte_carlo_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    params: HestonParams,
    dividend: float = 0.0,
    option_type: OptionType = "call",
    paths: int = 20_000,
    steps: int = 252,
    seed: int | None = None,
) -> float:
    """Price a European option under Heston with full-truncation Euler Monte Carlo."""
    params.validate()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if spot <= 0 or strike <= 0 or maturity <= 0:
        raise ValueError("spot, strike, and maturity must be positive")
    if paths < 1 or steps < 1:
        raise ValueError("paths and steps must be positive")

    rng = np.random.default_rng(seed)
    dt = maturity / steps
    log_spot = np.full(paths, np.log(spot), dtype=float)
    variance = np.full(paths, params.v0, dtype=float)

    for _ in range(steps):
        z_v = rng.normal(size=paths)
        z_independent = rng.normal(size=paths)
        z_s = params.rho * z_v + np.sqrt(1.0 - params.rho**2) * z_independent
        variance_positive = np.maximum(variance, 0.0)
        log_spot += (rate - dividend - 0.5 * variance_positive) * dt + np.sqrt(variance_positive * dt) * z_s
        variance = variance + params.kappa * (params.theta - variance_positive) * dt
        variance += params.sigma * np.sqrt(variance_positive * dt) * z_v

    terminal = np.exp(log_spot)
    if option_type == "call":
        payoff = np.maximum(terminal - strike, 0.0)
    else:
        payoff = np.maximum(strike - terminal, 0.0)
    return float(np.exp(-rate * maturity) * np.mean(payoff))


def black_scholes_monte_carlo_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend: float = 0.0,
    option_type: OptionType = "call",
    paths: int = 100_000,
    seed: int | None = None,
) -> float:
    """Black-Scholes Monte Carlo baseline for variance-reduction experiments."""
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    rng = np.random.default_rng(seed)
    z = rng.normal(size=paths)
    terminal = spot * np.exp((rate - dividend - 0.5 * volatility**2) * maturity + volatility * np.sqrt(maturity) * z)
    payoff = np.maximum(terminal - strike, 0.0) if option_type == "call" else np.maximum(strike - terminal, 0.0)
    return float(np.exp(-rate * maturity) * np.mean(payoff))
