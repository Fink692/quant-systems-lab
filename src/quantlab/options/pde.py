from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.linalg import solve_banded

OptionType = Literal["call", "put"]


def black_scholes_finite_difference_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend: float = 0.0,
    option_type: OptionType = "call",
    spot_steps: int = 250,
    time_steps: int = 250,
    spot_max_multiplier: float = 4.0,
) -> float:
    """Fully implicit finite-difference solver for European Black-Scholes options."""
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if spot <= 0 or strike <= 0 or maturity <= 0 or volatility <= 0:
        raise ValueError("spot, strike, maturity, and volatility must be positive")
    if spot_steps < 4 or time_steps < 1:
        raise ValueError("spot_steps must be >= 4 and time_steps must be positive")

    s_max = max(spot, strike) * spot_max_multiplier
    dt = maturity / time_steps
    grid = np.linspace(0.0, s_max, spot_steps + 1)

    if option_type == "call":
        values = np.maximum(grid - strike, 0.0)
    else:
        values = np.maximum(strike - grid, 0.0)

    i = np.arange(1, spot_steps)
    a = 0.5 * volatility**2 * i**2 - 0.5 * (rate - dividend) * i
    b = -(volatility**2 * i**2 + rate)
    c = 0.5 * volatility**2 * i**2 + 0.5 * (rate - dividend) * i

    lower = -dt * a
    diagonal = 1.0 - dt * b
    upper = -dt * c

    banded = np.zeros((3, spot_steps - 1))
    banded[0, 1:] = upper[:-1]
    banded[1, :] = diagonal
    banded[2, :-1] = lower[1:]

    def low_boundary(tau: float) -> float:
        return 0.0 if option_type == "call" else strike * np.exp(-rate * tau)

    def high_boundary(tau: float) -> float:
        if option_type == "call":
            return s_max * np.exp(-dividend * tau) - strike * np.exp(-rate * tau)
        return 0.0

    for step in range(1, time_steps + 1):
        tau = step * dt
        rhs = values[1:-1].copy()
        rhs[0] -= lower[0] * low_boundary(tau)
        rhs[-1] -= upper[-1] * high_boundary(tau)
        values[1:-1] = solve_banded((1, 1), banded, rhs)
        values[0] = low_boundary(tau)
        values[-1] = high_boundary(tau)

    return float(np.interp(spot, grid, values))
