from __future__ import annotations

from typing import Literal

import numpy as np

from quantlab.rough_vol.rough_bergomi import RoughBergomiParams, simulate_rough_bergomi

OptionType = Literal["call", "put"]


def rough_bergomi_option_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    params: RoughBergomiParams,
    option_type: OptionType = "call",
    paths: int = 20_000,
    steps: int = 128,
    seed: int | None = None,
) -> float:
    """Monte Carlo European option price under the rough Bergomi simulator."""
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    spot_paths, _ = simulate_rough_bergomi(spot, maturity, steps, paths, params, seed=seed)
    terminal = spot_paths[:, -1]
    payoff = np.maximum(terminal - strike, 0.0) if option_type == "call" else np.maximum(strike - terminal, 0.0)
    return float(np.exp(-rate * maturity) * np.mean(payoff))
