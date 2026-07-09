from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from quantlab.options.black_scholes import black_scholes_price
from quantlab.options.greeks import black_scholes_greeks

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class DeltaHedgeResult:
    history: pd.DataFrame

    @property
    def final_pnl(self) -> float:
        return float(self.history["portfolio_value"].iloc[-1])

    @property
    def total_transaction_cost(self) -> float:
        return float(self.history["transaction_cost"].sum())


def simulate_delta_hedge(
    spot_path: np.ndarray,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    dividend: float = 0.0,
    option_type: OptionType = "call",
    transaction_cost_bps: float = 0.0,
) -> DeltaHedgeResult:
    """Simulate a self-financing delta hedge for one short European option."""
    spots = np.asarray(spot_path, dtype=float)
    if spots.ndim != 1 or len(spots) < 2 or np.any(spots <= 0):
        raise ValueError("spot_path must be a positive one-dimensional array with at least two values")
    if strike <= 0 or maturity <= 0 or volatility <= 0 or transaction_cost_bps < 0:
        raise ValueError("strike, maturity, and volatility must be positive; transaction_cost_bps non-negative")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")

    dt = maturity / (len(spots) - 1)
    cost_rate = transaction_cost_bps / 10_000.0
    initial_greeks = black_scholes_greeks(spots[0], strike, maturity, rate, volatility, dividend, option_type)
    shares = initial_greeks.delta
    cash = initial_greeks.price - shares * spots[0]
    rows = [
        {
            "step": 0,
            "spot": float(spots[0]),
            "remaining_maturity": float(maturity),
            "delta": float(shares),
            "cash": float(cash),
            "transaction_cost": 0.0,
            "option_value": float(initial_greeks.price),
            "portfolio_value": 0.0,
        }
    ]

    for idx in range(1, len(spots)):
        cash *= np.exp(rate * dt)
        remaining = max(maturity - idx * dt, 0.0)
        spot = float(spots[idx])
        if remaining > 0.0:
            greeks = black_scholes_greeks(spot, strike, remaining, rate, volatility, dividend, option_type)
            target_delta = greeks.delta
            option_value = greeks.price
        else:
            target_delta = 0.0
            option_value = max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)

        trade = target_delta - shares
        transaction_cost = abs(trade) * spot * cost_rate
        cash -= trade * spot + transaction_cost
        shares = target_delta

        if remaining == 0.0:
            portfolio_value = cash + shares * spot - option_value
        else:
            portfolio_value = cash + shares * spot - option_value
        rows.append(
            {
                "step": idx,
                "spot": spot,
                "remaining_maturity": float(remaining),
                "delta": float(shares),
                "cash": float(cash),
                "transaction_cost": float(transaction_cost),
                "option_value": float(option_value),
                "portfolio_value": float(portfolio_value),
            }
        )

    return DeltaHedgeResult(pd.DataFrame(rows))
