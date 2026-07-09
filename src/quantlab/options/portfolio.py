from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from quantlab.options.black_scholes import black_scholes_price
from quantlab.options.greeks import black_scholes_greeks

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class OptionPosition:
    quantity: float
    strike: float
    maturity: float
    volatility: float
    option_type: OptionType = "call"
    dividend: float = 0.0


@dataclass(frozen=True)
class OptionBookStressResult:
    base_value: float
    scenario_values: pd.DataFrame
    greek_exposures: pd.Series


def option_book_value(spot: float, rate: float, positions: list[OptionPosition]) -> float:
    if spot <= 0:
        raise ValueError("spot must be positive")
    return float(
        sum(
            position.quantity
            * black_scholes_price(
                spot,
                position.strike,
                position.maturity,
                rate,
                position.volatility,
                position.dividend,
                position.option_type,
            )
            for position in positions
        )
    )


def option_book_greeks(spot: float, rate: float, positions: list[OptionPosition]) -> pd.Series:
    if spot <= 0:
        raise ValueError("spot must be positive")
    totals = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    for position in positions:
        greeks = black_scholes_greeks(
            spot,
            position.strike,
            position.maturity,
            rate,
            position.volatility,
            position.dividend,
            position.option_type,
        )
        totals["delta"] += position.quantity * greeks.delta
        totals["gamma"] += position.quantity * greeks.gamma
        totals["vega"] += position.quantity * greeks.vega
        totals["theta"] += position.quantity * greeks.theta
        totals["rho"] += position.quantity * greeks.rho
    return pd.Series(totals, name="greek_exposure")


def stress_option_book(
    spot: float,
    rate: float,
    positions: list[OptionPosition],
    spot_shocks: np.ndarray,
    volatility_shocks: np.ndarray,
) -> OptionBookStressResult:
    """Stress an option book across spot and volatility shocks."""
    if not positions:
        raise ValueError("positions cannot be empty")
    spot_shocks = np.asarray(spot_shocks, dtype=float)
    vol_shocks = np.asarray(volatility_shocks, dtype=float)
    rows = []
    base_value = option_book_value(spot, rate, positions)
    for spot_shock in spot_shocks:
        shocked_spot = spot * (1.0 + spot_shock)
        if shocked_spot <= 0:
            continue
        for vol_shock in vol_shocks:
            shocked_positions = [
                OptionPosition(
                    quantity=p.quantity,
                    strike=p.strike,
                    maturity=p.maturity,
                    volatility=max(p.volatility + vol_shock, 1e-8),
                    option_type=p.option_type,
                    dividend=p.dividend,
                )
                for p in positions
            ]
            value = option_book_value(shocked_spot, rate, shocked_positions)
            rows.append(
                {
                    "spot_shock": float(spot_shock),
                    "volatility_shock": float(vol_shock),
                    "value": value,
                    "pnl": value - base_value,
                }
            )
    return OptionBookStressResult(
        base_value=base_value,
        scenario_values=pd.DataFrame(rows),
        greek_exposures=option_book_greeks(spot, rate, positions),
    )
