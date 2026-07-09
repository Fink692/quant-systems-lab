from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AvellanedaStoikovParams:
    risk_aversion: float
    volatility: float
    order_book_liquidity: float
    horizon: float

    def validate(self) -> None:
        if self.risk_aversion <= 0:
            raise ValueError("risk_aversion must be positive")
        if self.volatility < 0:
            raise ValueError("volatility must be non-negative")
        if self.order_book_liquidity <= 0:
            raise ValueError("order_book_liquidity must be positive")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")


def reservation_price(mid_price: float, inventory: float, time: float, params: AvellanedaStoikovParams) -> float:
    params.validate()
    remaining = max(params.horizon - time, 0.0)
    return float(mid_price - inventory * params.risk_aversion * params.volatility**2 * remaining)


def optimal_spread(time: float, params: AvellanedaStoikovParams) -> float:
    params.validate()
    remaining = max(params.horizon - time, 0.0)
    inventory_term = params.risk_aversion * params.volatility**2 * remaining
    liquidity_term = (2.0 / params.risk_aversion) * np.log(1.0 + params.risk_aversion / params.order_book_liquidity)
    return float(inventory_term + liquidity_term)


def optimal_quotes(
    mid_price: float,
    inventory: float,
    time: float,
    params: AvellanedaStoikovParams,
) -> tuple[float, float]:
    center = reservation_price(mid_price, inventory, time, params)
    half_spread = 0.5 * optimal_spread(time, params)
    return float(center - half_spread), float(center + half_spread)
