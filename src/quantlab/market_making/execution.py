from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExecutionModelParams:
    fill_intensity: float
    order_book_liquidity: float
    latency: float = 0.0
    adverse_selection_bps: float = 0.0

    def validate(self) -> None:
        if self.fill_intensity < 0:
            raise ValueError("fill_intensity must be non-negative")
        if self.order_book_liquidity <= 0:
            raise ValueError("order_book_liquidity must be positive")
        if self.latency < 0:
            raise ValueError("latency must be non-negative")
        if self.adverse_selection_bps < 0:
            raise ValueError("adverse_selection_bps must be non-negative")


def fill_probability(quote_distance: float, horizon: float, params: ExecutionModelParams) -> float:
    """Poisson fill probability that decays with quote distance and latency."""
    params.validate()
    if quote_distance < 0 or horizon < 0:
        raise ValueError("quote_distance and horizon must be non-negative")
    effective_horizon = max(horizon - params.latency, 0.0)
    intensity = params.fill_intensity * np.exp(-params.order_book_liquidity * quote_distance)
    return float(1.0 - np.exp(-intensity * effective_horizon))


def expected_execution_value(
    mid_price: float,
    quote_price: float,
    side: str,
    horizon: float,
    params: ExecutionModelParams,
) -> float:
    """Expected spread capture net of a simple adverse-selection penalty."""
    if side not in {"bid", "ask"}:
        raise ValueError("side must be 'bid' or 'ask'")
    if mid_price <= 0 or quote_price <= 0:
        raise ValueError("mid_price and quote_price must be positive")
    distance = abs(mid_price - quote_price)
    probability = fill_probability(distance, horizon, params)
    gross_edge = mid_price - quote_price if side == "bid" else quote_price - mid_price
    adverse_selection = mid_price * params.adverse_selection_bps / 10_000.0
    return float(probability * (gross_edge - adverse_selection))
