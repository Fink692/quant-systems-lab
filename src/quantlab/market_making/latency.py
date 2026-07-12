from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LatencySlippageResult:
    intended_edge: np.ndarray
    realized_edge: np.ndarray
    slippage: np.ndarray
    total_slippage: float
    adverse_fill_rate: float


def latency_slippage_report(
    quote_mid_prices: np.ndarray,
    arrival_mid_prices: np.ndarray,
    quote_prices: np.ndarray,
    sides: list[str] | np.ndarray,
    quantities: np.ndarray | float = 1.0,
) -> LatencySlippageResult:
    """Measure edge lost between quote placement and latency-delayed execution."""
    quote_mid = np.asarray(quote_mid_prices, dtype=float)
    arrival_mid = np.asarray(arrival_mid_prices, dtype=float)
    quotes = np.asarray(quote_prices, dtype=float)
    side_array = np.asarray(sides)
    qty = np.full_like(quotes, float(quantities)) if np.isscalar(quantities) else np.asarray(quantities, dtype=float)
    if (
        quote_mid.shape != arrival_mid.shape
        or quote_mid.shape != quotes.shape
        or quote_mid.shape != side_array.shape
        or quote_mid.shape != qty.shape
    ):
        raise ValueError("all inputs must have matching one-dimensional shapes")
    if (
        quote_mid.ndim != 1
        or np.any(quote_mid <= 0)
        or np.any(arrival_mid <= 0)
        or np.any(quotes <= 0)
        or np.any(qty < 0)
    ):
        raise ValueError("price inputs must be positive and quantities non-negative")
    if not set(np.unique(side_array)).issubset({"bid", "ask"}):
        raise ValueError("sides must contain only 'bid' or 'ask'")

    bid_mask = side_array == "bid"
    intended = np.where(bid_mask, quote_mid - quotes, quotes - quote_mid)
    realized = np.where(bid_mask, arrival_mid - quotes, quotes - arrival_mid)
    slippage = (realized - intended) * qty
    adverse = realized < intended
    return LatencySlippageResult(
        intended_edge=intended,
        realized_edge=realized,
        slippage=slippage,
        total_slippage=float(slippage.sum()),
        adverse_fill_rate=float(adverse.mean()) if len(adverse) else 0.0,
    )
