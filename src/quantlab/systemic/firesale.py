from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FireSaleResult:
    prices: np.ndarray
    equity: np.ndarray
    defaulted: np.ndarray
    rounds: int
    history: list[np.ndarray]


def simulate_fire_sale(
    holdings: np.ndarray,
    capital: np.ndarray,
    initial_shock: np.ndarray,
    impact: np.ndarray,
    liquidation_fraction: float = 0.5,
    max_rounds: int = 20,
) -> FireSaleResult:
    """Simulate mark-to-market losses and liquidation-driven price impact."""
    holdings = np.asarray(holdings, dtype=float)
    capital = np.asarray(capital, dtype=float)
    prices = 1.0 + np.asarray(initial_shock, dtype=float)
    impact = np.asarray(impact, dtype=float)
    if holdings.ndim != 2:
        raise ValueError("holdings must be institution-by-asset")
    if capital.shape != (holdings.shape[0],) or prices.shape != (holdings.shape[1],) or impact.shape != prices.shape:
        raise ValueError("capital, initial_shock, and impact shapes are inconsistent with holdings")
    if np.any(holdings < 0) or np.any(capital <= 0) or np.any(impact < 0):
        raise ValueError("holdings and impact must be non-negative; capital positive")
    if np.any(prices < 0) or not 0 <= liquidation_fraction <= 1 or max_rounds < 1:
        raise ValueError("invalid fire-sale parameters")

    initial_asset_value = holdings @ np.ones(holdings.shape[1])
    defaulted = np.zeros(holdings.shape[0], dtype=bool)
    history = [prices.copy()]
    for round_idx in range(1, max_rounds + 1):
        equity = capital + holdings @ prices - initial_asset_value
        newly_defaulted = (equity < 0.0) & ~defaulted
        if not np.any(newly_defaulted):
            return FireSaleResult(prices=prices, equity=equity, defaulted=defaulted, rounds=round_idx - 1, history=history)
        defaulted |= newly_defaulted
        liquidation = liquidation_fraction * holdings[newly_defaulted].sum(axis=0)
        prices = np.maximum(prices - impact * liquidation, 0.0)
        history.append(prices.copy())

    equity = capital + holdings @ prices - initial_asset_value
    return FireSaleResult(prices=prices, equity=equity, defaulted=defaulted, rounds=max_rounds, history=history)
