from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LiquiditySpiralResult:
    prices: np.ndarray
    equity: np.ndarray
    leverage: np.ndarray
    defaulted: np.ndarray
    rounds: int
    total_liquidation: np.ndarray
    price_history: list[np.ndarray]


def simulate_liquidity_spiral(
    holdings: np.ndarray,
    capital: np.ndarray,
    initial_returns: np.ndarray,
    market_depth: np.ndarray,
    leverage_limit: float = 8.0,
    liquidation_speed: float = 0.5,
    max_rounds: int = 20,
) -> LiquiditySpiralResult:
    """Simulate deleveraging pressure from mark-to-market losses and price impact."""
    holdings = np.asarray(holdings, dtype=float)
    capital = np.asarray(capital, dtype=float)
    prices = 1.0 + np.asarray(initial_returns, dtype=float)
    depth = np.asarray(market_depth, dtype=float)
    if holdings.ndim != 2:
        raise ValueError("holdings must be institution-by-asset")
    if capital.shape != (holdings.shape[0],) or prices.shape != (holdings.shape[1],) or depth.shape != prices.shape:
        raise ValueError("capital, returns, and market_depth shapes are inconsistent with holdings")
    if np.any(holdings < 0) or np.any(capital <= 0) or np.any(prices <= 0) or np.any(depth <= 0):
        raise ValueError("holdings must be non-negative and capital/prices/depth positive")
    if leverage_limit <= 1 or not 0 < liquidation_speed <= 1 or max_rounds < 1:
        raise ValueError("invalid liquidity spiral parameters")

    current_holdings = holdings.copy()
    initial_asset_value = holdings @ np.ones(holdings.shape[1])
    total_liquidation = np.zeros(holdings.shape[1])
    price_history = [prices.copy()]
    equity = capital + current_holdings @ prices - initial_asset_value
    defaulted = equity <= 0.0

    for round_idx in range(1, max_rounds + 1):
        asset_value = current_holdings @ prices
        equity = capital + asset_value - initial_asset_value
        defaulted = equity <= 0.0
        positive_equity = np.maximum(equity, 1e-12)
        leverage = asset_value / positive_equity
        excess_leverage = np.maximum(leverage / leverage_limit - 1.0, 0.0)
        sale_fraction = np.clip(liquidation_speed * excess_leverage, 0.0, 1.0)
        sale_fraction[defaulted] = liquidation_speed
        liquidation = (current_holdings.T * sale_fraction).T
        if liquidation.sum() <= 1e-12:
            return LiquiditySpiralResult(
                prices=prices,
                equity=equity,
                leverage=leverage,
                defaulted=defaulted,
                rounds=round_idx - 1,
                total_liquidation=total_liquidation,
                price_history=price_history,
            )
        current_holdings = np.maximum(current_holdings - liquidation, 0.0)
        aggregate_liquidation = liquidation.sum(axis=0)
        total_liquidation += aggregate_liquidation
        prices = np.maximum(prices * (1.0 - aggregate_liquidation / depth), 1e-8)
        price_history.append(prices.copy())

    asset_value = current_holdings @ prices
    equity = capital + asset_value - initial_asset_value
    leverage = asset_value / np.maximum(equity, 1e-12)
    return LiquiditySpiralResult(
        prices=prices,
        equity=equity,
        leverage=leverage,
        defaulted=equity <= 0.0,
        rounds=max_rounds,
        total_liquidation=total_liquidation,
        price_history=price_history,
    )
