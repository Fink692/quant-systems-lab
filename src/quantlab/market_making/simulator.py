from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.market_making.avellaneda_stoikov import AvellanedaStoikovParams, optimal_quotes


@dataclass(frozen=True)
class MarketMakingSimulationResult:
    history: pd.DataFrame

    @property
    def final_pnl(self) -> float:
        return float(self.history["pnl"].iloc[-1])

    @property
    def max_inventory_abs(self) -> float:
        return float(self.history["inventory"].abs().max())


def simulate_market_maker(
    initial_mid_price: float,
    params: AvellanedaStoikovParams,
    steps: int = 500,
    dt: float = 1.0 / 500.0,
    price_volatility: float | None = None,
    fill_intensity: float = 2.0,
    seed: int | None = None,
) -> MarketMakingSimulationResult:
    """Simulate an Avellaneda-Stoikov market maker with Poisson quote fills."""
    params.validate()
    if initial_mid_price <= 0 or steps < 1 or dt <= 0 or fill_intensity < 0:
        raise ValueError("invalid simulation inputs")

    rng = np.random.default_rng(seed)
    mid = float(initial_mid_price)
    inventory = 0.0
    cash = 0.0
    price_volatility = params.volatility if price_volatility is None else price_volatility
    rows: list[dict[str, float | bool]] = []

    for step in range(steps):
        time = min(step * dt, params.horizon)
        bid, ask = optimal_quotes(mid, inventory, time, params)
        bid_depth = max(mid - bid, 0.0)
        ask_depth = max(ask - mid, 0.0)
        bid_fill_prob = 1.0 - np.exp(-fill_intensity * np.exp(-params.order_book_liquidity * bid_depth) * dt)
        ask_fill_prob = 1.0 - np.exp(-fill_intensity * np.exp(-params.order_book_liquidity * ask_depth) * dt)
        bid_filled = rng.random() < bid_fill_prob
        ask_filled = rng.random() < ask_fill_prob

        if bid_filled:
            inventory += 1.0
            cash -= bid
        if ask_filled:
            inventory -= 1.0
            cash += ask

        mid *= np.exp(-0.5 * price_volatility**2 * dt + price_volatility * np.sqrt(dt) * rng.normal())
        pnl = cash + inventory * mid
        rows.append(
            {
                "step": float(step),
                "time": float(time),
                "mid": float(mid),
                "bid": float(bid),
                "ask": float(ask),
                "inventory": float(inventory),
                "cash": float(cash),
                "pnl": float(pnl),
                "bid_filled": bid_filled,
                "ask_filled": ask_filled,
            }
        )

    return MarketMakingSimulationResult(pd.DataFrame(rows))
