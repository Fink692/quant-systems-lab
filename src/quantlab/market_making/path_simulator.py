from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.market_making.avellaneda_stoikov import AvellanedaStoikovParams, optimal_quotes
from quantlab.market_making.execution import ExecutionModelParams, fill_probability


@dataclass(frozen=True)
class PathMarketMakingResult:
    history: pd.DataFrame

    @property
    def final_pnl(self) -> float:
        return float(self.history["pnl"].iloc[-1])

    @property
    def max_inventory_abs(self) -> float:
        return float(self.history["inventory"].abs().max())

    @property
    def fill_rate(self) -> float:
        fill_columns = self.history[["bid_filled", "ask_filled"]].to_numpy(dtype=float)
        return float(fill_columns.mean())

    @property
    def total_slippage(self) -> float:
        return float(self.history["slippage"].sum())


def simulate_latency_market_maker_on_path(
    mid_prices: np.ndarray,
    quote_params: AvellanedaStoikovParams,
    execution_params: ExecutionModelParams,
    dt: float = 1.0 / 390.0,
    latency_steps: int = 1,
    order_size: float = 1.0,
    seed: int | None = None,
) -> PathMarketMakingResult:
    """Replay a market maker on an observed mid-price path with delayed fills."""
    quote_params.validate()
    execution_params.validate()
    mids = np.asarray(mid_prices, dtype=float)
    if mids.ndim != 1 or len(mids) < 2 or np.any(mids <= 0):
        raise ValueError("mid_prices must be a positive one-dimensional path with at least two values")
    if dt <= 0 or latency_steps < 0 or order_size <= 0:
        raise ValueError("dt/order_size must be positive and latency_steps non-negative")

    rng = np.random.default_rng(seed)
    inventory = 0.0
    cash = 0.0
    rows: list[dict[str, float | bool]] = []
    horizon = dt + execution_params.latency

    for step in range(len(mids) - 1):
        quote_mid = float(mids[step])
        mark_mid = float(mids[step + 1])
        arrival_mid = float(mids[min(step + latency_steps, len(mids) - 1)])
        time = min(step * dt, quote_params.horizon)
        bid, ask = optimal_quotes(quote_mid, inventory, time, quote_params)
        bid_depth = max(quote_mid - bid, 0.0)
        ask_depth = max(ask - quote_mid, 0.0)
        bid_filled = rng.random() < fill_probability(bid_depth, horizon, execution_params)
        ask_filled = rng.random() < fill_probability(ask_depth, horizon, execution_params)

        slippage = 0.0
        if bid_filled:
            inventory += order_size
            cash -= order_size * bid
            intended_edge = quote_mid - bid
            realized_edge = arrival_mid - bid
            slippage += order_size * (realized_edge - intended_edge)
        if ask_filled:
            inventory -= order_size
            cash += order_size * ask
            intended_edge = ask - quote_mid
            realized_edge = ask - arrival_mid
            slippage += order_size * (realized_edge - intended_edge)

        pnl = cash + inventory * mark_mid
        rows.append(
            {
                "step": float(step),
                "time": float(time),
                "quote_mid": quote_mid,
                "arrival_mid": arrival_mid,
                "mark_mid": mark_mid,
                "bid": float(bid),
                "ask": float(ask),
                "bid_filled": bool(bid_filled),
                "ask_filled": bool(ask_filled),
                "inventory": float(inventory),
                "cash": float(cash),
                "pnl": float(pnl),
                "slippage": float(slippage),
            }
        )

    return PathMarketMakingResult(pd.DataFrame(rows))
