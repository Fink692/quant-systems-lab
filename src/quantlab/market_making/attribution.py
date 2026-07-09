from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MarketMakingPnLAttribution:
    spread_capture: float
    latency_slippage: float
    inventory_mark_to_market: float
    final_pnl: float
    fill_count: int

    @property
    def explained_pnl(self) -> float:
        return self.spread_capture + self.latency_slippage + self.inventory_mark_to_market


def attribute_market_making_pnl(history: pd.DataFrame, order_size: float = 1.0) -> MarketMakingPnLAttribution:
    """Decompose path market-making PnL into spread, latency, and inventory components."""
    required = {"quote_mid", "bid", "ask", "bid_filled", "ask_filled", "slippage", "pnl"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"history is missing columns: {sorted(missing)}")
    if order_size <= 0:
        raise ValueError("order_size must be positive")
    bid_fills = history["bid_filled"].astype(bool)
    ask_fills = history["ask_filled"].astype(bool)
    bid_spread = (history.loc[bid_fills, "quote_mid"] - history.loc[bid_fills, "bid"]).sum() * order_size
    ask_spread = (history.loc[ask_fills, "ask"] - history.loc[ask_fills, "quote_mid"]).sum() * order_size
    spread_capture = float(bid_spread + ask_spread)
    latency_slippage = float(history["slippage"].sum())
    final_pnl = float(history["pnl"].iloc[-1])
    inventory_mark_to_market = final_pnl - spread_capture - latency_slippage
    return MarketMakingPnLAttribution(
        spread_capture=spread_capture,
        latency_slippage=latency_slippage,
        inventory_mark_to_market=float(inventory_mark_to_market),
        final_pnl=final_pnl,
        fill_count=int(bid_fills.sum() + ask_fills.sum()),
    )
