from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.market_making.avellaneda_stoikov import AvellanedaStoikovParams, optimal_quotes
from quantlab.market_making.limit_order_book import Fill, LimitOrderBook


@dataclass(frozen=True)
class OrderBookMarketMakingResult:
    history: pd.DataFrame

    @property
    def final_pnl(self) -> float:
        return float(self.history["pnl"].iloc[-1])

    @property
    def max_inventory_abs(self) -> float:
        return float(self.history["inventory"].abs().max())

    @property
    def fill_rate(self) -> float:
        fills = self.history[["bid_fill_qty", "ask_fill_qty"]].to_numpy(dtype=float)
        return float((fills > 0.0).mean())

    @property
    def average_spread(self) -> float:
        return float((self.history["best_ask"] - self.history["best_bid"]).mean())


def simulate_order_book_market_maker(
    initial_mid_price: float,
    params: AvellanedaStoikovParams,
    steps: int = 250,
    dt: float = 1.0 / 250.0,
    tick_size: float = 0.01,
    levels: int = 5,
    depth_per_level: float = 10.0,
    order_size: float = 1.0,
    market_order_intensity: float = 300.0,
    cancellation_rate: float = 0.05,
    seed: int | None = None,
) -> OrderBookMarketMakingResult:
    """Simulate an Avellaneda-Stoikov agent joining queues in a price-level book."""
    params.validate()
    if initial_mid_price <= 0 or steps < 1 or dt <= 0:
        raise ValueError("initial_mid_price, steps, and dt must be positive")
    if tick_size <= 0 or levels < 1 or min(depth_per_level, order_size, market_order_intensity) <= 0:
        raise ValueError("book depth, tick, order size, and market order intensity must be positive")
    if not 0 <= cancellation_rate < 1:
        raise ValueError("cancellation_rate must be in [0, 1)")

    rng = np.random.default_rng(seed)
    book = LimitOrderBook()
    _replenish_book(book, initial_mid_price, tick_size, levels, depth_per_level)
    inventory = 0.0
    cash = 0.0
    rows: list[dict[str, float]] = []

    for step in range(steps):
        mid = book.mid_price if book.mid_price is not None else initial_mid_price
        time = min(step * dt, params.horizon)
        raw_bid, raw_ask = optimal_quotes(float(mid), inventory, time, params)
        bid = min(_floor_tick(raw_bid, tick_size), _floor_tick(mid - tick_size, tick_size))
        ask = max(_ceil_tick(raw_ask, tick_size), _ceil_tick(mid + tick_size, tick_size))
        bid = max(bid, tick_size)
        ask = max(ask, bid + tick_size)

        bid_queue_ahead = book.depth_at("buy", bid)
        ask_queue_ahead = book.depth_at("sell", ask)
        book.add_limit_order("buy", bid, order_size)
        book.add_limit_order("sell", ask, order_size)

        buy_market_qty = float(rng.poisson(market_order_intensity * dt) * order_size)
        sell_market_qty = float(rng.poisson(market_order_intensity * dt) * order_size)
        buy_fills = book.market_order("buy", buy_market_qty) if buy_market_qty > 0 else []
        sell_fills = book.market_order("sell", sell_market_qty) if sell_market_qty > 0 else []

        executed_at_bid = _filled_quantity_at(sell_fills, bid)
        executed_at_ask = _filled_quantity_at(buy_fills, ask)
        bid_fill_qty = min(max(executed_at_bid - bid_queue_ahead, 0.0), order_size)
        ask_fill_qty = min(max(executed_at_ask - ask_queue_ahead, 0.0), order_size)
        if bid_fill_qty > 0.0:
            inventory += bid_fill_qty
            cash -= bid * bid_fill_qty
        if ask_fill_qty > 0.0:
            inventory -= ask_fill_qty
            cash += ask * ask_fill_qty

        book.cancel_quantity("buy", bid, max(order_size - bid_fill_qty, 0.0))
        book.cancel_quantity("sell", ask, max(order_size - ask_fill_qty, 0.0))
        _random_cancellations(book, cancellation_rate, rng)
        mark_mid = book.mid_price if book.mid_price is not None else mid
        _replenish_book(book, float(mark_mid), tick_size, levels, depth_per_level)
        mark_mid = float(book.mid_price if book.mid_price is not None else mark_mid)
        pnl = cash + inventory * mark_mid

        rows.append(
            {
                "step": float(step),
                "time": float(time),
                "mid": float(mid),
                "best_bid": float(book.best_bid if book.best_bid is not None else bid),
                "best_ask": float(book.best_ask if book.best_ask is not None else ask),
                "bid": float(bid),
                "ask": float(ask),
                "bid_queue_ahead": float(bid_queue_ahead),
                "ask_queue_ahead": float(ask_queue_ahead),
                "buy_market_qty": float(buy_market_qty),
                "sell_market_qty": float(sell_market_qty),
                "bid_fill_qty": float(bid_fill_qty),
                "ask_fill_qty": float(ask_fill_qty),
                "inventory": float(inventory),
                "cash": float(cash),
                "pnl": float(pnl),
            }
        )

    return OrderBookMarketMakingResult(pd.DataFrame(rows))


def _filled_quantity_at(fills: list[Fill], price: float) -> float:
    return float(sum(fill.quantity for fill in fills if abs(fill.price - price) < 1e-12))


def _floor_tick(value: float, tick_size: float) -> float:
    return float(np.floor(value / tick_size) * tick_size)


def _ceil_tick(value: float, tick_size: float) -> float:
    return float(np.ceil(value / tick_size) * tick_size)


def _replenish_book(book: LimitOrderBook, mid: float, tick_size: float, levels: int, depth_per_level: float) -> None:
    center = round(mid / tick_size) * tick_size
    for level in range(1, levels + 1):
        bid = max(center - level * tick_size, tick_size)
        ask = center + level * tick_size
        bid_shortfall = depth_per_level - book.depth_at("buy", bid)
        ask_shortfall = depth_per_level - book.depth_at("sell", ask)
        if bid_shortfall > 0.0:
            book.add_limit_order("buy", bid, bid_shortfall)
        if ask_shortfall > 0.0:
            book.add_limit_order("sell", ask, ask_shortfall)


def _random_cancellations(book: LimitOrderBook, cancellation_rate: float, rng: np.random.Generator) -> None:
    for side, levels in (("buy", list(book.bids.items())), ("sell", list(book.asks.items()))):
        for price, quantity in levels:
            cancel_qty = quantity * cancellation_rate * rng.random()
            if cancel_qty > 0.0:
                book.cancel_quantity(side, price, cancel_qty)
