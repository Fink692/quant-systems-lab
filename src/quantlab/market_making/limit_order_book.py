from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class Fill:
    price: float
    quantity: float


class LimitOrderBook:
    """Price-level limit order book for deterministic simulations."""

    def __init__(self) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}

    def add_limit_order(self, side: Side, price: float, quantity: float) -> None:
        if price <= 0 or quantity <= 0:
            raise ValueError("price and quantity must be positive")
        book = self.bids if side == "buy" else self.asks if side == "sell" else None
        if book is None:
            raise ValueError("side must be 'buy' or 'sell'")
        book[price] = book.get(price, 0.0) + quantity

    def depth_at(self, side: Side, price: float) -> float:
        if price <= 0:
            raise ValueError("price must be positive")
        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")
        book = self.bids if side == "buy" else self.asks
        return float(book.get(price, 0.0))

    def cancel_quantity(self, side: Side, price: float, quantity: float) -> float:
        if price <= 0 or quantity < 0:
            raise ValueError("price must be positive and quantity non-negative")
        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")
        book = self.bids if side == "buy" else self.asks
        available = book.get(price, 0.0)
        canceled = min(available, quantity)
        remaining = available - canceled
        if remaining <= 1e-12:
            book.pop(price, None)
        else:
            book[price] = remaining
        return float(canceled)

    def market_order(self, side: Side, quantity: float) -> list[Fill]:
        """Execute a market order; buy consumes asks, sell consumes bids."""
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")

        book = self.asks if side == "buy" else self.bids
        price_order = sorted(book) if side == "buy" else sorted(book, reverse=True)
        remaining = quantity
        fills: list[Fill] = []
        for price in price_order:
            if remaining <= 0:
                break
            available = book[price]
            traded = min(available, remaining)
            fills.append(Fill(price=price, quantity=traded))
            remaining -= traded
            available -= traded
            if available <= 1e-12:
                del book[price]
            else:
                book[price] = available
        return fills

    @property
    def best_bid(self) -> float | None:
        return max(self.bids) if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return min(self.asks) if self.asks else None

    @property
    def mid_price(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return 0.5 * (self.best_bid + self.best_ask)

    @property
    def spread(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    def snapshot(self, depth: int = 5) -> dict[str, list[tuple[float, float]]]:
        return {
            "bids": [(p, self.bids[p]) for p in sorted(self.bids, reverse=True)[:depth]],
            "asks": [(p, self.asks[p]) for p in sorted(self.asks)[:depth]],
        }
