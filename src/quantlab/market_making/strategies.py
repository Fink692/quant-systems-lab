from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt
from typing import Protocol

import numpy as np

from quantlab.market_making.avellaneda_stoikov import AvellanedaStoikovParams, optimal_quotes


@dataclass(frozen=True)
class MarketState:
    timestamp_ns: int
    best_bid: float
    best_ask: float
    bid_quantity: float
    ask_quantity: float
    inventory: float
    toxicity: float
    volatility: float

    @property
    def mid(self) -> float:
        return 0.5 * (self.best_bid + self.best_ask)


class QuotingPolicy(Protocol):
    @property
    def name(self) -> str: ...

    def quotes(self, state: MarketState, tick_size: float) -> tuple[float | None, float | None]: ...


def _maker_quotes(mid: float, bid_distance: float, ask_distance: float, tick: float) -> tuple[float, float]:
    bid = np.floor((mid - max(bid_distance, 0.5 * tick)) / tick + 1e-9) * tick
    ask = np.ceil((mid + max(ask_distance, 0.5 * tick)) / tick - 1e-9) * tick
    return float(bid), float(ask)


@dataclass(frozen=True)
class FixedSpreadPolicy:
    half_spread_ticks: float = 0.5
    name: str = "fixed_spread"

    def quotes(self, state: MarketState, tick_size: float) -> tuple[float, float]:
        distance = self.half_spread_ticks * tick_size
        return _maker_quotes(state.mid, distance, distance, tick_size)


@dataclass(frozen=True)
class AvellanedaStoikovPolicy:
    risk_aversion: float = 0.05
    liquidity: float = 100.0
    horizon: float = 1.0
    name: str = "avellaneda_stoikov"

    def quotes(self, state: MarketState, tick_size: float) -> tuple[float, float]:
        params = AvellanedaStoikovParams(
            risk_aversion=self.risk_aversion,
            volatility=max(state.volatility, tick_size),
            order_book_liquidity=self.liquidity,
            horizon=self.horizon,
        )
        bid, ask = optimal_quotes(state.mid, state.inventory, 0.0, params)
        return _maker_quotes(state.mid, state.mid - bid, ask - state.mid, tick_size)


@dataclass(frozen=True)
class QueueAwarePolicy:
    inventory_skew_ticks: float = 0.25
    name: str = "queue_aware"

    def quotes(self, state: MarketState, tick_size: float) -> tuple[float, float]:
        total = state.bid_quantity + state.ask_quantity
        imbalance = (state.bid_quantity - state.ask_quantity) / total if total > 0 else 0.0
        bid_ticks = 0.5 + max(imbalance, 0.0)
        ask_ticks = 0.5 + max(-imbalance, 0.0)
        bid_ticks += max(state.inventory, 0.0) * self.inventory_skew_ticks
        ask_ticks += max(-state.inventory, 0.0) * self.inventory_skew_ticks
        return _maker_quotes(state.mid, bid_ticks * tick_size, ask_ticks * tick_size, tick_size)


@dataclass(frozen=True)
class ToxicityAwarePolicy:
    toxicity_threshold: float = 0.15
    max_widen_ticks: float = 2.0
    name: str = "toxicity_aware"

    def quotes(self, state: MarketState, tick_size: float) -> tuple[float, float]:
        magnitude = max(abs(state.toxicity) - self.toxicity_threshold, 0.0)
        widen = min(magnitude * self.max_widen_ticks, self.max_widen_ticks)
        bid_ticks = 0.5 + (widen if state.toxicity < 0 else 0.0) + max(state.inventory, 0.0) * 0.2
        ask_ticks = 0.5 + (widen if state.toxicity > 0 else 0.0) + max(-state.inventory, 0.0) * 0.2
        return _maker_quotes(state.mid, bid_ticks * tick_size, ask_ticks * tick_size, tick_size)


@dataclass(frozen=True)
class LatencyAwarePolicy:
    latency_ns: int
    toxicity_threshold: float = 0.15
    name: str = "latency_aware"

    def quotes(self, state: MarketState, tick_size: float) -> tuple[float, float]:
        seconds = max(self.latency_ns, 0) / 1e9
        latency_buffer = ceil(state.volatility * sqrt(seconds) / tick_size) if seconds > 0 else 0
        toxicity_buffer = max(abs(state.toxicity) - self.toxicity_threshold, 0.0) * 2.0
        bid_ticks = 0.5 + latency_buffer + (toxicity_buffer if state.toxicity < 0 else 0.0)
        ask_ticks = 0.5 + latency_buffer + (toxicity_buffer if state.toxicity > 0 else 0.0)
        bid_ticks += max(state.inventory, 0.0) * 0.2
        ask_ticks += max(-state.inventory, 0.0) * 0.2
        return _maker_quotes(state.mid, bid_ticks * tick_size, ask_ticks * tick_size, tick_size)


def default_policy_set(latency_ns: int) -> tuple[QuotingPolicy, ...]:
    return (
        FixedSpreadPolicy(),
        AvellanedaStoikovPolicy(),
        QueueAwarePolicy(),
        ToxicityAwarePolicy(),
        LatencyAwarePolicy(latency_ns=latency_ns),
    )
