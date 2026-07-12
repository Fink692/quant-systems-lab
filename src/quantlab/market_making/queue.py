from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QueueSimulationResult:
    history: pd.DataFrame
    filled: bool
    fill_time: float | None


def simulate_queue_position(
    initial_ahead: float,
    order_size: float,
    market_order_intensity: float,
    cancellation_intensity: float,
    horizon: float,
    dt: float = 0.01,
    seed: int | None = None,
) -> QueueSimulationResult:
    """Simulate queue depletion ahead of a passive limit order."""
    if min(initial_ahead, order_size, market_order_intensity, cancellation_intensity, horizon, dt) < 0:
        raise ValueError("queue parameters must be non-negative")
    if order_size <= 0 or horizon <= 0 or dt <= 0:
        raise ValueError("order_size, horizon, and dt must be positive")
    rng = np.random.default_rng(seed)
    ahead = float(initial_ahead)
    remaining = float(order_size)
    rows: list[dict[str, float | bool]] = []
    fill_time: float | None = None
    steps = int(np.ceil(horizon / dt))
    for step in range(steps + 1):
        time = min(step * dt, horizon)
        if step > 0 and remaining > 0:
            market_volume = rng.poisson(market_order_intensity * dt)
            cancellations = rng.poisson(cancellation_intensity * dt)
            ahead = max(ahead - cancellations, 0.0)
            if market_volume <= ahead:
                ahead -= market_volume
            else:
                excess = market_volume - ahead
                ahead = 0.0
                remaining = max(remaining - excess, 0.0)
                if remaining == 0.0 and fill_time is None:
                    fill_time = time
        rows.append(
            {"time": float(time), "ahead": float(ahead), "remaining": float(remaining), "filled": remaining == 0.0}
        )
        if fill_time is not None:
            break
    return QueueSimulationResult(pd.DataFrame(rows), filled=fill_time is not None, fill_time=fill_time)
