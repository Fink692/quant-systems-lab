from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AdverseSelectionReport:
    average_signed_move: float
    hit_rate: float
    observations: int
    by_sign: pd.DataFrame


def order_flow_imbalance(buy_volume: np.ndarray, sell_volume: np.ndarray) -> np.ndarray:
    """Compute order-flow imbalance from buy and sell volume arrays."""
    buy = np.asarray(buy_volume, dtype=float)
    sell = np.asarray(sell_volume, dtype=float)
    if buy.shape != sell.shape:
        raise ValueError("buy_volume and sell_volume must have the same shape")
    total = buy + sell
    imbalance = np.zeros_like(total, dtype=float)
    valid = total > 0
    imbalance[valid] = (buy[valid] - sell[valid]) / total[valid]
    return imbalance


def volume_synchronized_pin(
    signed_volume: np.ndarray,
    bucket_volume: float,
) -> pd.Series:
    """Estimate VPIN-like toxicity from signed volume buckets."""
    signed = np.asarray(signed_volume, dtype=float)
    if signed.ndim != 1:
        raise ValueError("signed_volume must be one-dimensional")
    if bucket_volume <= 0:
        raise ValueError("bucket_volume must be positive")

    buckets: list[float] = []
    current_abs_volume = 0.0
    current_signed_volume = 0.0
    for value in signed:
        remaining_value = value
        while abs(remaining_value) + current_abs_volume >= bucket_volume:
            needed = bucket_volume - current_abs_volume
            signed_piece = np.sign(remaining_value) * needed
            current_signed_volume += signed_piece
            buckets.append(abs(current_signed_volume) / bucket_volume)
            remaining_value -= signed_piece
            current_abs_volume = 0.0
            current_signed_volume = 0.0
        current_abs_volume += abs(remaining_value)
        current_signed_volume += remaining_value
    return pd.Series(buckets, name="vpin")


def adverse_selection_report(
    trade_signs: np.ndarray,
    mid_prices: np.ndarray,
    horizon: int = 1,
) -> AdverseSelectionReport:
    """Measure whether signed trades predict future mid-price moves."""
    signs = np.asarray(trade_signs, dtype=float)
    prices = np.asarray(mid_prices, dtype=float)
    if signs.ndim != 1 or prices.ndim != 1 or len(signs) != len(prices):
        raise ValueError("trade_signs and mid_prices must be same-length one-dimensional arrays")
    if horizon < 1 or horizon >= len(prices):
        raise ValueError("horizon must be in [1, len(prices) - 1]")
    valid_signs = signs[:-horizon]
    future_moves = prices[horizon:] - prices[:-horizon]
    signed_moves = valid_signs * future_moves
    observations = int(np.sum(valid_signs != 0))
    active = valid_signs != 0
    average = float(np.mean(signed_moves[active])) if observations else 0.0
    hit_rate = float(np.mean(signed_moves[active] > 0.0)) if observations else 0.0
    rows = []
    for sign in [-1.0, 1.0]:
        mask = valid_signs == sign
        rows.append(
            {
                "trade_sign": sign,
                "observations": int(mask.sum()),
                "average_future_move": float(np.mean(future_moves[mask])) if mask.any() else 0.0,
            }
        )
    return AdverseSelectionReport(
        average_signed_move=average,
        hit_rate=hit_rate,
        observations=observations,
        by_sign=pd.DataFrame(rows).set_index("trade_sign"),
    )
