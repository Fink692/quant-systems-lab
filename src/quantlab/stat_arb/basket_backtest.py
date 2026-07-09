from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.stat_arb.johansen import basket_spread, johansen_hedge_vector
from quantlab.stat_arb.network import mean_reversion_signal


@dataclass(frozen=True)
class JohansenBasketBacktestResult:
    history: pd.DataFrame
    hedge_weights: pd.Series
    asset_positions: pd.DataFrame

    @property
    def total_pnl(self) -> float:
        return float(self.history["cumulative_pnl"].iloc[-1])

    @property
    def turnover(self) -> float:
        return float(self.history["turnover"].sum())

    @property
    def max_drawdown(self) -> float:
        cumulative = self.history["cumulative_pnl"]
        running_max = cumulative.cummax()
        return float((running_max - cumulative).max())


def backtest_johansen_basket_strategy(
    prices: pd.DataFrame,
    vector_index: int = 0,
    normalize_asset: str | None = None,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    window: int = 60,
    gross_exposure: float = 1.0,
    transaction_cost: float = 0.0,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> JohansenBasketBacktestResult:
    """Backtest a multi-asset Johansen cointegration basket mean-reversion strategy."""
    if gross_exposure <= 0:
        raise ValueError("gross_exposure must be positive")
    if transaction_cost < 0:
        raise ValueError("transaction_cost must be non-negative")
    clean = prices.dropna()
    if clean.shape[0] < max(window + 2, 10) or clean.shape[1] < 2:
        raise ValueError("prices must have enough rows and at least two assets")

    hedge = johansen_hedge_vector(clean, vector_index=vector_index, normalize_asset=normalize_asset, det_order=det_order, k_ar_diff=k_ar_diff)
    if hedge.abs().sum() <= 0:
        raise ValueError("hedge vector has zero gross exposure")
    weights = gross_exposure * hedge / hedge.abs().sum()
    spread = basket_spread(clean, weights)
    signal = mean_reversion_signal(spread.to_numpy(dtype=float), entry_z=entry_z, exit_z=exit_z, window=window)

    price_changes = clean[weights.index].diff().fillna(0.0)
    asset_positions = pd.DataFrame(
        signal[:, None] * weights.to_numpy()[None, :],
        index=clean.index,
        columns=weights.index,
    )
    pnl = np.zeros(len(clean), dtype=float)
    turnover = np.zeros(len(clean), dtype=float)
    position_values = asset_positions.to_numpy(dtype=float)
    for idx in range(1, len(clean)):
        pnl[idx] = float(position_values[idx - 1] @ price_changes.iloc[idx].to_numpy(dtype=float))
        turnover[idx] = float(np.abs(position_values[idx] - position_values[idx - 1]).sum())
        pnl[idx] -= transaction_cost * turnover[idx]

    history = pd.DataFrame(
        {
            "spread": spread.to_numpy(dtype=float),
            "signal": signal,
            "turnover": turnover,
            "pnl": pnl,
            "cumulative_pnl": np.cumsum(pnl),
        },
        index=clean.index,
    )
    return JohansenBasketBacktestResult(history=history, hedge_weights=weights, asset_positions=asset_positions)
