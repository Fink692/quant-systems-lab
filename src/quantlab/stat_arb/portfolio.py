from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.stat_arb.network import mean_reversion_signal
from quantlab.stat_arb.selection import PairCandidate, candidate_spread_weights


@dataclass(frozen=True)
class StatArbPortfolioResult:
    history: pd.DataFrame
    pair_pnl: pd.DataFrame
    allocations: pd.Series
    asset_weights: pd.DataFrame

    @property
    def total_pnl(self) -> float:
        return float(self.history["cumulative_pnl"].iloc[-1])

    @property
    def turnover(self) -> float:
        return float(self.history["turnover"].sum())

    @property
    def max_drawdown(self) -> float:
        cumulative = self.history["cumulative_pnl"]
        return float((cumulative.cummax() - cumulative).max())


def allocate_pair_capital(
    candidates: list[PairCandidate], total_gross: float = 1.0, score_power: float = 1.0
) -> pd.Series:
    """Allocate gross exposure across pair candidates by rank score."""
    if total_gross <= 0 or score_power < 0:
        raise ValueError("total_gross must be positive and score_power non-negative")
    if not candidates:
        raise ValueError("candidates cannot be empty")
    scores = np.array([max(candidate.score, 0.0) for candidate in candidates], dtype=float)
    if score_power == 0 or scores.sum() <= 0:
        weights = np.full(len(candidates), 1.0 / len(candidates))
    else:
        powered = scores**score_power
        weights = powered / powered.sum()
    labels = [_candidate_label(candidate) for candidate in candidates]
    return pd.Series(total_gross * weights, index=labels, name="gross_allocation")


def backtest_pair_portfolio(
    prices: pd.DataFrame,
    candidates: list[PairCandidate],
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    window: int = 60,
    total_gross: float = 1.0,
    transaction_cost: float = 0.0,
) -> StatArbPortfolioResult:
    """Backtest an aggregate portfolio of ranked cointegrated pair trades."""
    if not candidates:
        raise ValueError("candidates cannot be empty")
    if transaction_cost < 0:
        raise ValueError("transaction_cost must be non-negative")
    clean = prices.dropna()
    allocations = allocate_pair_capital(candidates, total_gross=total_gross)
    pair_pnl = pd.DataFrame(index=clean.index)
    asset_weight_rows = []
    total_turnover = np.zeros(len(clean), dtype=float)

    for candidate in candidates:
        label = _candidate_label(candidate)
        spread = clean[candidate.dependent].to_numpy(dtype=float) - candidate.hedge_ratio * clean[
            candidate.independent
        ].to_numpy(dtype=float)
        signal = mean_reversion_signal(spread, entry_z=entry_z, exit_z=exit_z, window=window)
        allocation = float(allocations.loc[label])
        pnl = np.zeros(len(clean), dtype=float)
        turnover = np.zeros(len(clean), dtype=float)
        spread_weights = candidate_spread_weights(candidate, gross_exposure=allocation)
        for idx in range(1, len(clean)):
            pnl[idx] = allocation * signal[idx - 1] * (spread[idx] - spread[idx - 1])
            turnover[idx] = allocation * abs(signal[idx] - signal[idx - 1])
            pnl[idx] -= transaction_cost * turnover[idx]
        pair_pnl[label] = pnl
        total_turnover += turnover
        for idx, date in enumerate(clean.index):
            row = pd.Series(0.0, index=clean.columns, name=date)
            if signal[idx] != 0.0:
                row.loc[spread_weights.index] = signal[idx] * spread_weights
            asset_weight_rows.append(row.rename((label, date)))

    total_pnl = pair_pnl.sum(axis=1)
    history = pd.DataFrame(
        {
            "pnl": total_pnl,
            "turnover": total_turnover,
            "cumulative_pnl": total_pnl.cumsum(),
        },
        index=clean.index,
    )
    asset_weights = pd.DataFrame(asset_weight_rows)
    return StatArbPortfolioResult(
        history=history, pair_pnl=pair_pnl, allocations=allocations, asset_weights=asset_weights
    )


def _candidate_label(candidate: PairCandidate) -> str:
    return f"{candidate.dependent}/{candidate.independent}"
