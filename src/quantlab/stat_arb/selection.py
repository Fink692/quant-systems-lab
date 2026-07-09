from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.stat_arb.cointegration import estimate_ou
from quantlab.stat_arb.network import CointegrationNetwork


@dataclass(frozen=True)
class PairCandidate:
    dependent: str
    independent: str
    p_value: float
    hedge_ratio: float
    spread_volatility: float
    half_life: float
    score: float


def rank_cointegrated_pairs(
    prices: pd.DataFrame,
    network: CointegrationNetwork,
    max_p_value: float = 0.05,
    max_half_life: float | None = None,
    top_n: int | None = None,
) -> list[PairCandidate]:
    """Rank directed cointegrated pairs by statistical strength and tradability."""
    if not 0 < max_p_value < 1:
        raise ValueError("max_p_value must be in (0, 1)")
    clean = prices.dropna()
    candidates: list[PairCandidate] = []
    names = list(clean.columns)
    for dependent in names:
        for independent in names:
            if dependent == independent:
                continue
            p_value = float(network.p_values.loc[dependent, independent])
            if not np.isfinite(p_value) or p_value > max_p_value:
                continue
            hedge_ratio = float(network.hedge_ratios.loc[dependent, independent])
            spread = clean[dependent].to_numpy(dtype=float) - hedge_ratio * clean[independent].to_numpy(dtype=float)
            ou = estimate_ou(spread)
            half_life = float(ou["half_life"])
            if max_half_life is not None and half_life > max_half_life:
                continue
            spread_vol = float(np.std(np.diff(spread), ddof=1))
            score = float(-np.log10(max(p_value, 1e-300)) / (1.0 + max(half_life, 0.0)))
            candidates.append(
                PairCandidate(
                    dependent=dependent,
                    independent=independent,
                    p_value=p_value,
                    hedge_ratio=hedge_ratio,
                    spread_volatility=spread_vol,
                    half_life=half_life,
                    score=score,
                )
            )
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates if top_n is None else candidates[:top_n]


def candidate_spread_weights(candidate: PairCandidate, gross_exposure: float = 1.0) -> pd.Series:
    """Convert a pair candidate into dollar-neutral spread weights."""
    if gross_exposure <= 0:
        raise ValueError("gross_exposure must be positive")
    raw = pd.Series({candidate.dependent: 1.0, candidate.independent: -candidate.hedge_ratio}, dtype=float)
    return gross_exposure * raw / raw.abs().sum()
