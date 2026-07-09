from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.stat_arb.cointegration import engle_granger, zscore


@dataclass(frozen=True)
class CointegrationNetwork:
    adjacency: pd.DataFrame
    p_values: pd.DataFrame
    hedge_ratios: pd.DataFrame


def pairwise_cointegration_network(prices: pd.DataFrame, p_value_threshold: float = 0.05) -> CointegrationNetwork:
    """Build a graph of pairwise Engle-Granger relationships."""
    if not 0 < p_value_threshold < 1:
        raise ValueError("p_value_threshold must be in (0, 1)")
    clean = prices.dropna()
    names = list(clean.columns)
    p_values = pd.DataFrame(np.nan, index=names, columns=names)
    hedge_ratios = pd.DataFrame(np.nan, index=names, columns=names)
    adjacency = pd.DataFrame(False, index=names, columns=names)
    for i, y_name in enumerate(names):
        for j, x_name in enumerate(names):
            if i == j:
                p_values.loc[y_name, x_name] = 0.0
                hedge_ratios.loc[y_name, x_name] = 1.0
                continue
            result = engle_granger(clean[y_name].to_numpy(), clean[x_name].to_numpy())
            p_values.loc[y_name, x_name] = result.p_value
            hedge_ratios.loc[y_name, x_name] = result.hedge_ratio
            adjacency.loc[y_name, x_name] = result.p_value <= p_value_threshold
    return CointegrationNetwork(adjacency=adjacency, p_values=p_values, hedge_ratios=hedge_ratios)


def mean_reversion_signal(spread: np.ndarray, entry_z: float = 2.0, exit_z: float = 0.5, window: int = 60) -> np.ndarray:
    """Generate spread position signals: -1 short spread, +1 long spread, 0 flat."""
    if entry_z <= exit_z or exit_z < 0:
        raise ValueError("entry_z must be greater than exit_z >= 0")
    scores = zscore(spread, window)
    positions = np.zeros(len(scores))
    current = 0.0
    for idx, score in enumerate(scores):
        if np.isnan(score):
            positions[idx] = current
            continue
        if current == 0.0:
            if score > entry_z:
                current = -1.0
            elif score < -entry_z:
                current = 1.0
        elif abs(score) < exit_z:
            current = 0.0
        positions[idx] = current
    return positions
