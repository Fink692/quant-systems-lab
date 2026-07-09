from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DebtRankResult:
    distress: np.ndarray
    incremental_distress: np.ndarray
    rounds: int
    total_impact: float


def debt_rank(
    exposures: np.ndarray,
    capital: np.ndarray,
    initial_distress: np.ndarray,
    damping: float = 1.0,
    max_rounds: int = 50,
    tolerance: float = 1e-10,
) -> DebtRankResult:
    """Propagate financial distress through an exposure-to-capital network."""
    exposures = np.asarray(exposures, dtype=float)
    capital = np.asarray(capital, dtype=float)
    distress = np.asarray(initial_distress, dtype=float)
    if exposures.ndim != 2 or exposures.shape[0] != exposures.shape[1]:
        raise ValueError("exposures must be square")
    if capital.shape != (exposures.shape[0],) or distress.shape != capital.shape:
        raise ValueError("capital and initial_distress shapes must match exposures")
    if np.any(capital <= 0) or np.any(exposures < 0):
        raise ValueError("capital must be positive and exposures non-negative")
    if not 0 <= damping <= 1 or max_rounds < 1 or tolerance <= 0:
        raise ValueError("invalid propagation parameters")

    distress = np.clip(distress, 0.0, 1.0)
    impact_matrix = np.minimum(exposures / capital[:, None], 1.0)
    incremental = distress.copy()
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        propagated = damping * (impact_matrix @ incremental)
        updated = np.maximum(distress, np.minimum(1.0, distress + propagated))
        incremental = updated - distress
        distress = updated
        if np.max(incremental) < tolerance:
            break
    total_impact = float(np.mean(distress) - np.mean(np.clip(initial_distress, 0.0, 1.0)))
    return DebtRankResult(distress=distress, incremental_distress=incremental, rounds=rounds, total_impact=total_impact)
