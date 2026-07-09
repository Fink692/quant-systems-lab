from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContagionResult:
    defaulted: np.ndarray
    equity_losses: np.ndarray
    rounds: int
    history: list[np.ndarray]


def simulate_contagion(
    exposures: np.ndarray,
    capital: np.ndarray,
    initial_defaults: list[int] | np.ndarray,
    recovery_rate: float = 0.4,
) -> ContagionResult:
    """Propagate defaults through an exposure network.

    exposures[i, j] is the exposure of institution i to default by institution j.
    """
    exposures = np.asarray(exposures, dtype=float)
    capital = np.asarray(capital, dtype=float)
    if exposures.ndim != 2 or exposures.shape[0] != exposures.shape[1]:
        raise ValueError("exposures must be a square matrix")
    if capital.shape != (exposures.shape[0],):
        raise ValueError("capital must have one value per institution")
    if np.any(capital <= 0):
        raise ValueError("capital values must be positive")
    if not 0 <= recovery_rate <= 1:
        raise ValueError("recovery_rate must be in [0, 1]")

    n_nodes = exposures.shape[0]
    defaulted = np.zeros(n_nodes, dtype=bool)
    defaulted[np.asarray(initial_defaults, dtype=int)] = True
    losses = np.zeros(n_nodes)
    history = [defaulted.copy()]
    rounds = 0

    while True:
        newly_defaulted = np.zeros(n_nodes, dtype=bool)
        for node in np.where(defaulted)[0]:
            losses += (1.0 - recovery_rate) * exposures[:, node]
            exposures[:, node] = 0.0
        breached = losses > capital
        newly_defaulted = breached & ~defaulted
        if not np.any(newly_defaulted):
            break
        defaulted |= newly_defaulted
        history.append(defaulted.copy())
        rounds += 1

    return ContagionResult(defaulted=defaulted, equity_losses=losses, rounds=rounds, history=history)


def eigenvalue_stability(exposures: np.ndarray, capital: np.ndarray) -> float:
    """Return the spectral radius of exposure-to-capital leverage matrix."""
    exposures = np.asarray(exposures, dtype=float)
    capital = np.asarray(capital, dtype=float)
    if exposures.ndim != 2 or exposures.shape[0] != exposures.shape[1]:
        raise ValueError("exposures must be a square matrix")
    leverage = exposures / capital[:, None]
    return float(max(abs(np.linalg.eigvals(leverage))))
