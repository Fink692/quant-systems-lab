from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClearingResult:
    payments: np.ndarray
    equity: np.ndarray
    defaulted: np.ndarray
    iterations: int
    converged: bool


def eisenberg_noe_clearing(
    liabilities: np.ndarray,
    external_assets: np.ndarray,
    tolerance: float = 1e-10,
    max_iterations: int = 1_000,
) -> ClearingResult:
    """Compute clearing payments for a financial network.

    liabilities[i, j] is the nominal amount institution i owes institution j.
    """
    liabilities = np.asarray(liabilities, dtype=float)
    external_assets = np.asarray(external_assets, dtype=float)
    if liabilities.ndim != 2 or liabilities.shape[0] != liabilities.shape[1]:
        raise ValueError("liabilities must be a square matrix")
    if external_assets.shape != (liabilities.shape[0],):
        raise ValueError("external_assets must have one value per institution")
    if np.any(liabilities < 0) or np.any(external_assets < 0):
        raise ValueError("liabilities and external_assets must be non-negative")
    if tolerance <= 0 or max_iterations < 1:
        raise ValueError("tolerance must be positive and max_iterations must be positive")

    total_liabilities = liabilities.sum(axis=1)
    relative = np.zeros_like(liabilities)
    solvent = total_liabilities > 0
    relative[solvent] = liabilities[solvent] / total_liabilities[solvent, None]

    payments = total_liabilities.copy()
    converged = False
    for iteration in range(1, max_iterations + 1):
        incoming = relative.T @ payments
        updated = np.minimum(total_liabilities, external_assets + incoming)
        if np.max(np.abs(updated - payments)) < tolerance:
            payments = updated
            converged = True
            break
        payments = updated

    incoming = relative.T @ payments
    equity = external_assets + incoming - payments
    defaulted = payments + tolerance < total_liabilities
    return ClearingResult(
        payments=payments, equity=equity, defaulted=defaulted, iterations=iteration, converged=converged
    )
