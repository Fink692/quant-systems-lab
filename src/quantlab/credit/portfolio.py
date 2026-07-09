from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class CreditPortfolioLossResult:
    losses: np.ndarray
    expected_loss: float
    value_at_risk: float
    expected_shortfall: float
    default_counts: np.ndarray


def gaussian_copula_default_losses(
    default_probabilities: np.ndarray,
    exposures: np.ndarray,
    recovery_rates: np.ndarray | float = 0.4,
    asset_correlation: float = 0.2,
    simulations: int = 20_000,
    confidence: float = 0.99,
    seed: int | None = None,
) -> CreditPortfolioLossResult:
    """Simulate credit portfolio losses with a one-factor Gaussian copula."""
    pds = np.asarray(default_probabilities, dtype=float)
    notionals = np.asarray(exposures, dtype=float)
    recoveries = np.full_like(pds, float(recovery_rates)) if np.isscalar(recovery_rates) else np.asarray(recovery_rates, dtype=float)
    if pds.ndim != 1 or notionals.shape != pds.shape or recoveries.shape != pds.shape:
        raise ValueError("default_probabilities, exposures, and recovery_rates must have matching one-dimensional shapes")
    if np.any((pds <= 0) | (pds >= 1)) or np.any(notionals < 0) or np.any((recoveries < 0) | (recoveries > 1)):
        raise ValueError("invalid probabilities, exposures, or recoveries")
    if not 0 <= asset_correlation < 1 or simulations < 1 or not 0 < confidence < 1:
        raise ValueError("invalid simulation parameters")

    rng = np.random.default_rng(seed)
    systematic = rng.normal(size=(simulations, 1))
    idiosyncratic = rng.normal(size=(simulations, len(pds)))
    latent = np.sqrt(asset_correlation) * systematic + np.sqrt(1.0 - asset_correlation) * idiosyncratic
    thresholds = norm.ppf(pds)
    defaults = latent < thresholds
    losses = defaults @ (notionals * (1.0 - recoveries))
    var = float(np.quantile(losses, confidence))
    tail = losses[losses >= var]
    expected_shortfall = float(np.mean(tail)) if len(tail) else var
    return CreditPortfolioLossResult(
        losses=losses,
        expected_loss=float(np.mean(losses)),
        value_at_risk=var,
        expected_shortfall=expected_shortfall,
        default_counts=defaults.sum(axis=1),
    )
