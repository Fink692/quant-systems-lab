from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrancheLossResult:
    loss_rates: np.ndarray
    expected_loss_rate: float
    value_at_risk: float
    expected_shortfall: float
    attachment: float
    detachment: float


def tranche_loss_distribution(
    portfolio_losses: np.ndarray,
    attachment: float,
    detachment: float,
    portfolio_notional: float,
    confidence: float = 0.99,
) -> TrancheLossResult:
    """Transform portfolio losses into tranche loss rates."""
    losses = np.asarray(portfolio_losses, dtype=float)
    if losses.ndim != 1 or len(losses) == 0 or np.any(losses < 0):
        raise ValueError("portfolio_losses must be a non-empty non-negative vector")
    if not 0 <= attachment < detachment <= 1:
        raise ValueError("attachment and detachment must satisfy 0 <= a < d <= 1")
    if portfolio_notional <= 0 or not 0 < confidence < 1:
        raise ValueError("portfolio_notional and confidence are invalid")

    attach_amount = attachment * portfolio_notional
    tranche_width = (detachment - attachment) * portfolio_notional
    tranche_losses = np.clip(losses - attach_amount, 0.0, tranche_width)
    loss_rates = tranche_losses / tranche_width
    var = float(np.quantile(loss_rates, confidence))
    tail = loss_rates[loss_rates >= var]
    expected_shortfall = float(tail.mean()) if len(tail) else var
    return TrancheLossResult(
        loss_rates=loss_rates,
        expected_loss_rate=float(loss_rates.mean()),
        value_at_risk=var,
        expected_shortfall=expected_shortfall,
        attachment=float(attachment),
        detachment=float(detachment),
    )
