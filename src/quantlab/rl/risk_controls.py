from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantlab.rl.evaluation import Policy
from quantlab.rl.trading_env import TradingState


@dataclass(frozen=True)
class RiskLimits:
    max_leverage: float = 1.0
    max_drawdown: float = 0.2
    de_risk_weight: float = 0.0

    def validate(self) -> None:
        if self.max_leverage <= 0:
            raise ValueError("max_leverage must be positive")
        if not 0 < self.max_drawdown < 1:
            raise ValueError("max_drawdown must be in (0, 1)")
        if abs(self.de_risk_weight) > self.max_leverage:
            raise ValueError("de_risk_weight cannot exceed max_leverage")


@dataclass(frozen=True)
class RiskLimitDecision:
    proposed_weight: float
    approved_weight: float
    drawdown: float
    limited: bool


def apply_risk_limits(state: TradingState, proposed_weight: float, limits: RiskLimits) -> RiskLimitDecision:
    """Clip leverage and scale down exposure as drawdown approaches the limit."""
    limits.validate()
    clipped = float(np.clip(proposed_weight, -limits.max_leverage, limits.max_leverage))
    drawdown = 0.0 if state.peak_equity <= 0 else max(1.0 - state.equity / state.peak_equity, 0.0)
    if drawdown >= limits.max_drawdown:
        approved = float(limits.de_risk_weight)
    else:
        buffer_ratio = (limits.max_drawdown - drawdown) / limits.max_drawdown
        scale = min(1.0, max(0.0, buffer_ratio))
        approved = limits.de_risk_weight + scale * (clipped - limits.de_risk_weight)
    return RiskLimitDecision(
        proposed_weight=float(proposed_weight),
        approved_weight=float(np.clip(approved, -limits.max_leverage, limits.max_leverage)),
        drawdown=float(drawdown),
        limited=bool(abs(approved - proposed_weight) > 1e-12),
    )


def risk_limited_policy(policy: Policy, limits: RiskLimits) -> Policy:
    """Wrap a policy with leverage and drawdown controls."""
    limits.validate()

    def wrapped(state: TradingState) -> float:
        return apply_risk_limits(state, policy(state), limits).approved_weight

    return wrapped


def volatility_target_weight(
    signal_weight: float,
    realized_volatility: float,
    target_volatility: float,
    max_leverage: float = 1.0,
) -> float:
    """Scale a signal weight to a target volatility with leverage clipping."""
    if realized_volatility < 0 or target_volatility <= 0 or max_leverage <= 0:
        raise ValueError("volatility and leverage inputs are invalid")
    if realized_volatility == 0:
        return float(np.clip(signal_weight, -max_leverage, max_leverage))
    scaled = signal_weight * target_volatility / realized_volatility
    return float(np.clip(scaled, -max_leverage, max_leverage))
