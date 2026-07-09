from __future__ import annotations

import numpy as np
from scipy.stats import norm


def merton_default_probability(
    asset_value: float,
    debt_face_value: float,
    maturity: float,
    rate: float,
    asset_volatility: float,
) -> float:
    """Risk-neutral default probability in the Merton structural model."""
    if asset_value <= 0 or debt_face_value <= 0:
        raise ValueError("asset_value and debt_face_value must be positive")
    if maturity <= 0 or asset_volatility <= 0:
        raise ValueError("maturity and asset_volatility must be positive")
    d2 = (np.log(asset_value / debt_face_value) + (rate - 0.5 * asset_volatility**2) * maturity) / (
        asset_volatility * np.sqrt(maturity)
    )
    return float(norm.cdf(-d2))


def survival_probability(hazard_rate: float, maturity: float) -> float:
    if hazard_rate < 0 or maturity < 0:
        raise ValueError("hazard_rate and maturity must be non-negative")
    return float(np.exp(-hazard_rate * maturity))


def credit_spread_from_hazard(hazard_rate: float, recovery_rate: float) -> float:
    """Reduced-form approximation: spread = hazard * loss-given-default."""
    if hazard_rate < 0:
        raise ValueError("hazard_rate must be non-negative")
    if not 0 <= recovery_rate <= 1:
        raise ValueError("recovery_rate must be in [0, 1]")
    return float(hazard_rate * (1.0 - recovery_rate))
