from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class DistanceToDefaultResult:
    distance_to_default: float
    expected_default_frequency: float


def distance_to_default(
    asset_value: float,
    default_point: float,
    asset_volatility: float,
    drift: float = 0.0,
    horizon: float = 1.0,
) -> DistanceToDefaultResult:
    """KMV-style distance-to-default and expected default frequency."""
    if min(asset_value, default_point, asset_volatility, horizon) <= 0:
        raise ValueError("asset_value, default_point, asset_volatility, and horizon must be positive")
    dd = (np.log(asset_value / default_point) + (drift - 0.5 * asset_volatility**2) * horizon) / (
        asset_volatility * np.sqrt(horizon)
    )
    return DistanceToDefaultResult(distance_to_default=float(dd), expected_default_frequency=float(norm.cdf(-dd)))


def default_point(short_term_debt: float, long_term_debt: float, long_term_weight: float = 0.5) -> float:
    if short_term_debt < 0 or long_term_debt < 0 or not 0 <= long_term_weight <= 1:
        raise ValueError("debt values must be non-negative and long_term_weight in [0, 1]")
    return float(short_term_debt + long_term_weight * long_term_debt)
