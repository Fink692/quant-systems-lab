from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CapitalAdequacyResult:
    capital_ratio: pd.Series
    capital_shortfall: pd.Series
    total_shortfall: float


def capital_adequacy(
    capital: pd.Series | np.ndarray,
    risk_weighted_assets: pd.Series | np.ndarray,
    minimum_ratio: float = 0.08,
    names: list[str] | None = None,
) -> CapitalAdequacyResult:
    """Compute capital ratios and shortfalls against a minimum capital ratio."""
    capital_values = np.asarray(capital, dtype=float)
    rwa = np.asarray(risk_weighted_assets, dtype=float)
    if capital_values.shape != rwa.shape or capital_values.ndim != 1:
        raise ValueError("capital and risk_weighted_assets must be same-length vectors")
    if np.any(rwa <= 0) or minimum_ratio < 0:
        raise ValueError("risk_weighted_assets must be positive and minimum_ratio non-negative")
    index = _index(capital, names)
    ratio = capital_values / rwa
    required = minimum_ratio * rwa
    shortfall = np.maximum(required - capital_values, 0.0)
    return CapitalAdequacyResult(
        capital_ratio=pd.Series(ratio, index=index, name="capital_ratio"),
        capital_shortfall=pd.Series(shortfall, index=index, name="capital_shortfall"),
        total_shortfall=float(shortfall.sum()),
    )


def systemic_capital_surcharge(
    systemic_scores: pd.Series,
    base_ratio: float = 0.08,
    surcharge_scale: float = 0.04,
) -> pd.Series:
    """Map normalized systemic-importance scores to capital ratio requirements."""
    if base_ratio < 0 or surcharge_scale < 0:
        raise ValueError("base_ratio and surcharge_scale must be non-negative")
    scores = systemic_scores.astype(float).clip(lower=0.0)
    max_score = scores.max()
    normalized = scores * 0.0 if max_score == 0 else scores / max_score
    return (base_ratio + surcharge_scale * normalized).rename("capital_requirement")


def _index(values: pd.Series | np.ndarray, names: list[str] | None) -> pd.Index:
    if isinstance(values, pd.Series):
        return values.index
    if names is not None:
        return pd.Index(names)
    return pd.Index([str(idx) for idx in range(len(values))])
