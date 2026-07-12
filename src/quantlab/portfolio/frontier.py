from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class EfficientFrontierResult:
    points: pd.DataFrame
    weights: pd.DataFrame


def efficient_frontier(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    target_returns: np.ndarray,
    asset_names: list[str] | None = None,
    long_only: bool = True,
) -> EfficientFrontierResult:
    """Compute minimum-variance portfolios for target returns."""
    mu = np.asarray(expected_returns, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    targets = np.asarray(target_returns, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or mu.shape != (cov.shape[0],):
        raise ValueError("expected_returns and covariance shapes are inconsistent")
    if targets.ndim != 1:
        raise ValueError("target_returns must be one-dimensional")
    names = asset_names or [f"asset_{idx}" for idx in range(len(mu))]
    if len(names) != len(mu):
        raise ValueError("asset_names length must match expected_returns")

    weight_rows = []
    point_rows = []
    bounds = [(0.0, 1.0)] * len(mu) if long_only else None
    for target in targets:
        constraints = [
            {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
            {"type": "eq", "fun": lambda weights, target_return=target: weights @ mu - target_return},
        ]
        result = minimize(
            lambda weights: float(weights @ cov @ weights),
            np.full(len(mu), 1.0 / len(mu)),
            bounds=bounds,
            constraints=constraints,
        )
        if not result.success:
            continue
        variance = float(result.x @ cov @ result.x)
        point_rows.append(
            {
                "target_return": float(target),
                "expected_return": float(result.x @ mu),
                "volatility": float(np.sqrt(variance)),
            }
        )
        weight_rows.append(pd.Series(result.x, index=names, name=float(target)))

    return EfficientFrontierResult(points=pd.DataFrame(point_rows), weights=pd.DataFrame(weight_rows))
