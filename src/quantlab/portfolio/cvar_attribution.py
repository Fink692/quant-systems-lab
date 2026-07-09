from __future__ import annotations

import numpy as np
import pandas as pd


def portfolio_cvar_contributions(
    returns: pd.DataFrame | np.ndarray,
    weights: np.ndarray,
    confidence: float = 0.95,
) -> pd.Series | np.ndarray:
    """Estimate asset contributions to empirical portfolio CVaR."""
    is_frame = isinstance(returns, pd.DataFrame)
    values = returns.to_numpy(dtype=float) if is_frame else np.asarray(returns, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 2 or weights.shape != (values.shape[1],):
        raise ValueError("returns must be scenario-by-asset and weights must match assets")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    portfolio_returns = values @ weights
    threshold = np.quantile(portfolio_returns, 1.0 - confidence)
    tail = values[portfolio_returns <= threshold]
    if len(tail) == 0:
        contributions = np.zeros_like(weights)
    else:
        contributions = -weights * tail.mean(axis=0)
    if is_frame:
        return pd.Series(contributions, index=returns.columns, name="cvar_contribution")
    return contributions
