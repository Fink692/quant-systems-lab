from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from quantlab.portfolio.drawdown import DrawdownSummary, portfolio_drawdown_summary


@dataclass(frozen=True)
class CDaROptimizationResult:
    weights: pd.Series
    objective_value: float
    drawdown_summary: DrawdownSummary
    success: bool
    message: str


def cdar_minimizing_weights(
    returns: pd.DataFrame | np.ndarray,
    confidence: float = 0.95,
    target_return: float | None = None,
    long_only: bool = True,
    asset_names: list[str] | None = None,
) -> CDaROptimizationResult:
    """Minimize empirical conditional drawdown-at-risk with a linear program."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    frame = _as_return_frame(returns, asset_names)
    values = frame.to_numpy(dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("returns must be a time-by-asset matrix with at least two rows")
    if np.any(~np.isfinite(values)):
        raise ValueError("returns cannot contain non-finite values")

    n_periods, n_assets = values.shape
    cumulative = np.cumsum(values, axis=0)
    peak_slice = slice(n_assets, n_assets + n_periods)
    drawdown_slice = slice(n_assets + n_periods, n_assets + 2 * n_periods)
    eta_index = n_assets + 2 * n_periods
    excess_slice = slice(eta_index + 1, eta_index + 1 + n_periods)
    n_variables = eta_index + 1 + n_periods

    objective = np.zeros(n_variables)
    objective[eta_index] = 1.0
    objective[excess_slice] = 1.0 / ((1.0 - confidence) * n_periods)

    a_eq = np.zeros((1, n_variables))
    a_eq[0, :n_assets] = 1.0
    b_eq = np.array([1.0])

    a_ub_rows: list[np.ndarray] = []
    b_ub: list[float] = []
    if target_return is not None:
        row = np.zeros(n_variables)
        row[:n_assets] = -values.mean(axis=0)
        a_ub_rows.append(row)
        b_ub.append(-float(target_return))

    for idx in range(n_periods):
        peak_idx = peak_slice.start + idx
        drawdown_idx = drawdown_slice.start + idx
        excess_idx = excess_slice.start + idx

        row = np.zeros(n_variables)
        row[:n_assets] = cumulative[idx]
        row[peak_idx] = -1.0
        a_ub_rows.append(row)
        b_ub.append(0.0)

        if idx > 0:
            row = np.zeros(n_variables)
            row[peak_idx - 1] = 1.0
            row[peak_idx] = -1.0
            a_ub_rows.append(row)
            b_ub.append(0.0)

        row = np.zeros(n_variables)
        row[:n_assets] = -cumulative[idx]
        row[peak_idx] = 1.0
        row[drawdown_idx] = -1.0
        a_ub_rows.append(row)
        b_ub.append(0.0)

        row = np.zeros(n_variables)
        row[drawdown_idx] = 1.0
        row[eta_index] = -1.0
        row[excess_idx] = -1.0
        a_ub_rows.append(row)
        b_ub.append(0.0)

    weight_bounds = [(0.0, 1.0)] * n_assets if long_only else [(None, None)] * n_assets
    bounds = (
        weight_bounds
        + [(0.0, None)] * n_periods
        + [(0.0, None)] * n_periods
        + [(None, None)]
        + [(0.0, None)] * n_periods
    )

    result = linprog(
        objective,
        A_ub=np.vstack(a_ub_rows),
        b_ub=np.asarray(b_ub),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(result.message)

    weights = pd.Series(result.x[:n_assets], index=frame.columns, name="weight")
    summary = portfolio_drawdown_summary(frame, weights.to_numpy(), confidence=confidence)
    return CDaROptimizationResult(
        weights=weights,
        objective_value=float(result.fun),
        drawdown_summary=summary,
        success=bool(result.success),
        message=str(result.message),
    )


def _as_return_frame(returns: pd.DataFrame | np.ndarray, asset_names: list[str] | None) -> pd.DataFrame:
    if isinstance(returns, pd.DataFrame):
        return returns.astype(float)
    values = np.asarray(returns, dtype=float)
    if values.ndim != 2:
        raise ValueError("returns must be two-dimensional")
    columns = asset_names if asset_names is not None else [f"asset_{idx}" for idx in range(values.shape[1])]
    if len(columns) != values.shape[1]:
        raise ValueError("asset_names length must match return columns")
    return pd.DataFrame(values, columns=columns)
