from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

MacroTransform = Literal["diff", "logdiff", "level"]


@dataclass(frozen=True)
class MacroFactorModelResult:
    intercepts: pd.Series
    exposures: pd.DataFrame
    factor_returns: pd.DataFrame
    factor_covariance: pd.DataFrame
    specific_variance: pd.Series
    residuals: pd.DataFrame
    r_squared: pd.Series

    def covariance_matrix(self) -> pd.DataFrame:
        systematic = self.exposures.to_numpy() @ self.factor_covariance.to_numpy() @ self.exposures.to_numpy().T
        total = systematic + np.diag(self.specific_variance.to_numpy())
        return pd.DataFrame(total, index=self.exposures.index, columns=self.exposures.index)

    def portfolio_exposure(self, weights: pd.Series) -> pd.Series:
        common_assets = weights.index.intersection(self.exposures.index)
        if len(common_assets) == 0:
            raise ValueError("weights and exposures must share assets")
        exposure = self.exposures.loc[common_assets].T @ weights.loc[common_assets].astype(float)
        return pd.Series(exposure, index=self.exposures.columns, name="portfolio_macro_exposure")

    def stress_pnl(self, weights: pd.Series, factor_shocks: pd.Series, portfolio_value: float = 1.0) -> float:
        if portfolio_value <= 0:
            raise ValueError("portfolio_value must be positive")
        shocks = factor_shocks.reindex(self.exposures.columns).astype(float)
        if shocks.isna().any():
            raise ValueError("factor_shocks must cover every macro factor")
        exposure = self.portfolio_exposure(weights)
        return float(portfolio_value * (exposure @ shocks))


def macro_surprise_factors(
    indicators: pd.DataFrame,
    transform: MacroTransform = "diff",
    standardize: bool = True,
    rolling_window: int | None = None,
) -> pd.DataFrame:
    """Convert macro indicator levels into aligned surprise-style factor returns."""
    if indicators.empty:
        raise ValueError("indicators cannot be empty")
    data = indicators.astype(float)
    if data.isna().any().any():
        raise ValueError("indicators cannot contain missing values")
    if transform == "diff":
        factors = data.diff().dropna()
    elif transform == "logdiff":
        if (data <= 0).any().any():
            raise ValueError("logdiff requires positive indicator levels")
        factors = np.log(data).diff().dropna()
    elif transform == "level":
        factors = data.copy()
    else:
        raise ValueError("transform must be 'diff', 'logdiff', or 'level'")

    if not standardize:
        return factors
    if rolling_window is not None:
        if rolling_window < 2:
            raise ValueError("rolling_window must be at least 2")
        mean = factors.rolling(rolling_window).mean()
        std = factors.rolling(rolling_window).std(ddof=1)
        standardized = ((factors - mean) / std.replace(0.0, np.nan)).dropna()
    else:
        standardized = factors.apply(_zscore, axis=0)
    return standardized.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def fit_macro_factor_model(
    asset_returns: pd.DataFrame,
    macro_factors: pd.DataFrame,
    ridge: float = 0.0,
) -> MacroFactorModelResult:
    """Fit asset returns to macro factor surprises with time-series regressions."""
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    aligned_assets, aligned_factors = asset_returns.align(macro_factors, join="inner", axis=0)
    if aligned_assets.empty or aligned_factors.empty:
        raise ValueError("asset_returns and macro_factors must overlap in time")
    if len(aligned_assets) <= aligned_factors.shape[1] + 1:
        raise ValueError("not enough observations for the macro factor model")
    if aligned_assets.isna().any().any() or aligned_factors.isna().any().any():
        raise ValueError("asset_returns and macro_factors cannot contain missing values")

    x = np.column_stack([np.ones(len(aligned_factors)), aligned_factors.to_numpy(dtype=float)])
    penalty = np.eye(x.shape[1]) * ridge
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(x.T @ x + penalty, x.T @ aligned_assets.to_numpy(dtype=float))
    fitted = x @ coefficients
    residual_values = aligned_assets.to_numpy(dtype=float) - fitted
    residuals = pd.DataFrame(residual_values, index=aligned_assets.index, columns=aligned_assets.columns)
    intercepts = pd.Series(coefficients[0, :], index=aligned_assets.columns, name="intercept")
    exposures = pd.DataFrame(coefficients[1:, :].T, index=aligned_assets.columns, columns=aligned_factors.columns)
    specific_variance = residuals.var(axis=0, ddof=x.shape[1]).clip(lower=0.0)
    total_variance = aligned_assets.var(axis=0, ddof=1).replace(0.0, np.nan)
    r_squared = (1.0 - residuals.var(axis=0, ddof=1) / total_variance).fillna(0.0).clip(lower=0.0, upper=1.0)
    r_squared.name = "r_squared"
    return MacroFactorModelResult(
        intercepts=intercepts,
        exposures=exposures,
        factor_returns=aligned_factors,
        factor_covariance=aligned_factors.cov(),
        specific_variance=specific_variance,
        residuals=residuals,
        r_squared=r_squared,
    )


def _zscore(values: pd.Series) -> pd.Series:
    std = values.std(ddof=1)
    if std == 0.0 or np.isnan(std):
        return values * 0.0
    return (values - values.mean()) / std
