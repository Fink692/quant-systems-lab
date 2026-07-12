from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorModelResult:
    intercepts: pd.Series
    exposures: pd.DataFrame
    factor_covariance: pd.DataFrame
    specific_variance: pd.Series
    residuals: pd.DataFrame

    def covariance_matrix(self) -> pd.DataFrame:
        systematic = self.exposures.to_numpy() @ self.factor_covariance.to_numpy() @ self.exposures.to_numpy().T
        total = systematic + np.diag(self.specific_variance.to_numpy())
        return pd.DataFrame(total, index=self.exposures.index, columns=self.exposures.index)


def fit_factor_model(asset_returns: pd.DataFrame, factor_returns: pd.DataFrame) -> FactorModelResult:
    """Fit asset returns to factor returns with OLS for each asset."""
    if len(asset_returns) != len(factor_returns):
        raise ValueError("asset_returns and factor_returns must have the same row count")
    aligned_assets, aligned_factors = asset_returns.align(factor_returns, join="inner", axis=0)
    x = np.column_stack([np.ones(len(aligned_factors)), aligned_factors.to_numpy()])
    coefficients = np.linalg.lstsq(x, aligned_assets.to_numpy(), rcond=None)[0]
    intercepts = pd.Series(coefficients[0, :], index=aligned_assets.columns, name="intercept")
    exposures = pd.DataFrame(coefficients[1:, :].T, index=aligned_assets.columns, columns=aligned_factors.columns)
    fitted = x @ coefficients
    residuals = pd.DataFrame(
        aligned_assets.to_numpy() - fitted, index=aligned_assets.index, columns=aligned_assets.columns
    )
    specific_variance = residuals.var(axis=0, ddof=x.shape[1]).clip(lower=0.0)
    factor_covariance = aligned_factors.cov()
    return FactorModelResult(intercepts, exposures, factor_covariance, specific_variance, residuals)


def shrink_covariance(returns: pd.DataFrame, shrinkage: float = 0.2) -> pd.DataFrame:
    """Shrink sample covariance toward a diagonal target."""
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be in [0, 1]")
    sample = returns.cov()
    target = pd.DataFrame(np.diag(np.diag(sample)), index=sample.index, columns=sample.columns)
    return (1.0 - shrinkage) * sample + shrinkage * target
