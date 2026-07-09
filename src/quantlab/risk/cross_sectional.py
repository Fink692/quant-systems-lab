from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CrossSectionalFactorResult:
    exposures: pd.DataFrame
    factor_returns: pd.DataFrame
    intercepts: pd.Series
    residuals: pd.DataFrame
    specific_variance: pd.Series

    def factor_covariance(self) -> pd.DataFrame:
        return self.factor_returns.cov()


def build_sector_exposures(sectors: pd.Series, prefix: str = "sector") -> pd.DataFrame:
    """Create one-hot sector exposures for cross-sectional risk models."""
    if sectors.empty or sectors.isna().any():
        raise ValueError("sectors must be non-empty and cannot contain missing values")
    labels = sectors.astype(str)
    exposures = pd.get_dummies(labels, prefix=prefix, dtype=float)
    exposures.index = sectors.index
    return exposures


def estimate_cross_sectional_factor_returns(
    asset_returns: pd.DataFrame,
    exposures: pd.DataFrame,
    regression_weights: pd.Series | None = None,
    ridge: float = 0.0,
) -> CrossSectionalFactorResult:
    """Estimate period-by-period factor returns from asset exposures with WLS."""
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    common_assets = asset_returns.columns.intersection(exposures.index)
    if len(common_assets) <= exposures.shape[1]:
        raise ValueError("not enough common assets for the exposure matrix")
    x_raw = exposures.loc[common_assets].astype(float)
    if regression_weights is None:
        weights = pd.Series(1.0, index=common_assets)
    else:
        weights = regression_weights.reindex(common_assets).astype(float)
        if weights.isna().any() or (weights <= 0).any():
            raise ValueError("regression_weights must be positive for every common asset")

    x = np.column_stack([np.ones(len(common_assets)), x_raw.to_numpy()])
    sqrt_w = np.sqrt(weights.to_numpy())
    x_weighted = x * sqrt_w[:, None]
    penalty = np.eye(x.shape[1]) * ridge
    penalty[0, 0] = 0.0
    factor_rows = []
    intercepts = []
    residual_rows = []
    for date, row in asset_returns[common_assets].iterrows():
        y = row.to_numpy(dtype=float)
        y_weighted = y * sqrt_w
        coefficients = np.linalg.solve(x_weighted.T @ x_weighted + penalty, x_weighted.T @ y_weighted)
        fitted = x @ coefficients
        intercepts.append(float(coefficients[0]))
        factor_rows.append(pd.Series(coefficients[1:], index=x_raw.columns, name=date))
        residual_rows.append(pd.Series(y - fitted, index=common_assets, name=date))

    residuals = pd.DataFrame(residual_rows)
    return CrossSectionalFactorResult(
        exposures=x_raw,
        factor_returns=pd.DataFrame(factor_rows),
        intercepts=pd.Series(intercepts, index=asset_returns.index, name="intercept"),
        residuals=residuals,
        specific_variance=residuals.var(axis=0, ddof=1).clip(lower=0.0),
    )


def neutralize_portfolio_exposures(
    weights: pd.Series,
    exposures: pd.DataFrame,
    target_exposure: pd.Series | None = None,
    preserve_sum: bool = True,
) -> pd.Series:
    """Find the closest portfolio whose factor exposures match a target."""
    common_assets = weights.index.intersection(exposures.index)
    if len(common_assets) == 0:
        raise ValueError("weights and exposures must share assets")
    w = weights.loc[common_assets].astype(float).to_numpy()
    b = exposures.loc[common_assets].astype(float).to_numpy()
    target = (
        pd.Series(0.0, index=exposures.columns)
        if target_exposure is None
        else target_exposure.reindex(exposures.columns).astype(float)
    )
    if target.isna().any():
        raise ValueError("target_exposure must cover every exposure column")
    constraints = [b.T]
    values = [target.to_numpy()]
    if preserve_sum:
        constraints.append(np.ones((1, len(common_assets))))
        values.append(np.array([weights.loc[common_assets].sum()]))
    a = np.vstack(constraints)
    c = np.concatenate(values)
    correction = a.T @ np.linalg.pinv(a @ a.T) @ (a @ w - c)
    adjusted = w - correction
    return pd.Series(adjusted, index=common_assets, name=weights.name or "neutralized_weight")


def factor_mimicking_portfolios(
    exposures: pd.DataFrame,
    asset_covariance: pd.DataFrame | np.ndarray | None = None,
    ridge: float = 1e-8,
) -> pd.DataFrame:
    """Construct minimum-risk portfolios with unit exposure to each factor."""
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    b = exposures.astype(float).to_numpy()
    if b.ndim != 2 or b.shape[0] <= b.shape[1]:
        raise ValueError("exposures must have more assets than factors")
    if asset_covariance is None:
        inv_cov = np.eye(b.shape[0])
    else:
        cov = np.asarray(asset_covariance, dtype=float)
        if cov.shape != (b.shape[0], b.shape[0]):
            raise ValueError("asset_covariance shape must match exposure assets")
        inv_cov = np.linalg.pinv(cov + ridge * np.eye(cov.shape[0]))
    middle = np.linalg.pinv(b.T @ inv_cov @ b)
    portfolios = middle @ b.T @ inv_cov
    return pd.DataFrame(portfolios, index=exposures.columns, columns=exposures.index)
