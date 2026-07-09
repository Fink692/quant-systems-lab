from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorRiskAttribution:
    total_variance: float
    factor_variance: float
    specific_variance: float
    factor_contributions: pd.Series
    asset_contributions: pd.Series


def factor_risk_attribution(
    weights: pd.Series,
    exposures: pd.DataFrame,
    factor_covariance: pd.DataFrame,
    specific_variance: pd.Series,
) -> FactorRiskAttribution:
    """Decompose portfolio variance into factor and specific contributions."""
    aligned_weights = weights.reindex(exposures.index).astype(float)
    aligned_specific = specific_variance.reindex(exposures.index).astype(float)
    if aligned_weights.isna().any() or aligned_specific.isna().any():
        raise ValueError("weights and specific_variance must cover all exposure assets")
    factor_covariance = factor_covariance.reindex(index=exposures.columns, columns=exposures.columns).astype(float)
    if factor_covariance.isna().any().any():
        raise ValueError("factor_covariance must cover all exposure factors")

    w = aligned_weights.to_numpy()
    b = exposures.to_numpy()
    f_cov = factor_covariance.to_numpy()
    specific = aligned_specific.to_numpy()
    portfolio_factor_exposure = b.T @ w
    factor_marginal = f_cov @ portfolio_factor_exposure
    factor_contributions = portfolio_factor_exposure * factor_marginal
    factor_variance = float(np.sum(factor_contributions))
    specific_variance_total = float(np.sum((w**2) * specific))
    total_covariance = b @ f_cov @ b.T + np.diag(specific)
    asset_contributions = w * (total_covariance @ w)
    total_variance = float(w @ total_covariance @ w)
    return FactorRiskAttribution(
        total_variance=total_variance,
        factor_variance=factor_variance,
        specific_variance=specific_variance_total,
        factor_contributions=pd.Series(factor_contributions, index=exposures.columns, name="variance_contribution"),
        asset_contributions=pd.Series(asset_contributions, index=exposures.index, name="variance_contribution"),
    )
