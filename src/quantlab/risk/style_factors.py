from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StyleFactorResult:
    exposures: pd.DataFrame
    factor_returns: pd.DataFrame
    residuals: pd.DataFrame


def build_style_exposures(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Build simple Barra-like style exposures from fundamentals."""
    required = {"market_cap", "book_to_market", "momentum", "volatility"}
    missing = required - set(fundamentals.columns)
    if missing:
        raise ValueError(f"fundamentals is missing columns: {sorted(missing)}")
    raw = pd.DataFrame(index=fundamentals.index)
    raw["size"] = np.log(fundamentals["market_cap"].astype(float))
    raw["value"] = fundamentals["book_to_market"].astype(float)
    raw["momentum"] = fundamentals["momentum"].astype(float)
    raw["low_volatility"] = -fundamentals["volatility"].astype(float)
    return raw.apply(_zscore, axis=0).fillna(0.0)


def estimate_style_factor_returns(asset_returns: pd.DataFrame, exposures: pd.DataFrame) -> StyleFactorResult:
    """Estimate period-by-period cross-sectional factor returns from style exposures."""
    common_assets = asset_returns.columns.intersection(exposures.index)
    if len(common_assets) < exposures.shape[1]:
        raise ValueError("not enough common assets to estimate style factor returns")
    x = np.column_stack([np.ones(len(common_assets)), exposures.loc[common_assets].to_numpy(dtype=float)])
    factor_rows = []
    residual_rows = []
    for date, row in asset_returns[common_assets].iterrows():
        coefficients = np.linalg.lstsq(x, row.to_numpy(dtype=float), rcond=None)[0]
        fitted = x @ coefficients
        factor_rows.append(pd.Series(coefficients[1:], index=exposures.columns, name=date))
        residual_rows.append(pd.Series(row.to_numpy(dtype=float) - fitted, index=common_assets, name=date))
    return StyleFactorResult(
        exposures=exposures.loc[common_assets],
        factor_returns=pd.DataFrame(factor_rows),
        residuals=pd.DataFrame(residual_rows),
    )


def _zscore(values: pd.Series) -> pd.Series:
    std = values.std(ddof=1)
    if std == 0 or np.isnan(std):
        return values * 0.0
    return (values - values.mean()) / std
