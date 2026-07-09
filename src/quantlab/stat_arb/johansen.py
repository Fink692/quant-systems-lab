from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen


@dataclass(frozen=True)
class JohansenResult:
    eigenvalues: np.ndarray
    eigenvectors: pd.DataFrame
    trace_statistics: pd.Series
    trace_critical_values: pd.DataFrame
    max_eigen_statistics: pd.Series
    max_eigen_critical_values: pd.DataFrame


def johansen_test(prices: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> JohansenResult:
    """Run Johansen cointegration test on a price panel."""
    clean = prices.dropna()
    if clean.shape[0] < 10 or clean.shape[1] < 2:
        raise ValueError("prices must contain at least 10 observations and 2 assets")
    result = coint_johansen(clean, det_order=det_order, k_ar_diff=k_ar_diff)
    rank_labels = [f"r<={i}" for i in range(clean.shape[1])]
    vector_labels = [f"vector_{i + 1}" for i in range(clean.shape[1])]
    critical_columns = ["90%", "95%", "99%"]
    return JohansenResult(
        eigenvalues=result.eig,
        eigenvectors=pd.DataFrame(result.evec, index=clean.columns, columns=vector_labels),
        trace_statistics=pd.Series(result.lr1, index=rank_labels, name="trace_statistic"),
        trace_critical_values=pd.DataFrame(result.cvt, index=rank_labels, columns=critical_columns),
        max_eigen_statistics=pd.Series(result.lr2, index=rank_labels, name="max_eigen_statistic"),
        max_eigen_critical_values=pd.DataFrame(result.cvm, index=rank_labels, columns=critical_columns),
    )


def johansen_hedge_vector(
    prices: pd.DataFrame,
    vector_index: int = 0,
    normalize_asset: str | None = None,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> pd.Series:
    """Extract a normalized Johansen hedge vector for basket trading."""
    result = johansen_test(prices, det_order=det_order, k_ar_diff=k_ar_diff)
    if vector_index < 0 or vector_index >= result.eigenvectors.shape[1]:
        raise ValueError("vector_index is out of range")
    vector = result.eigenvectors.iloc[:, vector_index].copy()
    asset = normalize_asset or vector.abs().idxmax()
    if asset not in vector.index:
        raise ValueError("normalize_asset must be one of the price columns")
    if abs(vector.loc[asset]) < 1e-12:
        raise ValueError("cannot normalize by a near-zero vector element")
    return vector / vector.loc[asset]


def basket_spread(prices: pd.DataFrame, hedge_vector: pd.Series) -> pd.Series:
    """Compute a cointegrating basket spread from prices and hedge vector."""
    aligned = prices[hedge_vector.index].dropna()
    return pd.Series(aligned.to_numpy() @ hedge_vector.to_numpy(), index=aligned.index, name="basket_spread")
