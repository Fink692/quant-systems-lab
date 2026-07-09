from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class PCAFactorResult:
    factor_returns: pd.DataFrame
    loadings: pd.DataFrame
    explained_variance_ratio: pd.Series


def fit_pca_factor_model(returns: pd.DataFrame, n_factors: int = 3, standardize: bool = True) -> PCAFactorResult:
    """Extract statistical factors from an asset return matrix."""
    if n_factors < 1:
        raise ValueError("n_factors must be positive")
    if n_factors > returns.shape[1]:
        raise ValueError("n_factors cannot exceed the number of assets")

    clean = returns.dropna()
    values = clean.to_numpy()
    if standardize:
        values = StandardScaler().fit_transform(values)

    pca = PCA(n_components=n_factors)
    factor_values = pca.fit_transform(values)
    factor_names = [f"PC{i + 1}" for i in range(n_factors)]
    return PCAFactorResult(
        factor_returns=pd.DataFrame(factor_values, index=clean.index, columns=factor_names),
        loadings=pd.DataFrame(pca.components_.T, index=clean.columns, columns=factor_names),
        explained_variance_ratio=pd.Series(pca.explained_variance_ratio_, index=factor_names, name="explained_variance_ratio"),
    )
