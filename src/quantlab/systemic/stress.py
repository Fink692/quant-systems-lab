from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StressResult:
    equity_after_shock: np.ndarray
    capital_shortfall: np.ndarray
    defaulted: np.ndarray


def external_asset_stress(
    holdings: np.ndarray,
    asset_returns: np.ndarray,
    capital: np.ndarray,
) -> StressResult:
    """Apply asset shocks to institution holdings and compare losses with capital."""
    holdings = np.asarray(holdings, dtype=float)
    asset_returns = np.asarray(asset_returns, dtype=float)
    capital = np.asarray(capital, dtype=float)
    if holdings.ndim != 2:
        raise ValueError("holdings must be institution-by-asset")
    if asset_returns.shape != (holdings.shape[1],):
        raise ValueError("asset_returns must have one value per asset")
    if capital.shape != (holdings.shape[0],):
        raise ValueError("capital must have one value per institution")
    pnl = holdings @ asset_returns
    equity_after = capital + pnl
    shortfall = np.maximum(-equity_after, 0.0)
    return StressResult(equity_after_shock=equity_after, capital_shortfall=shortfall, defaulted=equity_after < 0.0)


def exposure_centrality(exposures: np.ndarray, names: list[str] | None = None) -> pd.DataFrame:
    """Summarize network in/out exposure concentration."""
    exposures = np.asarray(exposures, dtype=float)
    if exposures.ndim != 2 or exposures.shape[0] != exposures.shape[1]:
        raise ValueError("exposures must be square")
    labels = names if names is not None else [str(i) for i in range(exposures.shape[0])]
    if len(labels) != exposures.shape[0]:
        raise ValueError("names length must match exposures")
    out_exposure = exposures.sum(axis=1)
    in_exposure = exposures.sum(axis=0)
    total = exposures.sum()
    systemic_share = (out_exposure + in_exposure) / total if total > 0 else np.zeros_like(out_exposure)
    return pd.DataFrame(
        {
            "out_exposure": out_exposure,
            "in_exposure": in_exposure,
            "systemic_share": systemic_share,
        },
        index=labels,
    )
