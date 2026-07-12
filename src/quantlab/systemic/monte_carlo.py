from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SystemicMonteCarloResult:
    asset_returns: pd.DataFrame
    institution_equity: pd.DataFrame
    shortfalls: pd.DataFrame
    total_shortfall: pd.Series
    default_probabilities: pd.Series
    value_at_risk: float
    expected_shortfall: float
    tail_shortfall_contribution: pd.Series

    @property
    def expected_total_shortfall(self) -> float:
        return float(self.total_shortfall.mean())

    @property
    def max_default_probability(self) -> float:
        return float(self.default_probabilities.max())


def simulate_systemic_monte_carlo(
    holdings: np.ndarray,
    capital: np.ndarray,
    mean_returns: np.ndarray,
    covariance: np.ndarray,
    simulations: int = 10_000,
    confidence: float = 0.95,
    institution_names: list[str] | None = None,
    asset_names: list[str] | None = None,
    seed: int | None = None,
) -> SystemicMonteCarloResult:
    """Monte Carlo asset shocks for systemic shortfall and default probabilities."""
    holdings = np.asarray(holdings, dtype=float)
    capital = np.asarray(capital, dtype=float)
    mean = np.asarray(mean_returns, dtype=float)
    cov = np.asarray(covariance, dtype=float)
    if holdings.ndim != 2:
        raise ValueError("holdings must be institution-by-asset")
    n_institutions, n_assets = holdings.shape
    if capital.shape != (n_institutions,) or mean.shape != (n_assets,) or cov.shape != (n_assets, n_assets):
        raise ValueError("capital, mean_returns, and covariance shapes are inconsistent with holdings")
    if simulations < 1 or not 0 < confidence < 1:
        raise ValueError("simulations must be positive and confidence in (0, 1)")
    if np.any(holdings < 0) or np.any(capital <= 0):
        raise ValueError("holdings must be non-negative and capital positive")
    if not np.allclose(cov, cov.T):
        raise ValueError("covariance must be symmetric")
    if np.min(np.linalg.eigvalsh(cov)) < -1e-10:
        raise ValueError("covariance must be positive semidefinite")

    institution_labels = institution_names or [f"institution_{idx}" for idx in range(n_institutions)]
    asset_labels = asset_names or [f"asset_{idx}" for idx in range(n_assets)]
    if len(institution_labels) != n_institutions or len(asset_labels) != n_assets:
        raise ValueError("name lengths must match holdings")

    rng = np.random.default_rng(seed)
    shocks = rng.multivariate_normal(mean, cov, size=simulations)
    pnl = shocks @ holdings.T
    equity = capital[None, :] + pnl
    shortfalls = np.maximum(-equity, 0.0)
    total_shortfall = shortfalls.sum(axis=1)
    var = float(np.quantile(total_shortfall, confidence))
    tail_mask = total_shortfall >= var
    tail_shortfalls = shortfalls[tail_mask]
    tail_contribution = tail_shortfalls.mean(axis=0) if len(tail_shortfalls) else np.zeros(n_institutions)
    default_probabilities = (equity < 0.0).mean(axis=0)
    index = pd.RangeIndex(simulations, name="scenario")
    return SystemicMonteCarloResult(
        asset_returns=pd.DataFrame(shocks, index=index, columns=asset_labels),
        institution_equity=pd.DataFrame(equity, index=index, columns=institution_labels),
        shortfalls=pd.DataFrame(shortfalls, index=index, columns=institution_labels),
        total_shortfall=pd.Series(total_shortfall, index=index, name="total_shortfall"),
        default_probabilities=pd.Series(default_probabilities, index=institution_labels, name="default_probability"),
        value_at_risk=var,
        expected_shortfall=float(total_shortfall[tail_mask].mean()) if np.any(tail_mask) else var,
        tail_shortfall_contribution=pd.Series(
            tail_contribution, index=institution_labels, name="tail_shortfall_contribution"
        ),
    )
