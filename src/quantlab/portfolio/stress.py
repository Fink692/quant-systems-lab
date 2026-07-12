from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PortfolioStressResult:
    scenario_pnl: pd.Series
    scenario_return: pd.Series
    worst_scenario: str


def stress_test_portfolio(
    weights: pd.Series,
    scenarios: pd.DataFrame,
    portfolio_value: float = 1.0,
) -> PortfolioStressResult:
    """Apply asset-return scenarios to a portfolio."""
    if portfolio_value <= 0:
        raise ValueError("portfolio_value must be positive")
    aligned = scenarios.reindex(columns=weights.index)
    if aligned.isna().any().any():
        raise ValueError("scenarios must cover every weighted asset")
    scenario_return = aligned @ weights.astype(float)
    scenario_pnl = scenario_return * portfolio_value
    return PortfolioStressResult(
        scenario_pnl=scenario_pnl.rename("pnl"),
        scenario_return=scenario_return.rename("return"),
        worst_scenario=str(scenario_pnl.idxmin()),
    )


def historical_stress_scenarios(returns: pd.DataFrame, quantile: float = 0.05) -> pd.DataFrame:
    """Build simple historical stress scenarios from marginal asset quantiles."""
    if not 0 < quantile < 1:
        raise ValueError("quantile must be in (0, 1)")
    downside = returns.quantile(quantile)
    upside = returns.quantile(1.0 - quantile)
    shock = returns.min()
    return pd.DataFrame(
        [downside, upside, shock],
        index=["downside_quantile", "upside_quantile", "historical_worst"],
    )
