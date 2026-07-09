from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.systemic.stress import external_asset_stress


@dataclass(frozen=True)
class SystemicScenarioResult:
    scenario_results: pd.DataFrame
    institution_shortfalls: pd.DataFrame
    worst_scenario: str
    expected_shortfall: float


def run_systemic_stress_scenarios(
    holdings: np.ndarray,
    scenario_returns: pd.DataFrame,
    capital: np.ndarray,
    scenario_probabilities: pd.Series | np.ndarray | None = None,
) -> SystemicScenarioResult:
    """Apply multiple asset-return stress scenarios and aggregate shortfalls/defaults."""
    if scenario_returns.empty:
        raise ValueError("scenario_returns cannot be empty")
    holdings = np.asarray(holdings, dtype=float)
    capital = np.asarray(capital, dtype=float)
    if holdings.ndim != 2 or scenario_returns.shape[1] != holdings.shape[1]:
        raise ValueError("scenario_returns must have one column per held asset")
    if capital.shape != (holdings.shape[0],):
        raise ValueError("capital must have one value per institution")
    if scenario_probabilities is None:
        probabilities = pd.Series(1.0 / len(scenario_returns), index=scenario_returns.index)
    else:
        probabilities = pd.Series(scenario_probabilities, index=scenario_returns.index, dtype=float)
        if probabilities.isna().any() or (probabilities < 0).any() or probabilities.sum() <= 0:
            raise ValueError("scenario_probabilities must be non-negative and non-zero")
        probabilities = probabilities / probabilities.sum()

    rows = []
    shortfall_rows = []
    for scenario_name, returns in scenario_returns.iterrows():
        result = external_asset_stress(holdings, returns.to_numpy(dtype=float), capital)
        total_shortfall = float(result.capital_shortfall.sum())
        rows.append(
            {
                "scenario": scenario_name,
                "probability": float(probabilities.loc[scenario_name]),
                "default_count": int(result.defaulted.sum()),
                "total_shortfall": total_shortfall,
                "weighted_shortfall": float(probabilities.loc[scenario_name] * total_shortfall),
                "minimum_equity": float(result.equity_after_shock.min()),
            }
        )
        shortfall_rows.append(pd.Series(result.capital_shortfall, name=scenario_name))

    scenario_frame = pd.DataFrame(rows).set_index("scenario")
    shortfall_frame = pd.DataFrame(shortfall_rows)
    shortfall_frame.columns = [f"institution_{idx}" for idx in range(holdings.shape[0])]
    return SystemicScenarioResult(
        scenario_results=scenario_frame,
        institution_shortfalls=shortfall_frame,
        worst_scenario=str(scenario_frame["total_shortfall"].idxmax()),
        expected_shortfall=float(scenario_frame["weighted_shortfall"].sum()),
    )
