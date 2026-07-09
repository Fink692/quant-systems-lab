from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class InventoryDiagnostics:
    max_inventory: float
    mean_abs_inventory: float
    inventory_var: float
    inventory_penalty: float
    pnl_inventory_correlation: float


def inventory_diagnostics(
    history: pd.DataFrame,
    inventory_limit: float,
    penalty_coefficient: float = 1.0,
) -> InventoryDiagnostics:
    """Summarize inventory risk from a market-making simulation history."""
    required = {"inventory", "pnl"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"history is missing columns: {sorted(missing)}")
    if inventory_limit <= 0 or penalty_coefficient < 0:
        raise ValueError("inventory_limit must be positive and penalty_coefficient non-negative")
    inventory = history["inventory"].to_numpy(dtype=float)
    pnl = history["pnl"].to_numpy(dtype=float)
    excess = np.maximum(np.abs(inventory) - inventory_limit, 0.0)
    corr = 0.0 if np.std(inventory) == 0 or np.std(pnl) == 0 else float(np.corrcoef(inventory, pnl)[0, 1])
    return InventoryDiagnostics(
        max_inventory=float(np.max(np.abs(inventory))),
        mean_abs_inventory=float(np.mean(np.abs(inventory))),
        inventory_var=float(np.var(inventory, ddof=1)) if len(inventory) > 1 else 0.0,
        inventory_penalty=float(penalty_coefficient * np.mean(excess**2)),
        pnl_inventory_correlation=corr,
    )


def inventory_skew_quote_adjustment(
    inventory: float,
    inventory_limit: float,
    tick_size: float,
    max_skew_ticks: int = 5,
) -> tuple[float, float]:
    """Return bid/ask quote adjustments that lean against inventory."""
    if inventory_limit <= 0 or tick_size <= 0 or max_skew_ticks < 0:
        raise ValueError("inventory_limit and tick_size must be positive; max_skew_ticks non-negative")
    normalized = float(np.clip(inventory / inventory_limit, -1.0, 1.0))
    skew = round(normalized * max_skew_ticks) * tick_size
    return -skew, -skew
