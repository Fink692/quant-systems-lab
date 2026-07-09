from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.options.calibration import calibrate_sabr_smile
from quantlab.options.sabr import SABRParams, sabr_implied_volatility


@dataclass(frozen=True)
class SABRSurfaceCalibrationResult:
    parameters: pd.DataFrame
    diagnostics: pd.DataFrame

    @property
    def mean_objective(self) -> float:
        return float(self.diagnostics["objective_value"].mean())


def calibrate_sabr_surface(
    option_chain: pd.DataFrame,
    beta: float = 0.6,
    min_strikes_per_maturity: int = 4,
) -> SABRSurfaceCalibrationResult:
    """Calibrate one SABR smile per maturity from an implied-volatility option chain."""
    required = {"spot", "strike", "maturity", "rate", "dividend", "option_type", "implied_volatility"}
    missing = required - set(option_chain.columns)
    if missing:
        raise ValueError(f"option_chain is missing columns: {sorted(missing)}")
    if min_strikes_per_maturity < 3:
        raise ValueError("min_strikes_per_maturity must be at least 3")

    calls = option_chain[option_chain["option_type"] == "call"].copy()
    if calls.empty:
        raise ValueError("option_chain must contain call rows")

    parameter_rows: list[dict[str, float]] = []
    diagnostic_rows: list[dict[str, float | bool | str]] = []
    for maturity, group in calls.groupby("maturity"):
        smile = group.sort_values("strike")
        if len(smile) < min_strikes_per_maturity:
            continue
        spot = float(smile["spot"].iloc[0])
        rate = float(smile["rate"].iloc[0])
        dividend = float(smile["dividend"].iloc[0])
        forward = spot * np.exp((rate - dividend) * float(maturity))
        result = calibrate_sabr_smile(
            forward=forward,
            maturity=float(maturity),
            strikes=smile["strike"].to_numpy(dtype=float),
            implied_volatilities=smile["implied_volatility"].to_numpy(dtype=float),
            beta=beta,
        )
        parameter_rows.append({"maturity": float(maturity), **result.parameters})
        diagnostic_rows.append(
            {
                "maturity": float(maturity),
                "objective_value": result.objective_value,
                "rmse": float(np.sqrt(np.mean(result.residuals**2))),
                "success": result.success,
                "message": result.message,
                "observations": int(len(smile)),
            }
        )

    if not parameter_rows:
        raise ValueError("no maturities had enough strikes for calibration")
    return SABRSurfaceCalibrationResult(
        parameters=pd.DataFrame(parameter_rows).set_index("maturity").sort_index(),
        diagnostics=pd.DataFrame(diagnostic_rows).set_index("maturity").sort_index(),
    )


def sabr_surface_implied_volatility(
    maturity: float,
    strike: float,
    forward: float,
    calibrated_surface: SABRSurfaceCalibrationResult,
) -> float:
    """Interpolate calibrated SABR parameters by maturity and evaluate implied volatility."""
    if maturity <= 0 or strike <= 0 or forward <= 0:
        raise ValueError("maturity, strike, and forward must be positive")
    params_frame = calibrated_surface.parameters
    maturities = params_frame.index.to_numpy(dtype=float)
    values = {
        column: float(np.interp(maturity, maturities, params_frame[column].to_numpy(dtype=float)))
        for column in ["alpha", "beta", "rho", "nu"]
    }
    return sabr_implied_volatility(forward, strike, maturity, SABRParams(**values))
