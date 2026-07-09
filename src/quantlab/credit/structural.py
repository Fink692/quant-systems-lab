from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import root
from scipy.stats import norm


@dataclass(frozen=True)
class MertonCalibrationResult:
    asset_value: float
    asset_volatility: float
    success: bool
    message: str


def merton_equity_value(
    asset_value: float,
    debt_face_value: float,
    maturity: float,
    rate: float,
    asset_volatility: float,
) -> float:
    """Equity value as a call option on firm assets in the Merton model."""
    if min(asset_value, debt_face_value, maturity, asset_volatility) <= 0:
        raise ValueError("asset_value, debt_face_value, maturity, and asset_volatility must be positive")
    d1 = (np.log(asset_value / debt_face_value) + (rate + 0.5 * asset_volatility**2) * maturity) / (
        asset_volatility * np.sqrt(maturity)
    )
    d2 = d1 - asset_volatility * np.sqrt(maturity)
    return float(asset_value * norm.cdf(d1) - debt_face_value * np.exp(-rate * maturity) * norm.cdf(d2))


def calibrate_merton_asset_parameters(
    equity_value: float,
    equity_volatility: float,
    debt_face_value: float,
    maturity: float,
    rate: float,
) -> MertonCalibrationResult:
    """Infer asset value and asset volatility from observed equity value and volatility."""
    if min(equity_value, equity_volatility, debt_face_value, maturity) <= 0:
        raise ValueError("equity_value, equity_volatility, debt_face_value, and maturity must be positive")

    def equations(raw: np.ndarray) -> np.ndarray:
        asset_value = np.exp(raw[0])
        asset_volatility = np.exp(raw[1])
        d1 = (np.log(asset_value / debt_face_value) + (rate + 0.5 * asset_volatility**2) * maturity) / (
            asset_volatility * np.sqrt(maturity)
        )
        equity = merton_equity_value(asset_value, debt_face_value, maturity, rate, asset_volatility)
        modeled_equity_vol = norm.cdf(d1) * asset_value * asset_volatility / equity
        return np.array([equity - equity_value, modeled_equity_vol - equity_volatility])

    initial_asset = equity_value + debt_face_value * np.exp(-rate * maturity)
    initial_vol = max(0.05, equity_value / initial_asset * equity_volatility)
    result = root(equations, np.log([initial_asset, initial_vol]))
    asset_value, asset_volatility = np.exp(result.x)
    return MertonCalibrationResult(
        asset_value=float(asset_value),
        asset_volatility=float(asset_volatility),
        success=bool(result.success),
        message=str(result.message),
    )
