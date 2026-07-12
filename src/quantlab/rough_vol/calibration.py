from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.rough_vol.rough_bergomi import RoughBergomiParams
from quantlab.rough_vol.skew import fit_atm_skew_power_law


@dataclass(frozen=True)
class RoughnessEstimate:
    hurst: float
    intercept: float
    r_squared: float


@dataclass(frozen=True)
class RoughBergomiCalibrationResult:
    params: RoughBergomiParams
    objective_value: float
    atm_vol_rmse: float
    atm_skew_rmse: float
    atm_volatilities: pd.Series
    atm_skews: pd.Series
    success: bool
    message: str


def estimate_hurst_from_variogram(log_variance_proxy: np.ndarray, max_lag: int = 20) -> RoughnessEstimate:
    """Estimate rough-volatility H from log-variance variogram scaling."""
    series = np.asarray(log_variance_proxy, dtype=float)
    if series.ndim != 1:
        raise ValueError("log_variance_proxy must be one-dimensional")
    if max_lag < 2 or max_lag >= len(series):
        raise ValueError("max_lag must be at least 2 and smaller than the series length")

    lags = np.arange(1, max_lag + 1)
    variogram = np.array([np.mean((series[lag:] - series[:-lag]) ** 2) for lag in lags])
    valid = variogram > 0
    if valid.sum() < 2:
        raise ValueError("variogram is degenerate")

    x = np.log(lags[valid])
    y = np.log(variogram[valid])
    slope, intercept = np.polyfit(x, y, deg=1)
    fitted = slope * x + intercept
    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return RoughnessEstimate(
        hurst=float(np.clip(0.5 * slope, 0.0, 1.0)), intercept=float(intercept), r_squared=float(r_squared)
    )


def calibrate_rough_bergomi_atm(
    maturities: np.ndarray,
    atm_volatilities: np.ndarray,
    atm_skews: np.ndarray,
    rho: float = -0.55,
) -> RoughBergomiCalibrationResult:
    """Calibrate a fast rough Bergomi ATM proxy from vol and skew term structures."""
    times = np.asarray(maturities, dtype=float)
    vols = np.asarray(atm_volatilities, dtype=float)
    skews = np.asarray(atm_skews, dtype=float)
    if times.ndim != 1 or vols.shape != times.shape or skews.shape != times.shape or len(times) < 2:
        raise ValueError(
            "maturities, atm_volatilities, and atm_skews must be matching vectors with at least two points"
        )
    if np.any(times <= 0) or np.any(vols <= 0) or np.any(skews == 0):
        raise ValueError("maturities and vols must be positive, and skews cannot be zero")
    if not -1 < rho < 1 or abs(rho) < 1e-8:
        raise ValueError("rho must be non-zero and in (-1, 1)")
    skew_sign = float(np.sign(np.median(skews)))
    if skew_sign != float(np.sign(rho)):
        raise ValueError("rho sign must match the dominant ATM skew sign")

    skew_fit = fit_atm_skew_power_law(times, skews)
    hurst = float(np.clip(skew_fit.hurst, 1e-4, 0.499))
    xi0 = float(np.mean(vols**2))
    eta = float(skew_fit.level / (abs(rho) * np.sqrt(xi0)))
    params = RoughBergomiParams(hurst=hurst, eta=eta, rho=float(rho), xi0=xi0)
    params.validate()

    model_vols = np.full_like(vols, np.sqrt(xi0), dtype=float)
    model_skews = np.sign(rho) * abs(rho) * eta * np.sqrt(xi0) * times ** (hurst - 0.5)
    vol_errors = model_vols - vols
    skew_errors = model_skews - skews
    objective = float(np.sum(vol_errors**2) + np.sum(skew_errors**2))
    index = pd.Index(times, name="maturity")
    return RoughBergomiCalibrationResult(
        params=params,
        objective_value=objective,
        atm_vol_rmse=float(np.sqrt(np.mean(vol_errors**2))),
        atm_skew_rmse=float(np.sqrt(np.mean(skew_errors**2))),
        atm_volatilities=pd.Series(vols, index=index, name="atm_volatility"),
        atm_skews=pd.Series(skews, index=index, name="atm_skew"),
        success=True,
        message="calibrated rough Bergomi ATM proxy",
    )


def calibrate_rough_bergomi_from_chain(
    option_chain: pd.DataFrame,
    rho: float = -0.55,
    local_strikes: int = 5,
) -> RoughBergomiCalibrationResult:
    """Extract ATM vol/skew by expiry from an option chain and calibrate the proxy."""
    required = {"spot", "strike", "maturity", "option_type", "implied_volatility"}
    missing = required - set(option_chain.columns)
    if missing:
        raise ValueError(f"option_chain is missing columns: {sorted(missing)}")
    if local_strikes < 3:
        raise ValueError("local_strikes must be at least 3")

    calls = option_chain[option_chain["option_type"] == "call"].copy()
    if calls.empty:
        raise ValueError("option_chain must contain call rows")

    maturities: list[float] = []
    atm_vols: list[float] = []
    atm_skews: list[float] = []
    for maturity, group in calls.groupby("maturity"):
        sorted_group = group.sort_values("strike")
        if len(sorted_group) < local_strikes:
            continue
        spot = float(sorted_group["spot"].iloc[0])
        rate = float(sorted_group["rate"].iloc[0]) if "rate" in sorted_group else 0.0
        dividend = float(sorted_group["dividend"].iloc[0]) if "dividend" in sorted_group else 0.0
        time = float(maturity)
        forward = spot * np.exp((rate - dividend) * time)
        log_moneyness = np.log(sorted_group["strike"].to_numpy(dtype=float) / forward)
        vols = sorted_group["implied_volatility"].to_numpy(dtype=float)
        if np.any(vols <= 0) or len(np.unique(log_moneyness)) < 2:
            continue

        atm_vol = float(np.interp(0.0, log_moneyness, vols))
        nearest = np.argsort(np.abs(log_moneyness))[: min(local_strikes, len(log_moneyness))]
        local_x = log_moneyness[nearest]
        local_y = vols[nearest]
        order = np.argsort(local_x)
        slope = float(np.polyfit(local_x[order], local_y[order], deg=1)[0])
        if slope == 0.0:
            continue
        maturities.append(time)
        atm_vols.append(atm_vol)
        atm_skews.append(slope)

    if len(maturities) < 2:
        raise ValueError("option_chain must provide at least two maturities with non-zero ATM skew")
    return calibrate_rough_bergomi_atm(np.asarray(maturities), np.asarray(atm_vols), np.asarray(atm_skews), rho=rho)
