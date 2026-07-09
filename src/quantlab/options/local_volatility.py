from __future__ import annotations

import numpy as np


def dupire_local_volatility(
    maturities: np.ndarray,
    strikes: np.ndarray,
    call_prices: np.ndarray,
    rate: float,
    dividend: float = 0.0,
    min_denominator: float = 1e-10,
) -> np.ndarray:
    """Estimate Dupire local volatility from a grid of arbitrage-free call prices."""
    maturities = np.asarray(maturities, dtype=float)
    strikes = np.asarray(strikes, dtype=float)
    prices = np.asarray(call_prices, dtype=float)
    if maturities.ndim != 1 or strikes.ndim != 1:
        raise ValueError("maturities and strikes must be one-dimensional")
    if prices.shape != (len(maturities), len(strikes)):
        raise ValueError("call_prices shape must be (len(maturities), len(strikes))")
    if len(maturities) < 3 or len(strikes) < 3:
        raise ValueError("at least three maturities and strikes are required")
    if np.any(np.diff(maturities) <= 0) or np.any(np.diff(strikes) <= 0):
        raise ValueError("maturities and strikes must be strictly increasing")

    d_c_dt = np.gradient(prices, maturities, axis=0, edge_order=2)
    d_c_dk = np.gradient(prices, strikes, axis=1, edge_order=2)
    d2_c_dk2 = np.gradient(d_c_dk, strikes, axis=1, edge_order=2)
    strike_grid = strikes[None, :]

    numerator = d_c_dt + (rate - dividend) * strike_grid * d_c_dk + dividend * prices
    denominator = 0.5 * strike_grid**2 * d2_c_dk2
    variance = np.full_like(prices, np.nan, dtype=float)
    valid = denominator > min_denominator
    variance[valid] = numerator[valid] / denominator[valid]
    variance[variance < 0.0] = np.nan
    return np.sqrt(variance)
