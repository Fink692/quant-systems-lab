from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.options.black_scholes import black_scholes_price
from quantlab.options.sabr import SABRParams, sabr_implied_volatility


@dataclass(frozen=True)
class SyntheticFactorPanel:
    asset_returns: pd.DataFrame
    factor_returns: pd.DataFrame
    true_exposures: pd.DataFrame
    prices: pd.DataFrame


def synthetic_option_chain(
    spot: float = 100.0,
    rate: float = 0.03,
    dividend: float = 0.0,
    maturities: np.ndarray | None = None,
    moneyness: np.ndarray | None = None,
    beta: float = 0.6,
) -> pd.DataFrame:
    """Create a smooth SABR-generated option chain for examples and calibration tests."""
    if spot <= 0:
        raise ValueError("spot must be positive")
    maturities = np.array([0.25, 0.5, 1.0, 2.0]) if maturities is None else np.asarray(maturities, dtype=float)
    moneyness = np.linspace(0.8, 1.2, 9) if moneyness is None else np.asarray(moneyness, dtype=float)
    if np.any(maturities <= 0) or np.any(moneyness <= 0):
        raise ValueError("maturities and moneyness values must be positive")

    rows: list[dict[str, float | str]] = []
    for maturity in maturities:
        forward = spot * np.exp((rate - dividend) * maturity)
        params = SABRParams(
            alpha=0.23 + 0.02 * np.sqrt(maturity),
            beta=beta,
            rho=-0.35 + 0.05 * min(maturity, 2.0),
            nu=0.55 + 0.05 * maturity,
        )
        for money in moneyness:
            strike = spot * money
            implied_vol = sabr_implied_volatility(forward, strike, maturity, params)
            for option_type in ("call", "put"):
                price = black_scholes_price(spot, strike, maturity, rate, implied_vol, dividend, option_type)
                rows.append(
                    {
                        "spot": float(spot),
                        "strike": float(strike),
                        "maturity": float(maturity),
                        "rate": float(rate),
                        "dividend": float(dividend),
                        "option_type": option_type,
                        "implied_volatility": float(implied_vol),
                        "price": float(price),
                    }
                )
    return pd.DataFrame(rows)


def synthetic_factor_panel(
    periods: int = 252,
    assets: int = 8,
    factors: int = 3,
    seed: int = 7,
) -> SyntheticFactorPanel:
    """Generate an equity return panel with known linear factor structure."""
    if min(periods, assets, factors) <= 0:
        raise ValueError("periods, assets, and factors must be positive")
    if factors > assets:
        raise ValueError("factors cannot exceed assets")
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=periods, freq="B")
    factor_names = [f"F{i + 1}" for i in range(factors)]
    asset_names = [f"Asset{i + 1}" for i in range(assets)]

    factor_values = rng.normal(0.0, 0.01, size=(periods, factors))
    exposures = rng.normal(0.7, 0.35, size=(assets, factors))
    specific = rng.normal(0.0, 0.006, size=(periods, assets))
    asset_values = factor_values @ exposures.T + specific
    prices = 100.0 * np.exp(np.cumsum(asset_values, axis=0))

    return SyntheticFactorPanel(
        asset_returns=pd.DataFrame(asset_values, index=dates, columns=asset_names),
        factor_returns=pd.DataFrame(factor_values, index=dates, columns=factor_names),
        true_exposures=pd.DataFrame(exposures, index=asset_names, columns=factor_names),
        prices=pd.DataFrame(prices, index=dates, columns=asset_names),
    )


def synthetic_cointegrated_prices(periods: int = 252, seed: int = 11) -> pd.DataFrame:
    """Generate a small price panel with one cointegrated pair and one independent series."""
    if periods < 30:
        raise ValueError("periods must be at least 30")
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=periods, freq="B")
    common_trend = np.cumsum(rng.normal(0.0, 1.0, size=periods))
    asset_a = 50.0 + common_trend + rng.normal(0.0, 0.25, size=periods)
    asset_b = 5.0 + 1.2 * common_trend + rng.normal(0.0, 0.25, size=periods)
    asset_c = 80.0 + np.cumsum(rng.normal(0.0, 1.2, size=periods))
    return pd.DataFrame({"PairA": asset_a, "PairB": asset_b, "Diversifier": asset_c}, index=dates)


def synthetic_credit_spreads() -> pd.DataFrame:
    """Create a simple upward-sloping CDS-style spread curve."""
    maturities = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    spreads = np.array([0.006, 0.008, 0.011, 0.014, 0.018, 0.021, 0.024])
    return pd.DataFrame({"maturity": maturities, "credit_spread": spreads})


def synthetic_exposure_network(institutions: int = 5, seed: int = 19) -> tuple[pd.DataFrame, pd.Series]:
    """Generate a directed interbank exposure network and capital vector."""
    if institutions < 2:
        raise ValueError("institutions must be at least 2")
    rng = np.random.default_rng(seed)
    names = [f"Bank{i + 1}" for i in range(institutions)]
    exposures = rng.gamma(shape=1.5, scale=12.0, size=(institutions, institutions))
    np.fill_diagonal(exposures, 0.0)
    capital = rng.uniform(35.0, 75.0, size=institutions)
    return pd.DataFrame(exposures, index=names, columns=names), pd.Series(capital, index=names, name="capital")
