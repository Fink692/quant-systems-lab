from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from quantlab.options.black_scholes import black_scholes_price
from quantlab.options.local_volatility import dupire_local_volatility
from quantlab.options.surface_arbitrage import ArbitrageViolation, detect_surface_arbitrage

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class VolatilitySurface:
    maturities: np.ndarray
    strikes: np.ndarray
    implied_volatilities: np.ndarray
    spot: float
    rate: float
    dividend: float = 0.0

    def __post_init__(self) -> None:
        maturities = np.asarray(self.maturities, dtype=float)
        strikes = np.asarray(self.strikes, dtype=float)
        vols = np.asarray(self.implied_volatilities, dtype=float)
        if maturities.ndim != 1 or strikes.ndim != 1:
            raise ValueError("maturities and strikes must be one-dimensional")
        if vols.shape != (len(maturities), len(strikes)):
            raise ValueError("implied_volatilities shape must be (len(maturities), len(strikes))")
        if np.any(np.diff(maturities) <= 0) or np.any(np.diff(strikes) <= 0):
            raise ValueError("maturities and strikes must be strictly increasing")
        if np.any(vols <= 0):
            raise ValueError("implied volatilities must be positive")
        if self.spot <= 0:
            raise ValueError("spot must be positive")

    def implied_volatility(self, maturity: float, strike: float) -> float:
        interpolator = RegularGridInterpolator(
            (self.maturities, self.strikes),
            self.implied_volatilities,
            bounds_error=False,
            fill_value=None,
        )
        return float(interpolator(np.array([[maturity, strike]], dtype=float))[0])

    def price(self, maturity: float, strike: float, option_type: OptionType = "call") -> float:
        vol = self.implied_volatility(maturity, strike)
        return black_scholes_price(self.spot, strike, maturity, self.rate, vol, self.dividend, option_type)

    def price_grid(self, option_type: OptionType = "call") -> np.ndarray:
        return np.array(
            [
                [
                    black_scholes_price(self.spot, strike, maturity, self.rate, vol, self.dividend, option_type)
                    for strike, vol in zip(self.strikes, row)
                ]
                for maturity, row in zip(self.maturities, self.implied_volatilities)
            ]
        )

    def local_volatility(self) -> np.ndarray:
        return dupire_local_volatility(self.maturities, self.strikes, self.price_grid("call"), self.rate, self.dividend)

    def arbitrage_violations(self) -> list[ArbitrageViolation]:
        return detect_surface_arbitrage(
            self.maturities,
            self.strikes,
            self.price_grid("call"),
            spot=self.spot,
            rate=self.rate,
            dividend=self.dividend,
        )


def build_volatility_surface_from_chain(option_chain: pd.DataFrame) -> VolatilitySurface:
    """Build a volatility surface from an option chain containing implied vols."""
    required = {"spot", "strike", "maturity", "rate", "dividend", "option_type", "implied_volatility"}
    missing = required - set(option_chain.columns)
    if missing:
        raise ValueError(f"option_chain is missing columns: {sorted(missing)}")
    calls = option_chain[option_chain["option_type"] == "call"]
    if calls.empty:
        raise ValueError("option_chain must contain call rows")
    pivot = calls.pivot_table(index="maturity", columns="strike", values="implied_volatility", aggfunc="mean")
    if pivot.isna().any().any():
        raise ValueError("option_chain must form a complete maturity/strike grid")
    return VolatilitySurface(
        maturities=pivot.index.to_numpy(dtype=float),
        strikes=pivot.columns.to_numpy(dtype=float),
        implied_volatilities=pivot.to_numpy(dtype=float),
        spot=float(calls["spot"].iloc[0]),
        rate=float(calls["rate"].iloc[0]),
        dividend=float(calls["dividend"].iloc[0]),
    )
