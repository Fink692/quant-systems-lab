from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ArbitrageViolation:
    kind: str
    maturity_index: int | None
    strike_index: int
    amount: float
    message: str


def detect_call_price_bounds(
    maturities: np.ndarray,
    strikes: np.ndarray,
    call_prices: np.ndarray,
    spot: float | None = None,
    rate: float = 0.0,
    dividend: float = 0.0,
    tolerance: float = 1e-8,
) -> list[ArbitrageViolation]:
    """Detect negative prices and, when spot is supplied, discounted call bounds."""
    maturities, strikes, prices = _validate_surface(maturities, strikes, call_prices)
    if spot is not None and spot <= 0:
        raise ValueError("spot must be positive when supplied")
    violations: list[ArbitrageViolation] = []
    for t_idx, maturity in enumerate(maturities):
        discounted_spot = None if spot is None else spot * np.exp(-dividend * maturity)
        for k_idx, strike in enumerate(strikes):
            price = prices[t_idx, k_idx]
            lower_bound = 0.0
            upper_bound = np.inf
            if discounted_spot is not None:
                discounted_strike = strike * np.exp(-rate * maturity)
                lower_bound = max(discounted_spot - discounted_strike, 0.0)
                upper_bound = discounted_spot
            if price < lower_bound - tolerance:
                violations.append(
                    ArbitrageViolation(
                        kind="bounds",
                        maturity_index=t_idx,
                        strike_index=k_idx,
                        amount=float(lower_bound - price),
                        message="Call price is below its discounted intrinsic lower bound.",
                    )
                )
            if price > upper_bound + tolerance:
                violations.append(
                    ArbitrageViolation(
                        kind="bounds",
                        maturity_index=t_idx,
                        strike_index=k_idx,
                        amount=float(price - upper_bound),
                        message="Call price is above its discounted spot upper bound.",
                    )
                )
    return violations


def detect_calendar_arbitrage(
    maturities: np.ndarray,
    strikes: np.ndarray,
    call_prices: np.ndarray,
    tolerance: float = 1e-8,
) -> list[ArbitrageViolation]:
    """Detect call prices decreasing with maturity for the same strike."""
    _maturities, _strikes, call_prices = _validate_surface(maturities, strikes, call_prices)
    violations: list[ArbitrageViolation] = []
    for t_idx in range(len(_maturities) - 1):
        decreases = call_prices[t_idx, :] - call_prices[t_idx + 1, :]
        for k_idx, amount in enumerate(decreases):
            if amount > tolerance:
                violations.append(
                    ArbitrageViolation(
                        kind="calendar",
                        maturity_index=t_idx,
                        strike_index=k_idx,
                        amount=float(amount),
                        message="Call price decreases as maturity increases.",
                    )
                )
    return violations


def detect_vertical_spread_arbitrage(
    maturities: np.ndarray,
    strikes: np.ndarray,
    call_prices: np.ndarray,
    rate: float = 0.0,
    tolerance: float = 1e-8,
) -> list[ArbitrageViolation]:
    """Detect adjacent vertical-spread violations across strikes."""
    maturities, strikes, prices = _validate_surface(maturities, strikes, call_prices)
    violations: list[ArbitrageViolation] = []
    for t_idx, maturity in enumerate(maturities):
        discount = np.exp(-rate * maturity)
        for k_idx in range(len(strikes) - 1):
            spread = prices[t_idx, k_idx] - prices[t_idx, k_idx + 1]
            max_spread = discount * (strikes[k_idx + 1] - strikes[k_idx])
            if spread < -tolerance:
                violations.append(
                    ArbitrageViolation(
                        kind="vertical",
                        maturity_index=t_idx,
                        strike_index=k_idx,
                        amount=float(-spread),
                        message="Call price increases with strike.",
                    )
                )
            if spread > max_spread + tolerance:
                violations.append(
                    ArbitrageViolation(
                        kind="vertical",
                        maturity_index=t_idx,
                        strike_index=k_idx,
                        amount=float(spread - max_spread),
                        message="Call vertical spread is wider than the discounted strike interval.",
                    )
                )
    return violations


def detect_butterfly_arbitrage(
    maturities: np.ndarray,
    strikes: np.ndarray,
    call_prices: np.ndarray,
    tolerance: float = 1e-8,
) -> list[ArbitrageViolation]:
    """Detect discrete convexity violations across strikes."""
    maturities, strikes, call_prices = _validate_surface(maturities, strikes, call_prices)
    violations: list[ArbitrageViolation] = []
    for t_idx in range(len(maturities)):
        for k_idx in range(1, len(strikes) - 1):
            left_width = strikes[k_idx] - strikes[k_idx - 1]
            right_width = strikes[k_idx + 1] - strikes[k_idx]
            left_slope = (call_prices[t_idx, k_idx] - call_prices[t_idx, k_idx - 1]) / left_width
            right_slope = (call_prices[t_idx, k_idx + 1] - call_prices[t_idx, k_idx]) / right_width
            convexity = right_slope - left_slope
            if convexity < -tolerance:
                violations.append(
                    ArbitrageViolation(
                        kind="butterfly",
                        maturity_index=t_idx,
                        strike_index=k_idx,
                        amount=float(-convexity),
                        message="Call price is not convex in strike.",
                    )
                )
    return violations


def detect_surface_arbitrage(
    maturities: np.ndarray,
    strikes: np.ndarray,
    call_prices: np.ndarray,
    spot: float | None = None,
    rate: float = 0.0,
    dividend: float = 0.0,
    tolerance: float = 1e-8,
) -> list[ArbitrageViolation]:
    """Run static no-arbitrage checks for a call price surface."""
    return (
        detect_call_price_bounds(maturities, strikes, call_prices, spot, rate, dividend, tolerance)
        + detect_vertical_spread_arbitrage(maturities, strikes, call_prices, rate, tolerance)
        + detect_calendar_arbitrage(maturities, strikes, call_prices, tolerance)
        + detect_butterfly_arbitrage(maturities, strikes, call_prices, tolerance)
    )


def _validate_surface(maturities: np.ndarray, strikes: np.ndarray, call_prices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    maturities = np.asarray(maturities, dtype=float)
    strikes = np.asarray(strikes, dtype=float)
    call_prices = np.asarray(call_prices, dtype=float)
    if maturities.ndim != 1 or strikes.ndim != 1:
        raise ValueError("maturities and strikes must be one-dimensional")
    if call_prices.shape != (len(maturities), len(strikes)):
        raise ValueError("call_prices shape must be (len(maturities), len(strikes))")
    if np.any(np.diff(maturities) <= 0):
        raise ValueError("maturities must be strictly increasing")
    if np.any(np.diff(strikes) <= 0):
        raise ValueError("strikes must be strictly increasing")
    if not np.all(np.isfinite(call_prices)):
        raise ValueError("call_prices must be finite")
    return maturities, strikes, call_prices
