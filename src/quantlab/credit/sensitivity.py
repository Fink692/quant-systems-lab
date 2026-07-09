from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantlab.credit.curve import HazardCurve
from quantlab.credit.pricing import risky_coupon_bond_price, risky_zero_coupon_price


@dataclass(frozen=True)
class SpreadSensitivityResult:
    base_price: float
    bumped_up_price: float
    bumped_down_price: float
    spread_duration: float
    spread_dv01: float


def shift_hazard_curve_by_spread(hazard_curve: HazardCurve, spread_shift_bps: float) -> HazardCurve:
    """Shift hazard rates by an equivalent parallel credit-spread move."""
    spread_shift = spread_shift_bps / 10_000.0
    hazard_shift = spread_shift / (1.0 - hazard_curve.recovery_rate)
    shifted = np.maximum(hazard_curve.hazard_rates + hazard_shift, 0.0)
    return HazardCurve(
        maturities=hazard_curve.maturities.copy(),
        hazard_rates=shifted,
        recovery_rate=hazard_curve.recovery_rate,
    )


def zero_coupon_spread_sensitivity(
    maturity: float,
    rate: float,
    hazard_curve: HazardCurve,
    spread_bump_bps: float = 1.0,
    face_value: float = 1.0,
) -> SpreadSensitivityResult:
    """Estimate spread duration and DV01 for a risky zero-coupon bond."""
    if spread_bump_bps <= 0:
        raise ValueError("spread_bump_bps must be positive")
    base = risky_zero_coupon_price(maturity, rate, hazard_curve, face_value=face_value)
    up = risky_zero_coupon_price(maturity, rate, shift_hazard_curve_by_spread(hazard_curve, spread_bump_bps), face_value=face_value)
    down = risky_zero_coupon_price(maturity, rate, shift_hazard_curve_by_spread(hazard_curve, -spread_bump_bps), face_value=face_value)
    return _spread_sensitivity(base, up, down, spread_bump_bps)


def coupon_bond_spread_sensitivity(
    coupon_rate: float,
    maturity: float,
    rate: float,
    hazard_curve: HazardCurve,
    payment_frequency: int = 2,
    spread_bump_bps: float = 1.0,
    face_value: float = 1.0,
) -> SpreadSensitivityResult:
    """Estimate spread duration and DV01 for a risky coupon bond."""
    if spread_bump_bps <= 0:
        raise ValueError("spread_bump_bps must be positive")
    base = risky_coupon_bond_price(coupon_rate, maturity, rate, hazard_curve, payment_frequency, face_value)
    up = risky_coupon_bond_price(
        coupon_rate,
        maturity,
        rate,
        shift_hazard_curve_by_spread(hazard_curve, spread_bump_bps),
        payment_frequency,
        face_value,
    )
    down = risky_coupon_bond_price(
        coupon_rate,
        maturity,
        rate,
        shift_hazard_curve_by_spread(hazard_curve, -spread_bump_bps),
        payment_frequency,
        face_value,
    )
    return _spread_sensitivity(base, up, down, spread_bump_bps)


def _spread_sensitivity(base: float, up: float, down: float, spread_bump_bps: float) -> SpreadSensitivityResult:
    bump_decimal = spread_bump_bps / 10_000.0
    price_derivative = (up - down) / (2.0 * bump_decimal)
    spread_duration = 0.0 if base == 0.0 else -price_derivative / base
    spread_dv01 = (down - up) / (2.0 * spread_bump_bps)
    return SpreadSensitivityResult(
        base_price=float(base),
        bumped_up_price=float(up),
        bumped_down_price=float(down),
        spread_duration=float(spread_duration),
        spread_dv01=float(spread_dv01),
    )
