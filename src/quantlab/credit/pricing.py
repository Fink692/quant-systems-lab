from __future__ import annotations

import numpy as np

from quantlab.credit.curve import HazardCurve


def risky_zero_coupon_price(
    maturity: float,
    rate: float,
    hazard_curve: HazardCurve,
    recovery_rate: float | None = None,
    face_value: float = 1.0,
) -> float:
    """Price a risky zero-coupon bond with recovery of par at maturity."""
    if maturity < 0 or face_value <= 0:
        raise ValueError("maturity must be non-negative and face_value positive")
    recovery = hazard_curve.recovery_rate if recovery_rate is None else recovery_rate
    if not 0 <= recovery <= 1:
        raise ValueError("recovery_rate must be in [0, 1]")
    survival = hazard_curve.survival(maturity)
    discount = np.exp(-rate * maturity)
    return float(face_value * discount * (survival + recovery * (1.0 - survival)))


def risky_coupon_bond_price(
    coupon_rate: float,
    maturity: float,
    rate: float,
    hazard_curve: HazardCurve,
    payment_frequency: int = 2,
    face_value: float = 1.0,
    recovery_rate: float | None = None,
) -> float:
    """Price a risky fixed-coupon bond with survival-weighted coupons."""
    if coupon_rate < 0 or maturity <= 0 or payment_frequency <= 0 or face_value <= 0:
        raise ValueError("invalid coupon bond inputs")
    recovery = hazard_curve.recovery_rate if recovery_rate is None else recovery_rate
    if not 0 <= recovery <= 1:
        raise ValueError("recovery_rate must be in [0, 1]")
    payment_times = np.arange(1, int(np.ceil(maturity * payment_frequency)) + 1) / payment_frequency
    payment_times = payment_times[payment_times <= maturity + 1e-12]
    if payment_times[-1] < maturity:
        payment_times = np.append(payment_times, maturity)
    previous_times = np.r_[0.0, payment_times[:-1]]
    accruals = payment_times - previous_times
    discounts = np.exp(-rate * payment_times)
    survivals = np.array([hazard_curve.survival(t) for t in payment_times])
    coupon_leg = float(np.sum(face_value * coupon_rate * accruals * discounts * survivals))
    principal_leg = risky_zero_coupon_price(maturity, rate, hazard_curve, recovery, face_value)
    return coupon_leg + principal_leg


def cds_par_spread(
    maturity: float,
    rate: float,
    hazard_curve: HazardCurve,
    payment_frequency: int = 4,
) -> float:
    """Compute a discrete premium-leg CDS par spread."""
    if maturity <= 0 or payment_frequency <= 0:
        raise ValueError("maturity and payment_frequency must be positive")
    payment_times = np.arange(1, int(np.ceil(maturity * payment_frequency)) + 1) / payment_frequency
    payment_times = payment_times[payment_times <= maturity + 1e-12]
    if payment_times[-1] < maturity:
        payment_times = np.append(payment_times, maturity)
    previous_times = np.r_[0.0, payment_times[:-1]]
    survival = np.array([hazard_curve.survival(t) for t in payment_times])
    previous_survival = np.array([hazard_curve.survival(t) for t in previous_times])
    discounts = np.exp(-rate * payment_times)
    accruals = payment_times - previous_times
    premium_leg = np.sum(discounts * survival * accruals)
    protection_leg = np.sum(discounts * (previous_survival - survival) * (1.0 - hazard_curve.recovery_rate))
    if premium_leg <= 0:
        raise ValueError("premium leg is zero")
    return float(protection_leg / premium_leg)
