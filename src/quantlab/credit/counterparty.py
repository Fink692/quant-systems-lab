from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.credit.curve import HazardCurve


@dataclass(frozen=True)
class ExposureProfile:
    times: np.ndarray
    expected_exposure: np.ndarray
    expected_positive_exposure: np.ndarray
    potential_future_exposure: np.ndarray

    @property
    def peak_pfe(self) -> float:
        return float(np.max(self.potential_future_exposure))

    @property
    def average_epe(self) -> float:
        return float(np.mean(self.expected_positive_exposure))


@dataclass(frozen=True)
class CVAResult:
    cva: float
    contributions: pd.DataFrame
    loss_given_default: float


def exposure_profile(
    times: np.ndarray,
    exposure_paths: np.ndarray,
    pfe_quantile: float = 0.95,
) -> ExposureProfile:
    """Build expected exposure, EPE, and PFE from mark-to-market exposure paths."""
    times = np.asarray(times, dtype=float)
    paths = np.asarray(exposure_paths, dtype=float)
    if times.ndim != 1 or paths.ndim != 2 or paths.shape[1] != len(times):
        raise ValueError("exposure_paths must be scenario-by-time and match times")
    if len(times) == 0 or np.any(times < 0) or np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing and non-negative")
    if not 0 < pfe_quantile < 1:
        raise ValueError("pfe_quantile must be in (0, 1)")
    positive = np.maximum(paths, 0.0)
    return ExposureProfile(
        times=times,
        expected_exposure=np.mean(paths, axis=0),
        expected_positive_exposure=np.mean(positive, axis=0),
        potential_future_exposure=np.quantile(positive, pfe_quantile, axis=0),
    )


def netting_set_exposure_profile(
    times: np.ndarray,
    trade_exposure_paths: np.ndarray,
    pfe_quantile: float = 0.95,
) -> ExposureProfile:
    """Aggregate trade-level exposure paths into a netting set profile."""
    paths = np.asarray(trade_exposure_paths, dtype=float)
    if paths.ndim != 3:
        raise ValueError("trade_exposure_paths must be scenario-by-trade-by-time")
    netted = paths.sum(axis=1)
    return exposure_profile(times, netted, pfe_quantile=pfe_quantile)


def unilateral_cva(
    profile: ExposureProfile,
    hazard_curve: HazardCurve,
    rate: float,
    recovery_rate: float | None = None,
) -> CVAResult:
    """Compute unilateral CVA from EPE, marginal default probabilities, and LGD."""
    if rate < 0:
        raise ValueError("rate must be non-negative")
    recovery = hazard_curve.recovery_rate if recovery_rate is None else float(recovery_rate)
    if not 0 <= recovery <= 1:
        raise ValueError("recovery_rate must be in [0, 1]")
    previous_times = np.r_[0.0, profile.times[:-1]]
    survival_start = np.array([hazard_curve.survival(t) for t in previous_times])
    survival_end = np.array([hazard_curve.survival(t) for t in profile.times])
    marginal_pd = survival_start - survival_end
    discounts = np.exp(-rate * profile.times)
    lgd = 1.0 - recovery
    contribution = lgd * discounts * profile.expected_positive_exposure * marginal_pd
    frame = pd.DataFrame(
        {
            "time": profile.times,
            "expected_positive_exposure": profile.expected_positive_exposure,
            "discount_factor": discounts,
            "marginal_default_probability": marginal_pd,
            "cva_contribution": contribution,
        }
    )
    return CVAResult(cva=float(np.sum(contribution)), contributions=frame, loss_given_default=float(lgd))


def wrong_way_adjusted_profile(profile: ExposureProfile, credit_factor: np.ndarray, beta: float = 0.25) -> ExposureProfile:
    """Apply a simple wrong-way-risk multiplier to EPE/PFE using a credit stress factor."""
    factor = np.asarray(credit_factor, dtype=float)
    if factor.shape != profile.times.shape:
        raise ValueError("credit_factor shape must match profile times")
    if not np.isfinite(factor).all():
        raise ValueError("credit_factor must be finite")
    std = factor.std(ddof=1) if len(factor) > 1 else 0.0
    standardized = np.zeros_like(factor) if std == 0.0 else (factor - factor.mean()) / std
    multiplier = np.exp(beta * standardized)
    return ExposureProfile(
        times=profile.times.copy(),
        expected_exposure=profile.expected_exposure * multiplier,
        expected_positive_exposure=profile.expected_positive_exposure * multiplier,
        potential_future_exposure=profile.potential_future_exposure * multiplier,
    )
