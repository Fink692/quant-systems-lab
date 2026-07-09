from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binom, chi2


@dataclass(frozen=True)
class VaRBacktestResult:
    observations: int
    exceptions: int
    exception_rate: float
    kupiec_statistic: float
    kupiec_p_value: float


@dataclass(frozen=True)
class ChristoffersenBacktestResult:
    observations: int
    exceptions: int
    transition_counts: dict[str, int]
    independence_statistic: float
    independence_p_value: float
    conditional_coverage_statistic: float
    conditional_coverage_p_value: float


@dataclass(frozen=True)
class BaselTrafficLightResult:
    observations: int
    exceptions: int
    confidence: float
    green_threshold: int
    yellow_threshold: int
    zone: str


def kupiec_var_backtest(returns: np.ndarray, var_estimates: np.ndarray, confidence: float = 0.95) -> VaRBacktestResult:
    """Kupiec unconditional coverage test for VaR exceptions."""
    returns = np.asarray(returns, dtype=float)
    var_estimates = np.asarray(var_estimates, dtype=float)
    if returns.ndim != 1 or var_estimates.shape != returns.shape:
        raise ValueError("returns and var_estimates must be one-dimensional arrays with the same shape")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    losses = -returns
    exceptions = int(np.sum(losses > var_estimates))
    observations = len(returns)
    expected_probability = 1.0 - confidence
    observed_probability = exceptions / observations if observations else 0.0
    if exceptions == 0 or exceptions == observations:
        statistic = 0.0
        p_value = 1.0
    else:
        log_likelihood_null = (observations - exceptions) * np.log(1.0 - expected_probability) + exceptions * np.log(expected_probability)
        log_likelihood_alt = (observations - exceptions) * np.log(1.0 - observed_probability) + exceptions * np.log(observed_probability)
        statistic = -2.0 * (log_likelihood_null - log_likelihood_alt)
        p_value = 1.0 - chi2.cdf(statistic, df=1)
    return VaRBacktestResult(observations, exceptions, float(observed_probability), float(statistic), float(p_value))


def christoffersen_var_backtest(
    returns: np.ndarray,
    var_estimates: np.ndarray,
    confidence: float = 0.95,
) -> ChristoffersenBacktestResult:
    """Christoffersen independence and conditional-coverage tests for VaR exceptions."""
    returns, var_estimates = _validate_var_inputs(returns, var_estimates, confidence)
    exceptions = (-returns > var_estimates).astype(int)
    observations = int(len(exceptions))
    if observations < 2:
        raise ValueError("at least two observations are required")
    previous = exceptions[:-1]
    current = exceptions[1:]
    n00 = int(np.sum((previous == 0) & (current == 0)))
    n01 = int(np.sum((previous == 0) & (current == 1)))
    n10 = int(np.sum((previous == 1) & (current == 0)))
    n11 = int(np.sum((previous == 1) & (current == 1)))

    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    pi0 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi1 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    ll_null = _binomial_log_likelihood(n01 + n11, n00 + n10, pi)
    ll_alt = _binomial_log_likelihood(n01, n00, pi0) + _binomial_log_likelihood(n11, n10, pi1)
    independence_statistic = max(-2.0 * (ll_null - ll_alt), 0.0)
    independence_p_value = 1.0 - chi2.cdf(independence_statistic, df=1)
    kupiec = kupiec_var_backtest(returns, var_estimates, confidence)
    conditional_statistic = kupiec.kupiec_statistic + independence_statistic
    conditional_p_value = 1.0 - chi2.cdf(conditional_statistic, df=2)
    return ChristoffersenBacktestResult(
        observations=observations,
        exceptions=int(exceptions.sum()),
        transition_counts={"n00": n00, "n01": n01, "n10": n10, "n11": n11},
        independence_statistic=float(independence_statistic),
        independence_p_value=float(independence_p_value),
        conditional_coverage_statistic=float(conditional_statistic),
        conditional_coverage_p_value=float(conditional_p_value),
    )


def basel_traffic_light(exceptions: int, observations: int = 250, confidence: float = 0.99) -> BaselTrafficLightResult:
    """Classify VaR exceptions into Basel-style green/yellow/red zones."""
    if observations <= 0 or exceptions < 0 or exceptions > observations:
        raise ValueError("invalid exception or observation count")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    exception_probability = 1.0 - confidence
    green_threshold = int(binom.ppf(0.95, observations, exception_probability))
    yellow_threshold = int(binom.ppf(0.9999, observations, exception_probability))
    if exceptions <= green_threshold:
        zone = "green"
    elif exceptions <= yellow_threshold:
        zone = "yellow"
    else:
        zone = "red"
    return BaselTrafficLightResult(
        observations=int(observations),
        exceptions=int(exceptions),
        confidence=float(confidence),
        green_threshold=green_threshold,
        yellow_threshold=yellow_threshold,
        zone=zone,
    )


def _validate_var_inputs(returns: np.ndarray, var_estimates: np.ndarray, confidence: float) -> tuple[np.ndarray, np.ndarray]:
    returns = np.asarray(returns, dtype=float)
    var_estimates = np.asarray(var_estimates, dtype=float)
    if returns.ndim != 1 or var_estimates.shape != returns.shape:
        raise ValueError("returns and var_estimates must be one-dimensional arrays with the same shape")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    return returns, var_estimates


def _binomial_log_likelihood(successes: int, failures: int, probability: float) -> float:
    probability = float(np.clip(probability, 1e-15, 1.0 - 1e-15))
    return float(successes * np.log(probability) + failures * np.log(1.0 - probability))
