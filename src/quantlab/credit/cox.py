from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class CoxPHFit:
    coefficients: pd.Series
    baseline_cumulative_hazard: pd.Series
    log_likelihood: float
    observations: int
    defaults: int

    def hazard_ratio(self, covariates: pd.DataFrame) -> pd.Series:
        aligned = covariates.reindex(columns=self.coefficients.index).astype(float)
        if aligned.isna().any().any():
            raise ValueError("covariates must contain every fitted feature")
        ratios = np.exp(aligned.to_numpy() @ self.coefficients.to_numpy())
        return pd.Series(ratios, index=covariates.index, name="hazard_ratio")

    def survival_probability(self, covariates: pd.DataFrame, maturity: float) -> pd.Series:
        if maturity < 0:
            raise ValueError("maturity must be non-negative")
        baseline = _step_value(self.baseline_cumulative_hazard, maturity)
        survival = np.exp(-baseline * self.hazard_ratio(covariates).to_numpy())
        return pd.Series(survival, index=covariates.index, name="survival_probability")

    def default_probability(self, covariates: pd.DataFrame, maturity: float) -> pd.Series:
        return 1.0 - self.survival_probability(covariates, maturity)


def fit_cox_ph(
    covariates: pd.DataFrame,
    durations: np.ndarray | pd.Series,
    default_observed: np.ndarray | pd.Series,
    l2_penalty: float = 1e-4,
) -> CoxPHFit:
    """Fit a Cox proportional-hazards default model by partial likelihood."""
    if covariates.empty:
        raise ValueError("covariates cannot be empty")
    if l2_penalty < 0:
        raise ValueError("l2_penalty must be non-negative")
    x_frame = covariates.astype(float)
    if x_frame.isna().any().any():
        raise ValueError("covariates cannot contain missing values")
    times = np.asarray(durations, dtype=float)
    observed = np.asarray(default_observed, dtype=bool)
    if times.shape != (len(x_frame),) or observed.shape != times.shape:
        raise ValueError("durations and default_observed must match covariate rows")
    if np.any(times <= 0):
        raise ValueError("durations must be positive")
    if int(observed.sum()) == 0:
        raise ValueError("at least one default observation is required")

    x = x_frame.to_numpy()

    def objective(beta: np.ndarray) -> float:
        likelihood = _partial_log_likelihood(beta, x, times, observed)
        penalty = 0.5 * l2_penalty * float(beta @ beta)
        return float(-likelihood + penalty)

    result = minimize(objective, np.zeros(x.shape[1], dtype=float), method="BFGS")
    if not result.success:
        raise RuntimeError(result.message)
    beta = np.asarray(result.x, dtype=float)
    coefficients = pd.Series(beta, index=x_frame.columns, name="coefficient")
    baseline = _breslow_baseline_cumulative_hazard(beta, x, times, observed)
    return CoxPHFit(
        coefficients=coefficients,
        baseline_cumulative_hazard=baseline,
        log_likelihood=float(_partial_log_likelihood(beta, x, times, observed)),
        observations=int(len(times)),
        defaults=int(observed.sum()),
    )


def _partial_log_likelihood(beta: np.ndarray, x: np.ndarray, times: np.ndarray, observed: np.ndarray) -> float:
    scores = x @ beta
    log_likelihood = 0.0
    for event_time in np.unique(times[observed]):
        event_mask = (times == event_time) & observed
        risk_mask = times >= event_time
        max_score = float(np.max(scores[risk_mask]))
        log_risk_sum = max_score + np.log(np.sum(np.exp(scores[risk_mask] - max_score)))
        log_likelihood += float(np.sum(scores[event_mask]) - event_mask.sum() * log_risk_sum)
    return log_likelihood


def _breslow_baseline_cumulative_hazard(
    beta: np.ndarray,
    x: np.ndarray,
    times: np.ndarray,
    observed: np.ndarray,
) -> pd.Series:
    scores = np.exp(x @ beta)
    cumulative = 0.0
    rows: list[tuple[float, float]] = []
    for event_time in np.unique(times[observed]):
        event_count = int(np.sum((times == event_time) & observed))
        risk_sum = float(np.sum(scores[times >= event_time]))
        cumulative += event_count / risk_sum
        rows.append((float(event_time), float(cumulative)))
    index = pd.Index([row[0] for row in rows], name="duration")
    return pd.Series([row[1] for row in rows], index=index, name="baseline_cumulative_hazard")


def _step_value(series: pd.Series, maturity: float) -> float:
    if series.empty or maturity < float(series.index[0]):
        return 0.0
    eligible = series[series.index <= maturity]
    return float(eligible.iloc[-1]) if len(eligible) else 0.0
