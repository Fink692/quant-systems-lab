from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


@dataclass(frozen=True)
class LogisticHazardFit:
    coefficients: pd.Series
    log_likelihood: float
    observations: int
    defaults: int

    def predict_hazard(self, covariates: pd.DataFrame) -> pd.Series:
        aligned = covariates.reindex(columns=[name for name in self.coefficients.index if name != "intercept"]).astype(
            float
        )
        if aligned.isna().any().any():
            raise ValueError("covariates must contain every fitted feature")
        linear = self.coefficients["intercept"] + aligned.to_numpy() @ self.coefficients.drop("intercept").to_numpy()
        return pd.Series(expit(linear), index=covariates.index, name="hazard_probability")

    def survival_probability(self, covariates: pd.DataFrame) -> float:
        hazards = self.predict_hazard(covariates).to_numpy()
        return float(np.prod(1.0 - hazards))

    def default_probability(self, covariates: pd.DataFrame) -> float:
        return float(1.0 - self.survival_probability(covariates))


def fit_logistic_hazard(
    covariates: pd.DataFrame,
    default_observed: np.ndarray | pd.Series,
    l2_penalty: float = 1e-4,
    sample_weights: np.ndarray | pd.Series | None = None,
) -> LogisticHazardFit:
    """Fit a discrete-time reduced-form default intensity model."""
    if covariates.empty:
        raise ValueError("covariates cannot be empty")
    if l2_penalty < 0:
        raise ValueError("l2_penalty must be non-negative")
    x_frame = covariates.astype(float)
    if x_frame.isna().any().any():
        raise ValueError("covariates cannot contain missing values")
    y = np.asarray(default_observed, dtype=float)
    if y.shape != (len(x_frame),) or not set(np.unique(y)).issubset({0.0, 1.0}):
        raise ValueError("default_observed must be binary and match covariate rows")
    weights = np.ones(len(y)) if sample_weights is None else np.asarray(sample_weights, dtype=float)
    if weights.shape != y.shape or np.any(weights <= 0):
        raise ValueError("sample_weights must be positive and match observations")

    x = np.column_stack([np.ones(len(x_frame)), x_frame.to_numpy()])

    def objective(beta: np.ndarray) -> float:
        linear = x @ beta
        log_likelihood = weights * (y * np.log(expit(linear) + 1e-15) + (1.0 - y) * np.log(1.0 - expit(linear) + 1e-15))
        penalty = 0.5 * l2_penalty * float(beta[1:] @ beta[1:])
        return float(-np.sum(log_likelihood) + penalty)

    result = minimize(objective, np.zeros(x.shape[1]), method="BFGS")
    if not result.success:
        raise RuntimeError(result.message)
    coefficients = pd.Series(result.x, index=["intercept", *list(x_frame.columns)], name="coefficient")
    return LogisticHazardFit(
        coefficients=coefficients,
        log_likelihood=float(-objective(result.x)),
        observations=int(len(y)),
        defaults=int(y.sum()),
    )
