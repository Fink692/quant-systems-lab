from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BayesianReturnPosterior:
    posterior_mean: np.ndarray
    posterior_covariance: np.ndarray
    sample_mean: np.ndarray
    prior_mean: np.ndarray
    effective_sample_size: float


def bayesian_return_posterior(
    returns: np.ndarray,
    prior_mean: np.ndarray | None = None,
    prior_strength: float = 20.0,
) -> BayesianReturnPosterior:
    """Shrink sample expected returns toward a prior mean."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("returns must be a two-dimensional matrix with at least two observations")
    if prior_strength < 0:
        raise ValueError("prior_strength must be non-negative")
    sample_mean = values.mean(axis=0)
    prior = np.zeros(values.shape[1]) if prior_mean is None else np.asarray(prior_mean, dtype=float)
    if prior.shape != sample_mean.shape:
        raise ValueError("prior_mean shape must match number of assets")
    n_obs = values.shape[0]
    posterior_mean = (n_obs * sample_mean + prior_strength * prior) / (n_obs + prior_strength)
    sample_covariance = np.cov(values, rowvar=False)
    posterior_covariance = sample_covariance * (n_obs - 1) / max(n_obs + prior_strength - 1, 1)
    return BayesianReturnPosterior(
        posterior_mean=posterior_mean,
        posterior_covariance=posterior_covariance,
        sample_mean=sample_mean,
        prior_mean=prior,
        effective_sample_size=float(n_obs + prior_strength),
    )


def bayesian_mean_variance_weights(
    returns: np.ndarray,
    prior_mean: np.ndarray | None = None,
    prior_strength: float = 20.0,
    risk_aversion: float = 5.0,
) -> np.ndarray:
    """Compute mean-variance weights from Bayesian-shrunk expected returns."""
    from quantlab.portfolio.optimization import mean_variance_weights

    posterior = bayesian_return_posterior(returns, prior_mean=prior_mean, prior_strength=prior_strength)
    covariance = posterior.posterior_covariance + np.eye(len(posterior.posterior_mean)) * 1e-10
    return mean_variance_weights(posterior.posterior_mean, covariance, risk_aversion=risk_aversion)
