import numpy as np
import pytest

from quantlab.portfolio.optimization import mean_variance_weights
from quantlab.portfolio.robust import ellipsoidal_robust_mean_variance_weights
from quantlab.workflows.demo_suite import run_full_demo


def test_ellipsoidal_robust_optimizer_matches_nominal_when_radius_is_zero():
    mu = np.array([0.04, 0.06, 0.05])
    covariance = np.diag([0.03, 0.04, 0.05])
    uncertainty = np.diag([0.001, 0.001, 0.001])

    robust = ellipsoidal_robust_mean_variance_weights(
        mu,
        covariance,
        uncertainty,
        uncertainty_radius=0.0,
        risk_aversion=4.0,
    )
    nominal = mean_variance_weights(mu, covariance, risk_aversion=4.0)

    assert np.allclose(robust.weights, nominal, atol=1e-3)
    assert robust.uncertainty_penalty == 0.0
    assert robust.worst_case_return == robust.nominal_return
    assert abs(robust.weights.sum() - 1.0) < 1e-10


def test_ellipsoidal_robust_optimizer_penalizes_uncertain_high_return_asset():
    mu = np.array([0.08, 0.04])
    covariance = np.diag([0.02, 0.02])
    mean_uncertainty = np.diag([0.25, 1e-6])

    nominal = mean_variance_weights(mu, covariance, risk_aversion=2.0)
    robust = ellipsoidal_robust_mean_variance_weights(
        mu,
        covariance,
        mean_uncertainty,
        uncertainty_radius=0.4,
        risk_aversion=2.0,
    )

    assert robust.weights[0] < nominal[0]
    assert robust.uncertainty_penalty > 0.0
    assert robust.worst_case_return <= robust.nominal_return


def test_ellipsoidal_robust_optimizer_rejects_bad_uncertainty_matrix():
    with pytest.raises(ValueError, match="positive semidefinite"):
        ellipsoidal_robust_mean_variance_weights(
            np.array([0.04, 0.05]),
            np.eye(2),
            np.array([[1.0, 2.0], [2.0, 1.0]]),
        )


def test_full_demo_exposes_ellipsoidal_robust_metrics():
    result = run_full_demo(seed=31).as_dict()

    assert 0.0 <= result["portfolio"]["ellipsoid_robust_first_weight"] <= 1.0
    assert result["portfolio"]["ellipsoid_uncertainty_penalty"] >= 0.0
    assert result["portfolio"]["ellipsoid_worst_case_return"] <= 1.0
