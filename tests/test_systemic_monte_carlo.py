import numpy as np
import pytest

from quantlab.systemic.monte_carlo import simulate_systemic_monte_carlo
from quantlab.workflows.demo_suite import run_full_demo


def test_systemic_monte_carlo_outputs_tail_risk_and_default_probabilities():
    result = simulate_systemic_monte_carlo(
        holdings=np.array([[100.0, 0.0], [0.0, 100.0], [80.0, 60.0]]),
        capital=np.array([20.0, 20.0, 35.0]),
        mean_returns=np.array([-0.05, -0.03]),
        covariance=np.array([[0.2**2, 0.2 * 0.15 * 0.4], [0.2 * 0.15 * 0.4, 0.15**2]]),
        simulations=2_000,
        confidence=0.95,
        institution_names=["BankA", "BankB", "BankC"],
        asset_names=["equity", "credit"],
        seed=4,
    )

    assert result.asset_returns.shape == (2_000, 2)
    assert result.institution_equity.shape == (2_000, 3)
    assert result.shortfalls.shape == (2_000, 3)
    assert list(result.default_probabilities.index) == ["BankA", "BankB", "BankC"]
    assert 0.0 <= result.max_default_probability <= 1.0
    assert result.expected_shortfall >= result.value_at_risk

    tail = result.shortfalls[result.total_shortfall >= result.value_at_risk]
    assert np.isclose(result.tail_shortfall_contribution.sum(), tail.sum(axis=1).mean())


def test_systemic_monte_carlo_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="positive semidefinite"):
        simulate_systemic_monte_carlo(
            holdings=np.array([[100.0, 50.0]]),
            capital=np.array([20.0]),
            mean_returns=np.array([0.0, 0.0]),
            covariance=np.array([[1.0, 2.0], [2.0, 1.0]]),
        )


def test_full_demo_exposes_systemic_monte_carlo_metrics():
    result = run_full_demo(seed=27).as_dict()

    assert result["systemic"]["monte_carlo_expected_shortfall"] >= result["systemic"]["monte_carlo_value_at_risk"]
    assert 0.0 <= result["systemic"]["monte_carlo_max_default_probability"] <= 1.0
