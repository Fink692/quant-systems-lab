import numpy as np

from quantlab.credit.default_models import credit_spread_from_hazard, merton_default_probability, survival_probability
from quantlab.portfolio.optimization import cvar_weights, min_variance_weights, risk_parity_weights
from quantlab.systemic.contagion import eigenvalue_stability, simulate_contagion


def test_portfolio_optimizers_return_budgeted_weights():
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    returns = np.array([[0.01, -0.02], [-0.03, 0.01], [0.02, 0.015], [-0.01, -0.025]])
    for weights in [min_variance_weights(cov), risk_parity_weights(cov), cvar_weights(returns)]:
        assert np.all(weights >= -1e-10)
        assert abs(np.sum(weights) - 1.0) < 1e-8


def test_credit_models_are_bounded_and_monotone():
    low_vol = merton_default_probability(120.0, 100.0, 1.0, 0.03, 0.15)
    high_vol = merton_default_probability(120.0, 100.0, 1.0, 0.03, 0.35)
    assert 0.0 <= low_vol <= 1.0
    assert 0.0 <= high_vol <= 1.0
    assert high_vol > low_vol
    assert survival_probability(0.02, 5.0) < 1.0
    assert credit_spread_from_hazard(0.02, 0.4) == 0.012


def test_systemic_contagion_propagates_defaults():
    exposures = np.array([[0.0, 70.0, 0.0], [0.0, 0.0, 60.0], [0.0, 0.0, 0.0]])
    capital = np.array([40.0, 35.0, 50.0])
    result = simulate_contagion(exposures, capital, initial_defaults=[1], recovery_rate=0.0)
    assert result.defaulted.tolist() == [True, True, False]
    assert eigenvalue_stability(exposures, capital) >= 0.0
