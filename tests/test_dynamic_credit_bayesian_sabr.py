import numpy as np

from quantlab.credit.survival import fit_exponential_hazard
from quantlab.data.synthetic import synthetic_cointegrated_prices, synthetic_factor_panel, synthetic_option_chain
from quantlab.options.sabr_surface import calibrate_sabr_surface, sabr_surface_implied_volatility
from quantlab.portfolio.bayesian import bayesian_mean_variance_weights, bayesian_return_posterior
from quantlab.stat_arb.kalman import kalman_dynamic_hedge_ratio
from quantlab.workflows.demo_suite import run_full_demo


def test_sabr_surface_calibrates_each_expiry():
    chain = synthetic_option_chain()
    result = calibrate_sabr_surface(chain, beta=0.6)
    assert len(result.parameters) == chain["maturity"].nunique()
    assert result.mean_objective < 1e-8
    vol = sabr_surface_implied_volatility(1.0, 100.0, 100.0 * np.exp(0.03), result)
    assert vol > 0.0


def test_kalman_dynamic_hedge_ratio_tracks_cointegrated_pair():
    prices = synthetic_cointegrated_prices(periods=150, seed=21)
    result = kalman_dynamic_hedge_ratio(prices["PairB"], prices["PairA"])
    assert result.states.shape == (150, 2)
    assert result.spread.shape == (150,)
    assert np.isfinite(result.states["hedge_ratio"].iloc[-1])
    assert result.spread.std(ddof=1) < prices["PairB"].std(ddof=1)


def test_exponential_hazard_mle_with_censoring():
    durations = np.array([1.0, 2.0, 3.0, 4.0])
    observed = np.array([True, False, True, False])
    fit = fit_exponential_hazard(durations, observed)
    assert fit.hazard_rate == 0.2
    assert fit.defaults == 2
    assert 0.0 < fit.default_probability(5.0) < 1.0
    assert fit.survival_probability(0.0) == 1.0


def test_bayesian_return_posterior_shrinks_mean_and_weights_sum():
    panel = synthetic_factor_panel(periods=80, assets=4, factors=2, seed=14)
    values = panel.asset_returns.to_numpy()
    posterior = bayesian_return_posterior(values, prior_strength=100.0)
    assert np.linalg.norm(posterior.posterior_mean) < np.linalg.norm(posterior.sample_mean)
    assert posterior.posterior_covariance.shape == (4, 4)
    weights = bayesian_mean_variance_weights(values, prior_strength=100.0, risk_aversion=5.0)
    assert abs(weights.sum() - 1.0) < 1e-8
    assert np.all(weights >= -1e-10)


def test_full_demo_exposes_dynamic_credit_and_bayesian_metrics():
    result = run_full_demo(seed=10).as_dict()
    assert result["options"]["sabr_surface_mean_objective"] >= 0.0
    assert np.isfinite(result["stat_arb"]["kalman_last_hedge_ratio"])
    assert result["credit"]["fitted_hazard_rate"] > 0.0
    assert result["portfolio"]["bayesian_first_weight"] >= 0.0
