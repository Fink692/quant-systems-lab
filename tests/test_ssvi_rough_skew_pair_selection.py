import numpy as np

from quantlab.data.synthetic import synthetic_cointegrated_prices
from quantlab.options.ssvi import SSVIParams, check_ssvi_no_arbitrage, ssvi_implied_volatility, ssvi_surface, ssvi_total_variance
from quantlab.rough_vol.skew import fit_atm_skew_power_law
from quantlab.stat_arb.network import pairwise_cointegration_network
from quantlab.stat_arb.selection import candidate_spread_weights, rank_cointegrated_pairs
from quantlab.workflows.demo_suite import run_full_demo


def test_ssvi_total_variance_surface_and_arbitrage_checks():
    params = SSVIParams(rho=-0.4, eta=0.45, gamma=0.25)
    maturities = np.array([0.25, 0.5, 1.0, 2.0])
    theta = 0.04 * maturities
    log_moneyness = np.linspace(-0.3, 0.3, 7)
    surface = ssvi_surface(maturities, log_moneyness, theta, params)
    check = check_ssvi_no_arbitrage(theta, params)

    assert surface.shape == (4, 7)
    assert check.passes
    assert abs(ssvi_total_variance(0.0, theta[2], params).item() - theta[2]) < 1e-12
    assert abs(ssvi_implied_volatility(0.0, maturities[2], theta[2], params).item() - 0.2) < 1e-12


def test_atm_skew_power_law_recovers_hurst_exponent():
    maturities = np.array([0.1, 0.25, 0.5, 1.0, 2.0])
    true_hurst = 0.14
    skews = -0.42 * maturities ** (true_hurst - 0.5)
    fit = fit_atm_skew_power_law(maturities, skews)

    assert abs(fit.hurst - true_hurst) < 1e-12
    assert fit.skew_sign == -1.0
    assert fit.r_squared > 0.999999
    assert np.allclose(fit.predict_skew(maturities), skews)


def test_ranked_cointegration_pairs_produce_spread_weights():
    prices = synthetic_cointegrated_prices(periods=180, seed=21)
    network = pairwise_cointegration_network(prices, p_value_threshold=0.05)
    candidates = rank_cointegrated_pairs(prices, network, max_p_value=0.05, top_n=3)
    weights = candidate_spread_weights(candidates[0])

    assert len(candidates) >= 1
    assert candidates[0].score >= candidates[-1].score
    assert abs(weights.abs().sum() - 1.0) < 1e-12
    assert candidates[0].dependent in weights.index
    assert candidates[0].independent in weights.index


def test_full_demo_exposes_ssvi_rough_skew_and_pair_selection_metrics():
    result = run_full_demo(seed=19).as_dict()
    assert result["options"]["ssvi_no_arbitrage_passes"]
    assert result["options"]["ssvi_mean_volatility"] > 0.0
    assert abs(result["rough_vol"]["atm_skew_power_law_hurst"] - 0.12) < 1e-12
    assert result["stat_arb"]["ranked_pair_count"] >= 1
    assert result["stat_arb"]["top_pair_gross_exposure"] == 1.0
