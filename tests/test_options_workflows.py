import numpy as np

from quantlab.options.black_scholes import black_scholes_price
from quantlab.options.calibration import calibrate_sabr_smile
from quantlab.options.heston import HestonParams, heston_price
from quantlab.options.monte_carlo import black_scholes_monte_carlo_price, heston_monte_carlo_price
from quantlab.options.pde import black_scholes_finite_difference_price
from quantlab.options.sabr import SABRParams, sabr_implied_volatility


def test_sabr_smile_calibration_recovers_clean_surface():
    true = SABRParams(alpha=0.28, beta=0.6, rho=-0.35, nu=0.7)
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    vols = np.array([sabr_implied_volatility(100.0, strike, 1.5, true) for strike in strikes])
    result = calibrate_sabr_smile(100.0, 1.5, strikes, vols, beta=0.6, initial=(0.2, -0.1, 0.4))
    assert result.success
    assert result.objective_value < 1e-8
    assert abs(result.parameters["alpha"] - true.alpha) < 1e-3
    assert abs(result.parameters["rho"] - true.rho) < 1e-3
    assert abs(result.parameters["nu"] - true.nu) < 1e-3


def test_finite_difference_matches_black_scholes_baseline():
    analytic = black_scholes_price(100.0, 100.0, 1.0, 0.03, 0.2, option_type="call")
    pde = black_scholes_finite_difference_price(100.0, 100.0, 1.0, 0.03, 0.2, option_type="call")
    assert abs(pde - analytic) < 0.25


def test_monte_carlo_pricers_are_finite_and_near_baseline():
    analytic = black_scholes_price(100.0, 100.0, 1.0, 0.03, 0.2, option_type="call")
    mc = black_scholes_monte_carlo_price(100.0, 100.0, 1.0, 0.03, 0.2, paths=40_000, seed=123)
    assert abs(mc - analytic) < 0.75

    heston = heston_monte_carlo_price(
        100.0,
        100.0,
        1.0,
        0.03,
        HestonParams(kappa=2.0, theta=0.04, sigma=0.35, rho=-0.5, v0=0.04),
        paths=1_000,
        steps=24,
        seed=321,
    )
    assert np.isfinite(heston)
    assert heston > 0.0


def test_heston_monte_carlo_is_near_fourier_price():
    params = HestonParams(kappa=2.0, theta=0.04, sigma=0.35, rho=-0.5, v0=0.04)
    fourier = heston_price(100.0, 100.0, 1.0, 0.03, params)
    monte_carlo = heston_monte_carlo_price(100.0, 100.0, 1.0, 0.03, params, paths=30_000, steps=96, seed=42)
    assert abs(monte_carlo - fourier) < 0.75
