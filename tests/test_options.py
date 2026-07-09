import numpy as np

from quantlab.options.bates import BatesParams, bates_price
from quantlab.options.black_scholes import black_scholes_price, implied_volatility
from quantlab.options.heston import HestonParams, heston_price
from quantlab.options.sabr import SABRParams, sabr_implied_volatility
from quantlab.options.surface_arbitrage import detect_surface_arbitrage


def test_black_scholes_put_call_parity_and_implied_volatility():
    spot, strike, maturity, rate, dividend, vol = 100.0, 103.0, 1.2, 0.04, 0.01, 0.23
    call = black_scholes_price(spot, strike, maturity, rate, vol, dividend, "call")
    put = black_scholes_price(spot, strike, maturity, rate, vol, dividend, "put")
    parity = spot * np.exp(-dividend * maturity) - strike * np.exp(-rate * maturity)
    assert abs(call - put - parity) < 1e-10
    assert abs(implied_volatility(call, spot, strike, maturity, rate, dividend, "call") - vol) < 1e-8


def test_heston_and_bates_prices_are_finite():
    params = HestonParams(kappa=2.0, theta=0.04, sigma=0.35, rho=-0.5, v0=0.04)
    heston = heston_price(100.0, 100.0, 1.0, 0.03, params)
    bates = bates_price(100.0, 100.0, 1.0, 0.03, BatesParams(params, 0.1, -0.05, 0.2))
    assert np.isfinite(heston)
    assert np.isfinite(bates)
    assert 0.0 < heston < 100.0
    assert 0.0 < bates < 100.0


def test_sabr_implied_volatility_positive():
    vol = sabr_implied_volatility(100.0, 105.0, 2.0, SABRParams(alpha=0.25, beta=0.7, rho=-0.2, nu=0.5))
    assert vol > 0


def test_surface_arbitrage_detector_flags_calendar_and_butterfly():
    maturities = np.array([0.5, 1.0])
    strikes = np.array([90.0, 100.0, 110.0])
    prices = np.array([[15.0, 14.0, 8.0], [16.0, 10.0, 7.5]])
    violations = detect_surface_arbitrage(maturities, strikes, prices)
    kinds = {violation.kind for violation in violations}
    assert "calendar" in kinds
    assert "butterfly" in kinds
