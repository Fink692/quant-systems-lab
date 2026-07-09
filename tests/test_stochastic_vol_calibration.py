import numpy as np

from quantlab.options.bates import BatesParams, bates_price
from quantlab.options.calibration import OptionQuote, calibrate_bates, calibrate_heston
from quantlab.options.heston import HestonParams, heston_price
from quantlab.workflows.demo_suite import run_full_demo


def test_heston_calibration_recovers_model_generated_quotes():
    spot = 100.0
    rate = 0.03
    true_params = HestonParams(kappa=1.5, theta=0.04, sigma=0.3, rho=-0.4, v0=0.04)
    initial = HestonParams(kappa=1.3, theta=0.05, sigma=0.35, rho=-0.3, v0=0.05)
    quote_specs = [(90.0, 0.5), (100.0, 0.75), (110.0, 1.0), (100.0, 1.5), (95.0, 2.0)]
    quotes = [
        OptionQuote(strike=strike, maturity=maturity, price=heston_price(spot, strike, maturity, rate, true_params))
        for strike, maturity in quote_specs
    ]

    result = calibrate_heston(quotes, spot, rate, initial=initial, max_nfev=30)

    assert result.success
    assert result.objective_value < 1e-10
    assert abs(result.parameters["theta"] - true_params.theta) < 1e-5
    assert abs(result.parameters["v0"] - true_params.v0) < 1e-5
    assert abs(result.parameters["rho"] - true_params.rho) < 1e-5


def test_bates_jump_calibration_recovers_fixed_heston_jump_parameters():
    spot = 100.0
    rate = 0.03
    heston = HestonParams(kappa=1.5, theta=0.04, sigma=0.3, rho=-0.4, v0=0.04)
    true_params = BatesParams(heston=heston, jump_intensity=0.15, jump_mean=-0.04, jump_volatility=0.2)
    quote_specs = [(90.0, 0.5), (100.0, 0.75), (110.0, 1.0), (100.0, 1.5)]
    quotes = [
        OptionQuote(strike=strike, maturity=maturity, price=bates_price(spot, strike, maturity, rate, true_params))
        for strike, maturity in quote_specs
    ]

    result = calibrate_bates(quotes, spot, rate, heston, initial_jumps=(0.1, -0.02, 0.15), max_nfev=30)

    assert result.success
    assert result.objective_value < 1e-10
    assert abs(result.parameters["jump_intensity"] - true_params.jump_intensity) < 1e-5
    assert abs(result.parameters["jump_mean"] - true_params.jump_mean) < 1e-5
    assert abs(result.parameters["jump_volatility"] - true_params.jump_volatility) < 1e-5


def test_full_demo_exposes_heston_and_bates_calibration_metrics():
    result = run_full_demo(seed=37).as_dict()

    assert result["options"]["heston_calibration_objective"] < 1e-10
    assert result["options"]["bates_calibration_objective"] < 1e-10
    assert np.isfinite(result["options"]["heston_calibrated_v0"])
    assert result["options"]["bates_calibrated_jump_intensity"] >= 0.0
