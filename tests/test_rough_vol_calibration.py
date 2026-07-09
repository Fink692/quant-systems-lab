import numpy as np
import pandas as pd
import pytest

from quantlab.rough_vol.calibration import calibrate_rough_bergomi_atm, calibrate_rough_bergomi_from_chain
from quantlab.workflows.demo_suite import run_full_demo


def test_rough_bergomi_atm_proxy_recovers_power_law_parameters():
    maturities = np.array([0.1, 0.25, 0.5, 1.0, 2.0])
    true_hurst = 0.13
    true_xi0 = 0.04
    true_eta = 1.4
    rho = -0.55
    atm_vols = np.full_like(maturities, np.sqrt(true_xi0))
    atm_skews = rho * true_eta * np.sqrt(true_xi0) * maturities ** (true_hurst - 0.5)

    result = calibrate_rough_bergomi_atm(maturities, atm_vols, atm_skews, rho=rho)

    assert abs(result.params.hurst - true_hurst) < 1e-12
    assert abs(result.params.xi0 - true_xi0) < 1e-12
    assert abs(result.params.eta - true_eta) < 1e-12
    assert result.objective_value < 1e-12


def test_rough_bergomi_chain_calibration_extracts_atm_smile_slopes():
    maturities = np.array([0.1, 0.25, 0.5, 1.0, 2.0])
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    true_hurst = 0.16
    xi0 = 0.04
    eta = 1.1
    rho = -0.5
    rows = []
    for maturity in maturities:
        atm_vol = np.sqrt(xi0)
        skew = rho * eta * np.sqrt(xi0) * maturity ** (true_hurst - 0.5)
        for strike in strikes:
            log_moneyness = np.log(strike / 100.0)
            rows.append(
                {
                    "spot": 100.0,
                    "strike": strike,
                    "maturity": maturity,
                    "rate": 0.0,
                    "dividend": 0.0,
                    "option_type": "call",
                    "implied_volatility": atm_vol + skew * log_moneyness,
                }
            )
    chain = pd.DataFrame(rows)

    result = calibrate_rough_bergomi_from_chain(chain, rho=rho)

    assert abs(result.params.hurst - true_hurst) < 1e-12
    assert abs(result.params.eta - eta) < 1e-12
    assert result.atm_vol_rmse < 1e-12


def test_rough_bergomi_atm_proxy_rejects_wrong_rho_sign():
    with pytest.raises(ValueError, match="rho sign"):
        calibrate_rough_bergomi_atm(
            np.array([0.25, 0.5, 1.0]),
            np.array([0.2, 0.2, 0.2]),
            np.array([0.4, 0.3, 0.2]),
            rho=-0.5,
        )


def test_full_demo_exposes_rough_bergomi_proxy_metrics():
    result = run_full_demo(seed=28).as_dict()

    assert 0.0 < result["rough_vol"]["rough_proxy_hurst"] < 0.5
    assert result["rough_vol"]["rough_proxy_eta"] > 0.0
    assert result["rough_vol"]["rough_proxy_xi0"] > 0.0
