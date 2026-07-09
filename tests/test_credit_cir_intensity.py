import numpy as np
import pytest

from quantlab.credit.intensity_process import CIRIntensityParams, simulate_cir_intensity
from quantlab.workflows.demo_suite import run_full_demo


def test_cir_intensity_simulation_outputs_paths_and_default_metrics():
    params = CIRIntensityParams(kappa=1.2, theta=0.03, sigma=0.12, lambda0=0.02)

    result = simulate_cir_intensity(params, maturity=3.0, steps=36, paths=500, seed=9)

    assert result.intensities.shape == (37, 500)
    assert result.integrated_hazard.shape == (500,)
    assert np.all(result.intensities.to_numpy() >= 0.0)
    assert np.all(result.integrated_hazard.to_numpy() >= 0.0)
    assert 0.0 <= result.default_probability <= 1.0
    assert 0.0 <= result.mean_survival_probability <= 1.0
    assert result.mean_terminal_intensity >= 0.0
    assert result.default_times.dropna().between(0.0, 3.0).all()


def test_cir_intensity_reduces_to_deterministic_flat_hazard_when_no_vol_or_mean_reversion_gap():
    params = CIRIntensityParams(kappa=1.0, theta=0.04, sigma=0.0, lambda0=0.04)

    result = simulate_cir_intensity(params, maturity=2.0, steps=24, paths=100, seed=5)

    assert np.allclose(result.intensities.to_numpy(), 0.04)
    assert np.allclose(result.integrated_hazard.to_numpy(), 0.08)
    assert np.allclose(result.survival_probabilities.to_numpy(), np.exp(-0.08))


def test_cir_intensity_validates_parameters():
    with pytest.raises(ValueError, match="kappa"):
        simulate_cir_intensity(CIRIntensityParams(kappa=0.0, theta=0.03, sigma=0.1, lambda0=0.02), maturity=1.0)


def test_full_demo_exposes_cir_intensity_metrics():
    result = run_full_demo(seed=36).as_dict()

    assert 0.0 <= result["credit"]["cir_default_probability"] <= 1.0
    assert 0.0 <= result["credit"]["cir_mean_survival_probability"] <= 1.0
    assert result["credit"]["cir_terminal_intensity"] >= 0.0
