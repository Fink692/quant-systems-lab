import numpy as np

from quantlab.data.synthetic import synthetic_option_chain
from quantlab.options.surface import VolatilitySurface, build_volatility_surface_from_chain
from quantlab.options.surface_stability import diagnose_surface_interpolation_stability
from quantlab.workflows.demo_suite import run_full_demo


def test_surface_interpolation_stability_passes_for_smooth_synthetic_chain():
    surface = build_volatility_surface_from_chain(synthetic_option_chain())

    report = diagnose_surface_interpolation_stability(surface, maturity_points=9, strike_points=21)

    assert report.invalid_implied_vol_count == 0
    assert report.implied_volatilities.shape == (9, 21)
    assert report.call_prices.shape == (9, 21)
    assert report.local_vol_nan_fraction < 0.25
    assert report.max_implied_vol_step >= 0.0
    assert report.passes


def test_surface_interpolation_stability_flags_spiky_unstable_surface():
    maturities = np.array([0.25, 0.75, 1.5, 2.0])
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    vols = np.full((4, 5), 0.2)
    vols[1, 2] = 1.25
    surface = VolatilitySurface(maturities, strikes, vols, spot=100.0, rate=0.03)

    report = diagnose_surface_interpolation_stability(surface, maturity_points=9, strike_points=21)

    assert report.invalid_implied_vol_count == 0
    assert len(report.arbitrage_violations) > 0 or report.local_vol_nan_fraction >= 0.25
    assert report.strike_curvature_rms > 0.0


def test_full_demo_exposes_surface_interpolation_stability_metrics():
    result = run_full_demo(seed=38).as_dict()

    assert "interpolation_stability_passes" in result["surface_arbitrage"]
    assert result["surface_arbitrage"]["interpolation_dense_violations"] >= 0
    assert 0.0 <= result["surface_arbitrage"]["interpolation_local_vol_nan_fraction"] <= 1.0
