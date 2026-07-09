import numpy as np

from quantlab.credit.counterparty import exposure_profile, netting_set_exposure_profile, unilateral_cva, wrong_way_adjusted_profile
from quantlab.credit.curve import HazardCurve, bootstrap_hazard_curve
from quantlab.workflows.demo_suite import run_full_demo


def test_exposure_profile_clips_positive_exposure_and_pfe():
    times = np.array([0.5, 1.0, 2.0])
    paths = np.array(
        [
            [10.0, -5.0, 20.0],
            [-2.0, 15.0, 30.0],
            [4.0, 5.0, -10.0],
        ]
    )
    profile = exposure_profile(times, paths, pfe_quantile=0.8)
    assert np.allclose(profile.expected_exposure, paths.mean(axis=0))
    assert np.all(profile.expected_positive_exposure >= 0.0)
    assert profile.peak_pfe >= profile.average_epe


def test_netting_set_exposure_profile_aggregates_trade_paths():
    times = np.array([1.0, 2.0])
    trade_paths = np.array(
        [
            [[10.0, -5.0], [-3.0, 9.0]],
            [[4.0, 8.0], [1.0, -2.0]],
        ]
    )
    profile = netting_set_exposure_profile(times, trade_paths)
    direct = exposure_profile(times, trade_paths.sum(axis=1))
    assert np.allclose(profile.expected_positive_exposure, direct.expected_positive_exposure)


def test_unilateral_cva_is_positive_and_sensitive_to_credit_quality():
    times = np.array([1.0, 2.0, 3.0])
    paths = np.array([[100.0, 80.0, 50.0], [120.0, 90.0, 40.0], [-20.0, 30.0, 60.0]])
    profile = exposure_profile(times, paths)
    low_hazard = bootstrap_hazard_curve(times, np.array([0.005, 0.006, 0.007]), recovery_rate=0.4)
    high_hazard = bootstrap_hazard_curve(times, np.array([0.02, 0.025, 0.03]), recovery_rate=0.4)
    low_cva = unilateral_cva(profile, low_hazard, rate=0.03)
    high_cva = unilateral_cva(profile, high_hazard, rate=0.03)
    high_recovery = unilateral_cva(profile, high_hazard, rate=0.03, recovery_rate=0.8)

    assert low_cva.cva > 0.0
    assert high_cva.cva > low_cva.cva
    assert high_recovery.cva < high_cva.cva
    assert abs(high_cva.contributions["cva_contribution"].sum() - high_cva.cva) < 1e-12


def test_wrong_way_adjusted_profile_changes_cva():
    times = np.array([1.0, 2.0, 3.0, 4.0])
    paths = np.full((4, 4), 100.0)
    profile = exposure_profile(times, paths)
    curve = HazardCurve(maturities=times, hazard_rates=np.full(4, 0.02), recovery_rate=0.4)
    adjusted = wrong_way_adjusted_profile(profile, np.array([0.01, 0.02, 0.04, 0.08]), beta=0.5)
    base_cva = unilateral_cva(profile, curve, rate=0.01)
    adjusted_cva = unilateral_cva(adjusted, curve, rate=0.01)
    assert adjusted.peak_pfe > profile.peak_pfe
    assert adjusted_cva.cva != base_cva.cva


def test_full_demo_exposes_counterparty_cva_metrics():
    result = run_full_demo(seed=26).as_dict()
    assert result["credit"]["counterparty_cva"] > 0.0
    assert result["credit"]["wrong_way_cva"] > 0.0
    assert result["credit"]["counterparty_peak_pfe"] > 0.0
