import numpy as np

from quantlab.options.surface_arbitrage import (
    detect_call_price_bounds,
    detect_surface_arbitrage,
    detect_vertical_spread_arbitrage,
)
from quantlab.workflows.demo_suite import run_full_demo


def test_call_price_bounds_detect_intrinsic_and_upper_bound_violations():
    maturities = np.array([1.0])
    strikes = np.array([80.0, 100.0])
    prices = np.array([[10.0, 101.0]])

    violations = detect_call_price_bounds(maturities, strikes, prices, spot=100.0, rate=0.03)

    assert len(violations) == 2
    assert {violation.kind for violation in violations} == {"bounds"}
    assert violations[0].amount > 0.0
    assert violations[1].amount > 0.0


def test_vertical_spread_detects_increasing_calls_and_too_wide_spreads():
    maturities = np.array([1.0])
    strikes = np.array([90.0, 100.0, 110.0])
    prices = np.array([[20.0, 22.0, 2.0]])

    violations = detect_vertical_spread_arbitrage(maturities, strikes, prices, rate=0.03)

    assert len(violations) == 2
    assert all(violation.kind == "vertical" for violation in violations)
    assert any("increases with strike" in violation.message for violation in violations)
    assert any("wider than" in violation.message for violation in violations)


def test_surface_arbitrage_combines_bounds_vertical_calendar_and_butterfly():
    maturities = np.array([0.5, 1.0])
    strikes = np.array([90.0, 100.0, 110.0])
    prices = np.array([[20.0, 105.0, 2.0], [19.0, 18.0, 1.0]])

    violations = detect_surface_arbitrage(maturities, strikes, prices, spot=100.0, rate=0.03)
    kinds = {violation.kind for violation in violations}

    assert {"bounds", "vertical", "calendar", "butterfly"}.issubset(kinds)


def test_full_demo_exposes_extended_surface_arbitrage_metrics():
    result = run_full_demo(seed=34).as_dict()

    assert "bound_violations" in result["surface_arbitrage"]
    assert "vertical_violations" in result["surface_arbitrage"]
    assert result["surface_arbitrage"]["bound_violations"] >= 0
    assert result["surface_arbitrage"]["vertical_violations"] >= 0
