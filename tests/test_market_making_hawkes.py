import numpy as np
import pytest

from quantlab.market_making.hawkes import (
    HawkesOrderFlowParams,
    hawkes_branching_matrix,
    hawkes_stability_radius,
    simulate_hawkes_order_flow,
)
from quantlab.workflows.demo_suite import run_full_demo


def test_hawkes_branching_matrix_and_stability_radius():
    excitation = np.array([[6.0, 2.0], [1.0, 5.0]])
    branching = hawkes_branching_matrix(excitation, decay=20.0)

    assert np.allclose(branching, excitation / 20.0)
    assert 0.0 < hawkes_stability_radius(excitation, 20.0) < 1.0


def test_hawkes_order_flow_simulates_signed_events():
    params = HawkesOrderFlowParams(
        base_intensity=np.array([25.0, 22.0]),
        excitation=np.array([[5.0, 1.0], [2.0, 4.0]]),
        decay=18.0,
        volume=2.0,
    )

    result = simulate_hawkes_order_flow(params, horizon=1.0, seed=7)

    assert result.event_count > 0
    assert result.buy_count + result.sell_count == result.event_count
    assert set(result.events["side"]).issubset({"buy", "sell"})
    assert result.events["time"].is_monotonic_increasing
    assert result.events["time"].between(0.0, 1.0).all()
    assert np.isclose(result.net_order_flow, result.events["signed_volume"].sum())
    assert -1.0 <= result.order_flow_imbalance <= 1.0
    assert result.realized_intensity == result.event_count


def test_hawkes_order_flow_rejects_unstable_branching():
    params = HawkesOrderFlowParams(
        base_intensity=np.array([10.0, 10.0]),
        excitation=np.array([[20.0, 5.0], [5.0, 20.0]]),
        decay=10.0,
    )

    with pytest.raises(ValueError, match="spectral radius"):
        simulate_hawkes_order_flow(params, horizon=1.0, seed=1)


def test_full_demo_exposes_hawkes_order_flow_metrics():
    result = run_full_demo(seed=30).as_dict()

    assert result["market_making"]["hawkes_event_count"] > 0
    assert 0.0 < result["market_making"]["hawkes_branching_ratio"] < 1.0
    assert -1.0 <= result["market_making"]["hawkes_order_flow_imbalance"] <= 1.0
    assert result["market_making"]["hawkes_realized_intensity"] > 0.0
