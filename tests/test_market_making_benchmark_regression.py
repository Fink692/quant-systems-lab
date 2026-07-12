import json
from pathlib import Path

import pytest

from quantlab.market_making.avellaneda_stoikov import AvellanedaStoikovParams
from quantlab.market_making.book_simulator import simulate_order_book_market_maker


def test_seed12_market_making_benchmark_does_not_drift() -> None:
    benchmark = json.loads(
        (Path(__file__).resolve().parents[1] / "config" / "benchmarks" / "market_making_seed12.json").read_text()
    )
    result = simulate_order_book_market_maker(
        100.0,
        AvellanedaStoikovParams(risk_aversion=0.08, volatility=0.18, order_book_liquidity=1.2, horizon=1.0),
        steps=benchmark["steps"],
        dt=1.0 / benchmark["steps"],
        levels=3,
        depth_per_level=2.0,
        order_size=1.0,
        market_order_intensity=500.0,
        seed=benchmark["seed"],
    )
    tolerance = benchmark["absolute_tolerance"]
    actual = {
        "final_pnl": result.final_pnl,
        "fill_rate": result.fill_rate,
        "average_spread": result.average_spread,
        "max_inventory_abs": result.max_inventory_abs,
    }
    for name, expected in benchmark["expected"].items():
        assert actual[name] == pytest.approx(expected, abs=tolerance), name
