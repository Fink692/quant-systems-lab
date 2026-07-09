import numpy as np

from quantlab.market_making.avellaneda_stoikov import AvellanedaStoikovParams
from quantlab.market_making.book_simulator import simulate_order_book_market_maker
from quantlab.market_making.limit_order_book import LimitOrderBook
from quantlab.workflows.demo_suite import run_full_demo


def test_limit_order_book_depth_and_cancel_quantity():
    book = LimitOrderBook()
    book.add_limit_order("buy", 99.0, 10.0)
    book.add_limit_order("buy", 99.0, 2.0)

    assert book.depth_at("buy", 99.0) == 12.0
    assert book.cancel_quantity("buy", 99.0, 5.0) == 5.0
    assert book.depth_at("buy", 99.0) == 7.0
    assert book.cancel_quantity("buy", 99.0, 20.0) == 7.0
    assert book.depth_at("buy", 99.0) == 0.0


def test_order_book_market_maker_receives_queue_based_fills():
    result = simulate_order_book_market_maker(
        100.0,
        AvellanedaStoikovParams(risk_aversion=0.08, volatility=0.18, order_book_liquidity=1.2, horizon=1.0),
        steps=60,
        dt=1.0 / 60.0,
        levels=3,
        depth_per_level=2.0,
        order_size=1.0,
        market_order_intensity=500.0,
        seed=12,
    )

    assert len(result.history) == 60
    assert result.fill_rate > 0.0
    assert result.average_spread > 0.0
    assert result.max_inventory_abs >= 0.0
    assert np.isfinite(result.final_pnl)
    assert (result.history[["bid_fill_qty", "ask_fill_qty"]] >= 0.0).all().all()


def test_full_demo_exposes_order_book_market_making_metrics():
    result = run_full_demo(seed=39).as_dict()

    assert result["market_making"]["book_fill_rate"] > 0.0
    assert result["market_making"]["book_average_spread"] > 0.0
    assert result["market_making"]["book_max_inventory_abs"] >= 0.0
