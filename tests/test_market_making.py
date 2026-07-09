from quantlab.market_making.avellaneda_stoikov import AvellanedaStoikovParams, optimal_quotes
from quantlab.market_making.limit_order_book import LimitOrderBook


def test_limit_order_book_executes_best_prices_first():
    book = LimitOrderBook()
    book.add_limit_order("buy", 99.0, 10.0)
    book.add_limit_order("sell", 101.0, 4.0)
    book.add_limit_order("sell", 100.5, 3.0)
    fills = book.market_order("buy", 5.0)
    assert [(fill.price, fill.quantity) for fill in fills] == [(100.5, 3.0), (101.0, 2.0)]
    assert book.best_ask == 101.0
    assert book.spread == 2.0


def test_avellaneda_stoikov_quotes_are_ordered():
    bid, ask = optimal_quotes(
        mid_price=100.0,
        inventory=5.0,
        time=0.1,
        params=AvellanedaStoikovParams(risk_aversion=0.1, volatility=0.2, order_book_liquidity=1.5, horizon=1.0),
    )
    assert bid < ask
    assert bid < 100.0
