"""Statistical arbitrage diagnostics."""

from quantlab.stat_arb.basket_backtest import JohansenBasketBacktestResult, backtest_johansen_basket_strategy
from quantlab.stat_arb.backtest import SpreadBacktestResult, backtest_spread_strategy
from quantlab.stat_arb.cointegration import CointegrationResult, engle_granger, estimate_ou, zscore
from quantlab.stat_arb.dynamic_backtest import DynamicHedgeBacktestResult, backtest_kalman_spread_strategy
from quantlab.stat_arb.johansen import JohansenResult, basket_spread, johansen_hedge_vector, johansen_test
from quantlab.stat_arb.kalman import KalmanHedgeResult, kalman_dynamic_hedge_ratio
from quantlab.stat_arb.network import CointegrationNetwork, mean_reversion_signal, pairwise_cointegration_network
from quantlab.stat_arb.portfolio import StatArbPortfolioResult, allocate_pair_capital, backtest_pair_portfolio
from quantlab.stat_arb.selection import PairCandidate, candidate_spread_weights, rank_cointegrated_pairs

__all__ = [
    "CointegrationNetwork",
    "CointegrationResult",
    "DynamicHedgeBacktestResult",
    "JohansenResult",
    "JohansenBasketBacktestResult",
    "KalmanHedgeResult",
    "PairCandidate",
    "SpreadBacktestResult",
    "StatArbPortfolioResult",
    "allocate_pair_capital",
    "backtest_kalman_spread_strategy",
    "backtest_johansen_basket_strategy",
    "backtest_pair_portfolio",
    "backtest_spread_strategy",
    "basket_spread",
    "candidate_spread_weights",
    "engle_granger",
    "estimate_ou",
    "johansen_hedge_vector",
    "johansen_test",
    "kalman_dynamic_hedge_ratio",
    "mean_reversion_signal",
    "pairwise_cointegration_network",
    "rank_cointegrated_pairs",
    "zscore",
]
