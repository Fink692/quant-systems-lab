from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.credit.counterparty import exposure_profile, unilateral_cva, wrong_way_adjusted_profile
from quantlab.credit.cox import fit_cox_ph
from quantlab.credit.curve import bootstrap_hazard_curve
from quantlab.credit.default_models import merton_default_probability
from quantlab.credit.intensity import fit_logistic_hazard
from quantlab.credit.intensity_process import CIRIntensityParams, simulate_cir_intensity
from quantlab.credit.kmv import default_point, distance_to_default
from quantlab.credit.migration import cumulative_default_probability
from quantlab.credit.portfolio import gaussian_copula_default_losses
from quantlab.credit.pricing import cds_par_spread, risky_coupon_bond_price, risky_zero_coupon_price
from quantlab.credit.sensitivity import coupon_bond_spread_sensitivity
from quantlab.credit.structural import calibrate_merton_asset_parameters
from quantlab.credit.survival import fit_exponential_hazard
from quantlab.credit.tranches import tranche_loss_distribution
from quantlab.data.loaders import (
    returns_from_prices,
    validate_credit_spread_curve,
    validate_option_chain,
    validate_price_panel,
)
from quantlab.data.synthetic import (
    synthetic_cointegrated_prices,
    synthetic_credit_spreads,
    synthetic_exposure_network,
    synthetic_factor_panel,
    synthetic_option_chain,
)
from quantlab.market_making.attribution import attribute_market_making_pnl
from quantlab.market_making.avellaneda_stoikov import AvellanedaStoikovParams
from quantlab.market_making.book_simulator import simulate_order_book_market_maker
from quantlab.market_making.execution import ExecutionModelParams, expected_execution_value
from quantlab.market_making.fill_calibration import calibrate_fill_intensity
from quantlab.market_making.hawkes import HawkesOrderFlowParams, simulate_hawkes_order_flow
from quantlab.market_making.inventory import inventory_diagnostics
from quantlab.market_making.latency import latency_slippage_report
from quantlab.market_making.path_simulator import simulate_latency_market_maker_on_path
from quantlab.market_making.queue import simulate_queue_position
from quantlab.market_making.simulator import simulate_market_maker
from quantlab.market_making.toxicity import adverse_selection_report, order_flow_imbalance, volume_synchronized_pin
from quantlab.options.bates import BatesParams, bates_price
from quantlab.options.black_scholes import black_scholes_price
from quantlab.options.calibration import OptionQuote, calibrate_bates, calibrate_heston, calibrate_sabr_smile
from quantlab.options.density import breeden_litzenberger_density
from quantlab.options.diagnostics import pricing_error_report
from quantlab.options.greeks import black_scholes_greeks
from quantlab.options.hedging import simulate_delta_hedge
from quantlab.options.heston import HestonParams, heston_price
from quantlab.options.local_volatility import dupire_local_volatility
from quantlab.options.monte_carlo import heston_monte_carlo_price
from quantlab.options.portfolio import OptionPosition, stress_option_book
from quantlab.options.sabr import SABRParams, sabr_implied_volatility
from quantlab.options.sabr_surface import calibrate_sabr_surface
from quantlab.options.ssvi import SSVIParams, check_ssvi_no_arbitrage, ssvi_surface
from quantlab.options.surface import build_volatility_surface_from_chain
from quantlab.options.surface_arbitrage import detect_surface_arbitrage
from quantlab.options.surface_repair import repair_call_price_surface
from quantlab.options.surface_stability import diagnose_surface_interpolation_stability
from quantlab.options.svi import calibrate_svi_slice
from quantlab.options.variance_reduction import black_scholes_antithetic_price, black_scholes_control_variate_price
from quantlab.portfolio.bayesian import bayesian_mean_variance_weights, bayesian_return_posterior
from quantlab.portfolio.black_litterman import black_litterman_posterior
from quantlab.portfolio.cdar import cdar_minimizing_weights
from quantlab.portfolio.constraints import turnover_constrained_mean_variance_weights
from quantlab.portfolio.cvar_attribution import portfolio_cvar_contributions
from quantlab.portfolio.drawdown import portfolio_drawdown_summary
from quantlab.portfolio.frontier import efficient_frontier
from quantlab.portfolio.optimization import mean_variance_weights, min_variance_weights, risk_parity_weights
from quantlab.portfolio.risk_budget import portfolio_risk_contributions, risk_budget_weights
from quantlab.portfolio.robust import ellipsoidal_robust_mean_variance_weights, robust_mean_variance_weights
from quantlab.portfolio.stress import historical_stress_scenarios, stress_test_portfolio
from quantlab.risk.attribution import factor_risk_attribution
from quantlab.risk.backtesting import basel_traffic_light, christoffersen_var_backtest, kupiec_var_backtest
from quantlab.risk.covariance import ledoit_wolf_covariance
from quantlab.risk.cross_sectional import (
    build_sector_exposures,
    estimate_cross_sectional_factor_returns,
    factor_mimicking_portfolios,
    neutralize_portfolio_exposures,
)
from quantlab.risk.factor_model import fit_factor_model
from quantlab.risk.macro import fit_macro_factor_model, macro_surprise_factors
from quantlab.risk.model_validation import rolling_factor_model_validation
from quantlab.risk.statistical_factors import fit_pca_factor_model
from quantlab.risk.style_factors import build_style_exposures, estimate_style_factor_returns
from quantlab.risk.var import historical_var
from quantlab.rl.deep_q import train_deep_q_learning
from quantlab.rl.evaluation import constant_weight_policy, run_policy
from quantlab.rl.policy_gradient import (
    PolicyGradientRiskConstraints,
    train_constrained_policy_gradient,
    train_softmax_policy_gradient,
)
from quantlab.rl.policy_search import grid_search_constant_weight_policy
from quantlab.rl.portfolio_env import (
    PortfolioTradingEnv,
    constant_mix_policy,
    momentum_rotation_policy,
    run_portfolio_policy,
)
from quantlab.rl.q_learning import train_tabular_q_learning
from quantlab.rl.risk_controls import RiskLimits, risk_limited_policy, volatility_target_weight
from quantlab.rl.risk_metrics import performance_summary
from quantlab.rl.trading_env import TradingEnv
from quantlab.rl.walk_forward import walk_forward_q_learning
from quantlab.rough_vol.calibration import calibrate_rough_bergomi_from_chain, estimate_hurst_from_variogram
from quantlab.rough_vol.pricing import rough_bergomi_option_price
from quantlab.rough_vol.rough_bergomi import RoughBergomiParams, simulate_rough_bergomi
from quantlab.rough_vol.skew import fit_atm_skew_power_law
from quantlab.stat_arb.backtest import backtest_spread_strategy
from quantlab.stat_arb.basket_backtest import backtest_johansen_basket_strategy
from quantlab.stat_arb.cointegration import engle_granger, estimate_ou
from quantlab.stat_arb.dynamic_backtest import backtest_kalman_spread_strategy
from quantlab.stat_arb.johansen import basket_spread, johansen_hedge_vector
from quantlab.stat_arb.kalman import kalman_dynamic_hedge_ratio
from quantlab.stat_arb.network import mean_reversion_signal, pairwise_cointegration_network
from quantlab.stat_arb.portfolio import backtest_pair_portfolio
from quantlab.stat_arb.selection import candidate_spread_weights, rank_cointegrated_pairs
from quantlab.systemic.capital import capital_adequacy, systemic_capital_surcharge
from quantlab.systemic.clearing import eisenberg_noe_clearing
from quantlab.systemic.contagion import eigenvalue_stability, simulate_contagion
from quantlab.systemic.debtrank import debt_rank
from quantlab.systemic.firesale import simulate_fire_sale
from quantlab.systemic.liquidity import simulate_liquidity_spiral
from quantlab.systemic.monte_carlo import simulate_systemic_monte_carlo
from quantlab.systemic.scenarios import run_systemic_stress_scenarios
from quantlab.systemic.stress import exposure_centrality, external_asset_stress


@dataclass(frozen=True)
class DemoSuiteResult:
    summaries: dict[str, dict[str, float | int | bool]]

    def as_dict(self) -> dict[str, dict[str, float | int | bool]]:
        return self.summaries


def run_full_demo(seed: int = 7) -> DemoSuiteResult:
    """Run a deterministic smoke workflow across all ten quant project families."""
    option_chain = synthetic_option_chain()
    option_chain_validation = validate_option_chain(option_chain)
    call_chain = option_chain[option_chain["option_type"] == "call"]
    first_maturity = float(call_chain["maturity"].min())
    smile = call_chain[call_chain["maturity"] == first_maturity].sort_values("strike")
    sabr_calibration = calibrate_sabr_smile(
        forward=100.0 * np.exp(0.03 * first_maturity),
        maturity=first_maturity,
        strikes=smile["strike"].to_numpy(),
        implied_volatilities=smile["implied_volatility"].to_numpy(),
        beta=0.6,
    )
    forward = 100.0 * np.exp(0.03 * first_maturity)
    svi_calibration = calibrate_svi_slice(
        np.log(smile["strike"].to_numpy() / forward),
        smile["implied_volatility"].to_numpy(),
        first_maturity,
    )
    sabr_surface = calibrate_sabr_surface(option_chain, beta=0.6)
    smile_quotes = [
        OptionQuote(strike=float(row.strike), maturity=float(row.maturity), price=float(row.price), option_type="call")
        for row in smile.itertuples()
    ]
    calibrated_sabr = SABRParams(
        alpha=sabr_calibration.parameters["alpha"],
        beta=sabr_calibration.parameters["beta"],
        rho=sabr_calibration.parameters["rho"],
        nu=sabr_calibration.parameters["nu"],
    )
    sabr_model_prices = np.array(
        [
            black_scholes_price(
                100.0,
                quote.strike,
                quote.maturity,
                0.03,
                sabr_implied_volatility(forward, quote.strike, quote.maturity, calibrated_sabr),
            )
            for quote in smile_quotes
        ]
    )
    pricing_diagnostics = pricing_error_report(smile_quotes, sabr_model_prices)
    heston_params = HestonParams(kappa=2.0, theta=0.04, sigma=0.35, rho=-0.5, v0=0.04)
    heston_fourier = heston_price(100.0, 100.0, 1.0, 0.03, heston_params)
    heston_mc = heston_monte_carlo_price(100.0, 100.0, 1.0, 0.03, heston_params, paths=1_000, steps=32, seed=seed)
    heston_calibration_quotes = [
        OptionQuote(strike=95.0, maturity=0.5, price=heston_price(100.0, 95.0, 0.5, 0.03, heston_params)),
        OptionQuote(strike=100.0, maturity=1.0, price=heston_price(100.0, 100.0, 1.0, 0.03, heston_params)),
        OptionQuote(strike=105.0, maturity=1.5, price=heston_price(100.0, 105.0, 1.5, 0.03, heston_params)),
    ]
    heston_calibration = calibrate_heston(heston_calibration_quotes, 100.0, 0.03, initial=heston_params, max_nfev=2)
    bates_params = BatesParams(heston_params, jump_intensity=0.12, jump_mean=-0.04, jump_volatility=0.18)
    bates_calibration_quotes = [
        OptionQuote(strike=95.0, maturity=0.5, price=bates_price(100.0, 95.0, 0.5, 0.03, bates_params)),
        OptionQuote(strike=100.0, maturity=1.0, price=bates_price(100.0, 100.0, 1.0, 0.03, bates_params)),
        OptionQuote(strike=105.0, maturity=1.5, price=bates_price(100.0, 105.0, 1.5, 0.03, bates_params)),
    ]
    bates_calibration = calibrate_bates(
        bates_calibration_quotes,
        100.0,
        0.03,
        heston_params,
        initial_jumps=(bates_params.jump_intensity, bates_params.jump_mean, bates_params.jump_volatility),
        max_nfev=2,
    )
    antithetic = black_scholes_antithetic_price(100.0, 100.0, 1.0, 0.03, 0.2, paths=5_000, seed=seed)
    control_variate = black_scholes_control_variate_price(100.0, 100.0, 1.0, 0.03, 0.2, paths=5_000, seed=seed)
    hedge_path = np.linspace(100.0, 104.0, 21)
    hedge = simulate_delta_hedge(hedge_path, 100.0, 1.0, 0.03, 0.2, transaction_cost_bps=0.5)
    option_stress = stress_option_book(
        100.0,
        0.03,
        [
            OptionPosition(quantity=10.0, strike=100.0, maturity=1.0, volatility=0.2, option_type="call"),
            OptionPosition(quantity=-5.0, strike=95.0, maturity=1.0, volatility=0.22, option_type="put"),
        ],
        spot_shocks=np.array([-0.1, 0.0, 0.1]),
        volatility_shocks=np.array([-0.02, 0.0, 0.02]),
    )

    pivot_prices = call_chain.pivot(index="maturity", columns="strike", values="price").sort_index().sort_index(axis=1)
    first_price_slice = pivot_prices.loc[first_maturity].to_numpy(dtype=float)
    density = breeden_litzenberger_density(
        pivot_prices.columns.to_numpy(dtype=float), first_price_slice, first_maturity, 0.03
    )
    arbitrage_violations = detect_surface_arbitrage(
        pivot_prices.index.to_numpy(),
        pivot_prices.columns.to_numpy(dtype=float),
        pivot_prices.to_numpy(),
        spot=100.0,
        rate=0.03,
    )
    surface_repair = repair_call_price_surface(
        pivot_prices.index.to_numpy(),
        pivot_prices.columns.to_numpy(dtype=float),
        pivot_prices.to_numpy(),
    )
    local_vol = dupire_local_volatility(
        pivot_prices.index.to_numpy(),
        pivot_prices.columns.to_numpy(dtype=float),
        pivot_prices.to_numpy(),
        rate=0.03,
    )
    surface_stability = diagnose_surface_interpolation_stability(
        build_volatility_surface_from_chain(option_chain),
        maturity_points=9,
        strike_points=21,
    )
    greeks = black_scholes_greeks(100.0, 100.0, 1.0, 0.03, 0.2)
    ssvi_params = SSVIParams(rho=-0.45, eta=0.55, gamma=0.3)
    ssvi_log_moneyness = np.linspace(-0.25, 0.25, 9)
    ssvi_theta = 0.04 * pivot_prices.index.to_numpy(dtype=float)
    ssvi_vols = ssvi_surface(pivot_prices.index.to_numpy(dtype=float), ssvi_log_moneyness, ssvi_theta, ssvi_params)
    ssvi_check = check_ssvi_no_arbitrage(ssvi_theta, ssvi_params)

    market_making = simulate_market_maker(
        100.0,
        AvellanedaStoikovParams(risk_aversion=0.1, volatility=0.2, order_book_liquidity=1.2, horizon=1.0),
        steps=100,
        seed=seed,
    )
    book_market_making = simulate_order_book_market_maker(
        100.0,
        AvellanedaStoikovParams(risk_aversion=0.08, volatility=0.18, order_book_liquidity=1.2, horizon=1.0),
        steps=80,
        dt=1.0 / 80.0,
        levels=4,
        depth_per_level=2.0,
        market_order_intensity=700.0,
        seed=seed,
    )
    path_market_making = simulate_latency_market_maker_on_path(
        market_making.history["mid"].to_numpy(),
        AvellanedaStoikovParams(risk_aversion=0.1, volatility=0.2, order_book_liquidity=1.2, horizon=1.0),
        ExecutionModelParams(fill_intensity=500.0, order_book_liquidity=1.2, latency=0.005, adverse_selection_bps=0.5),
        dt=1.0 / 100.0,
        latency_steps=2,
        seed=seed,
    )
    path_pnl_attribution = attribute_market_making_pnl(path_market_making.history)
    inventory_report = inventory_diagnostics(market_making.history, inventory_limit=5.0)
    expected_bid_value = expected_execution_value(
        100.0,
        99.9,
        "bid",
        horizon=1.0,
        params=ExecutionModelParams(
            fill_intensity=2.0, order_book_liquidity=1.2, latency=0.01, adverse_selection_bps=0.5
        ),
    )
    latency_report = latency_slippage_report(
        quote_mid_prices=np.array([100.0, 100.0, 100.0, 100.0]),
        arrival_mid_prices=np.array([99.98, 100.04, 99.95, 100.07]),
        quote_prices=np.array([99.99, 100.01, 99.98, 100.02]),
        sides=np.array(["bid", "ask", "bid", "ask"]),
    )
    queue = simulate_queue_position(
        20.0, 5.0, market_order_intensity=40.0, cancellation_intensity=8.0, horizon=1.0, seed=seed
    )
    quote_distances = np.array([0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20])
    horizons = np.ones_like(quote_distances)
    fill_flags = np.array([True, False, True, True, False, True, False, False])
    fill_calibration = calibrate_fill_intensity(quote_distances, horizons, fill_flags)
    trade_signs = np.sign(
        np.diff(market_making.history["mid"].to_numpy(), prepend=market_making.history["mid"].iloc[0])
    )
    toxicity = adverse_selection_report(trade_signs, market_making.history["mid"].to_numpy(), horizon=1)
    imbalance = order_flow_imbalance(np.maximum(trade_signs, 0.0), np.maximum(-trade_signs, 0.0))
    vpin = volume_synchronized_pin(trade_signs, bucket_volume=5.0)
    hawkes_flow = simulate_hawkes_order_flow(
        HawkesOrderFlowParams(
            base_intensity=np.array([35.0, 32.0]),
            excitation=np.array([[7.0, 2.0], [2.5, 6.0]]),
            decay=20.0,
            volume=1.0,
        ),
        horizon=1.0,
        seed=seed,
    )

    factor_panel = synthetic_factor_panel(seed=seed)
    price_panel_validation = validate_price_panel(factor_panel.prices)
    price_panel_returns = returns_from_prices(factor_panel.prices, method="log")
    benchmark_prices = factor_panel.prices["Asset1"].to_numpy()
    rl_backtest = run_policy(TradingEnv(benchmark_prices, transaction_cost_bps=1.0), constant_weight_policy(0.6))
    rl_summary = performance_summary(rl_backtest.history["equity"].to_numpy())
    risk_limited_backtest = run_policy(
        TradingEnv(benchmark_prices, transaction_cost_bps=1.0),
        risk_limited_policy(
            constant_weight_policy(1.0), RiskLimits(max_leverage=1.0, max_drawdown=0.05, de_risk_weight=0.0)
        ),
    )
    realized_vol = float(np.std(np.diff(np.log(benchmark_prices)), ddof=1) * np.sqrt(252.0))
    vol_target = volatility_target_weight(1.0, realized_vol, target_volatility=0.12, max_leverage=1.5)
    policy_search = grid_search_constant_weight_policy(
        benchmark_prices, candidate_weights=np.array([-1.0, 0.0, 0.5, 1.0]), transaction_cost_bps=1.0
    )
    q_learning = train_tabular_q_learning(
        benchmark_prices, candidate_weights=np.array([-1.0, 0.0, 1.0]), episodes=25, epsilon=0.1, seed=seed
    )
    policy_gradient = train_softmax_policy_gradient(
        benchmark_prices,
        candidate_weights=np.array([-1.0, 0.0, 1.0]),
        episodes=15,
        learning_rate=0.03,
        seed=seed,
    )
    constrained_policy_gradient = train_constrained_policy_gradient(
        benchmark_prices,
        candidate_weights=np.array([-1.0, 0.0, 1.0]),
        constraints=PolicyGradientRiskConstraints(max_drawdown=0.03, max_turnover=0.75),
        episodes=12,
        learning_rate=0.03,
        penalty_learning_rate=2.0,
        seed=seed,
    )
    deep_q = train_deep_q_learning(
        benchmark_prices,
        candidate_weights=np.array([-1.0, 0.0, 1.0]),
        episodes=8,
        hidden_units=6,
        learning_rate=0.01,
        epsilon=0.15,
        batch_size=8,
        replay_capacity=128,
        seed=seed,
    )
    q_walk_forward = walk_forward_q_learning(
        benchmark_prices,
        train_size=80,
        test_size=40,
        candidate_weights=np.array([-1.0, 0.0, 1.0]),
        episodes=10,
        seed=seed,
    )
    portfolio_rl_prices = factor_panel.prices.iloc[:, :4]
    portfolio_constant = run_portfolio_policy(
        PortfolioTradingEnv(
            portfolio_rl_prices,
            transaction_cost_bps=1.0,
            drawdown_penalty=0.05,
            volatility_penalty=0.01,
            volatility_window=20,
        ),
        constant_mix_policy(np.full(portfolio_rl_prices.shape[1], 1.0 / portfolio_rl_prices.shape[1])),
    )
    portfolio_momentum = run_portfolio_policy(
        PortfolioTradingEnv(
            portfolio_rl_prices,
            transaction_cost_bps=2.0,
            drawdown_penalty=0.05,
            volatility_penalty=0.01,
            volatility_window=20,
        ),
        momentum_rotation_policy(lookback=20, top_n=2),
    )

    factor_result = fit_factor_model(factor_panel.asset_returns, factor_panel.factor_returns)
    factor_validation = rolling_factor_model_validation(
        factor_panel.asset_returns,
        factor_panel.factor_returns,
        train_window=120,
        test_window=40,
        step_size=40,
    )
    pca_result = fit_pca_factor_model(factor_panel.asset_returns, n_factors=3)
    rng = np.random.default_rng(seed)
    fundamentals = pd.DataFrame(
        {
            "market_cap": rng.lognormal(mean=9.0, sigma=0.8, size=factor_panel.asset_returns.shape[1]),
            "book_to_market": rng.lognormal(mean=0.0, sigma=0.4, size=factor_panel.asset_returns.shape[1]),
            "momentum": factor_panel.asset_returns.tail(60).sum(axis=0).to_numpy(),
            "volatility": factor_panel.asset_returns.tail(60).std(axis=0).to_numpy(),
        },
        index=factor_panel.asset_returns.columns,
    )
    style_exposures = build_style_exposures(fundamentals)
    style_result = estimate_style_factor_returns(factor_panel.asset_returns, style_exposures)
    sectors = pd.Series(
        ["technology", "technology", "financials", "financials", "energy", "energy", "healthcare", "healthcare"],
        index=factor_panel.asset_returns.columns,
        name="sector",
    )
    sector_exposures = build_sector_exposures(sectors)
    cross_sectional_exposures = pd.concat([style_exposures[["size", "momentum"]], sector_exposures], axis=1)
    cross_sectional_result = estimate_cross_sectional_factor_returns(
        factor_panel.asset_returns,
        cross_sectional_exposures,
        regression_weights=fundamentals["market_cap"] / fundamentals["market_cap"].mean(),
        ridge=1e-6,
    )
    macro_levels = pd.DataFrame(
        {
            "growth": 100.0 + np.cumsum(0.04 + rng.normal(0.0, 0.8, size=len(factor_panel.asset_returns))),
            "inflation": 2.0 + np.cumsum(rng.normal(0.0, 0.03, size=len(factor_panel.asset_returns))),
            "rates": 3.0 + np.cumsum(rng.normal(0.0, 0.04, size=len(factor_panel.asset_returns))),
        },
        index=factor_panel.asset_returns.index,
    )
    macro_factors = macro_surprise_factors(macro_levels, transform="diff", standardize=True)
    macro_model = fit_macro_factor_model(factor_panel.asset_returns, macro_factors, ridge=1e-6)
    covariance = factor_panel.asset_returns.cov().to_numpy()
    lw_covariance = ledoit_wolf_covariance(factor_panel.asset_returns).to_numpy()
    expected_returns = factor_panel.asset_returns.mean().to_numpy()
    min_var = min_variance_weights(covariance)
    mean_var = mean_variance_weights(expected_returns, covariance, risk_aversion=8.0)
    risk_parity = risk_parity_weights(covariance)
    cdar = cdar_minimizing_weights(factor_panel.asset_returns, confidence=0.9)
    robust = robust_mean_variance_weights(expected_returns, covariance, risk_aversion=8.0, uncertainty_penalty=0.05)
    ellipsoid_robust = ellipsoidal_robust_mean_variance_weights(
        expected_returns,
        covariance,
        mean_uncertainty=lw_covariance / len(factor_panel.asset_returns),
        uncertainty_radius=1.5,
        risk_aversion=8.0,
    )
    bayesian_posterior = bayesian_return_posterior(factor_panel.asset_returns.to_numpy(), prior_strength=25.0)
    bayesian_weights = bayesian_mean_variance_weights(
        factor_panel.asset_returns.to_numpy(), prior_strength=25.0, risk_aversion=8.0
    )
    turnover_weights = turnover_constrained_mean_variance_weights(
        expected_returns, covariance, min_var, max_turnover=0.25, risk_aversion=8.0
    )
    budget_weights = risk_budget_weights(covariance, np.full(covariance.shape[0], 1.0 / covariance.shape[0]))
    budget_contributions = portfolio_risk_contributions(budget_weights, covariance)
    frontier = efficient_frontier(
        expected_returns,
        covariance,
        target_returns=np.linspace(float(np.min(expected_returns)), float(np.max(expected_returns)), 5),
        asset_names=list(factor_panel.asset_returns.columns),
    )
    min_var_weights = pd.Series(min_var, index=factor_panel.asset_returns.columns)
    macro_stress_pnl = macro_model.stress_pnl(
        min_var_weights,
        pd.Series({"growth": -2.0, "inflation": 1.5, "rates": 1.25}),
        portfolio_value=1_000_000.0,
    )
    neutralized_weights = neutralize_portfolio_exposures(min_var_weights, style_exposures[["size", "momentum"]])
    neutralized_style_exposure = (
        style_exposures.loc[neutralized_weights.index, ["size", "momentum"]].T @ neutralized_weights
    )
    mimicking = factor_mimicking_portfolios(style_exposures[["size", "momentum"]])
    portfolio_returns = factor_panel.asset_returns.to_numpy() @ min_var
    cvar_contrib = portfolio_cvar_contributions(factor_panel.asset_returns, min_var)
    drawdown_summary = portfolio_drawdown_summary(factor_panel.asset_returns, min_var, confidence=0.9)
    var_estimate = historical_var(portfolio_returns)
    var_backtest = kupiec_var_backtest(portfolio_returns, np.full_like(portfolio_returns, var_estimate))
    christoffersen = christoffersen_var_backtest(portfolio_returns, np.full_like(portfolio_returns, var_estimate))
    traffic_light = basel_traffic_light(
        var_backtest.exceptions, observations=var_backtest.observations, confidence=0.95
    )
    stress_scenarios = historical_stress_scenarios(factor_panel.asset_returns)
    portfolio_stress = stress_test_portfolio(min_var_weights, stress_scenarios, portfolio_value=1_000_000.0)
    attribution = factor_risk_attribution(
        weights=min_var_weights,
        exposures=factor_result.exposures,
        factor_covariance=factor_result.factor_covariance,
        specific_variance=factor_result.specific_variance,
    )
    black_litterman = black_litterman_posterior(
        covariance,
        market_weights=np.full(covariance.shape[0], 1.0 / covariance.shape[0]),
        views_matrix=np.eye(1, covariance.shape[0], 0),
        views=np.array([0.001]),
    )

    rough_spots, rough_variance = simulate_rough_bergomi(
        100.0,
        maturity=0.5,
        steps=64,
        paths=16,
        params=RoughBergomiParams(hurst=0.12, eta=1.1, rho=-0.55, xi0=0.04),
        seed=seed,
    )
    roughness = estimate_hurst_from_variogram(np.log(rough_variance[0] + 1e-12), max_lag=12)
    rough_option = rough_bergomi_option_price(
        100.0,
        100.0,
        0.5,
        0.03,
        RoughBergomiParams(hurst=0.12, eta=1.1, rho=-0.55, xi0=0.04),
        paths=512,
        steps=32,
        seed=seed,
    )
    atm_skew_fit = fit_atm_skew_power_law(
        np.array([0.1, 0.25, 0.5, 1.0, 2.0]),
        -0.35 * np.array([0.1, 0.25, 0.5, 1.0, 2.0]) ** (0.12 - 0.5),
    )
    rough_proxy = calibrate_rough_bergomi_from_chain(option_chain, rho=-0.55)

    stat_prices = synthetic_cointegrated_prices(seed=seed)
    coint = engle_granger(stat_prices["PairB"].to_numpy(), stat_prices["PairA"].to_numpy())
    ou = estimate_ou(coint.spread)
    network = pairwise_cointegration_network(stat_prices, p_value_threshold=0.05)
    ranked_pairs = rank_cointegrated_pairs(stat_prices, network, max_p_value=0.05, top_n=2)
    top_pair_weights = candidate_spread_weights(ranked_pairs[0]) if ranked_pairs else pd.Series(dtype=float)
    pair_portfolio = (
        backtest_pair_portfolio(
            stat_prices, ranked_pairs, entry_z=1.5, exit_z=0.25, window=30, total_gross=1.0, transaction_cost=0.001
        )
        if ranked_pairs
        else None
    )
    basket_rng = np.random.default_rng(seed + 20_000)
    basket_prices = stat_prices[["PairA", "PairB"]].copy()
    basket_prices["BasketC"] = (
        0.55 * basket_prices["PairA"]
        + 0.35 * basket_prices["PairB"]
        + basket_rng.normal(0.0, 0.15, size=len(basket_prices))
    )
    johansen_basket_backtest = backtest_johansen_basket_strategy(
        basket_prices,
        entry_z=1.4,
        exit_z=0.25,
        window=30,
        gross_exposure=1.0,
        transaction_cost=0.001,
    )
    johansen_vector = johansen_hedge_vector(stat_prices[["PairA", "PairB"]], normalize_asset="PairA")
    johansen_spread = basket_spread(stat_prices[["PairA", "PairB"]], johansen_vector)
    kalman = kalman_dynamic_hedge_ratio(stat_prices["PairB"], stat_prices["PairA"])
    dynamic_backtest = backtest_kalman_spread_strategy(
        stat_prices["PairB"], stat_prices["PairA"], entry_z=1.5, exit_z=0.25, window=30, transaction_cost=0.001
    )
    signal = mean_reversion_signal(coint.spread, entry_z=1.5, exit_z=0.25, window=30)
    spread_backtest = backtest_spread_strategy(coint.spread, signal, transaction_cost=0.001)

    spread_curve = synthetic_credit_spreads()
    spread_curve_validation = validate_credit_spread_curve(spread_curve)
    hazard_curve = bootstrap_hazard_curve(
        spread_curve["maturity"].to_numpy(),
        spread_curve["credit_spread"].to_numpy(),
        recovery_rate=0.4,
    )
    default_probability = merton_default_probability(120.0, 100.0, 1.0, 0.03, 0.25)
    dd = distance_to_default(120.0, default_point(50.0, 100.0), 0.25, drift=0.03)
    merton_calibration = calibrate_merton_asset_parameters(30.0, 0.45, 90.0, 1.0, 0.03)
    risky_bond = risky_zero_coupon_price(5.0, 0.03, hazard_curve)
    risky_coupon = risky_coupon_bond_price(0.04, 5.0, 0.03, hazard_curve)
    spread_sensitivity = coupon_bond_spread_sensitivity(0.04, 5.0, 0.03, hazard_curve)
    cds_spread = cds_par_spread(5.0, 0.03, hazard_curve)
    hazard_fit = fit_exponential_hazard(
        durations=np.array([0.8, 1.4, 2.0, 2.7, 3.2, 4.0, 4.5, 5.0]),
        default_observed=np.array([True, False, True, False, False, True, False, False]),
    )
    logistic_hazard = fit_logistic_hazard(
        pd.DataFrame(
            {
                "leverage": [0.35, 0.42, 0.55, 0.67, 0.72, 0.88, 0.95, 1.10],
                "spread": [0.006, 0.008, 0.011, 0.015, 0.018, 0.025, 0.032, 0.045],
                "profitability": [0.14, 0.12, 0.08, 0.05, 0.03, -0.01, -0.02, -0.05],
            }
        ),
        np.array([False, False, False, False, True, True, True, True]),
        l2_penalty=0.25,
    )
    stressed_credit_covariates = pd.DataFrame({"leverage": [1.0], "spread": [0.035], "profitability": [-0.02]})
    cox_hazard = fit_cox_ph(
        pd.DataFrame(
            {
                "leverage": [0.30, 0.38, 0.52, 0.64, 0.72, 0.84, 0.95, 1.08],
                "spread": [0.005, 0.007, 0.010, 0.014, 0.019, 0.026, 0.034, 0.048],
                "profitability": [0.16, 0.13, 0.10, 0.07, 0.03, 0.00, -0.02, -0.05],
            }
        ),
        durations=np.array([5.0, 4.5, 4.0, 3.2, 2.6, 1.8, 1.3, 0.9]),
        default_observed=np.array([False, False, True, False, True, True, True, True]),
        l2_penalty=0.5,
    )
    cir_intensity = simulate_cir_intensity(
        CIRIntensityParams(kappa=1.4, theta=0.035, sigma=0.16, lambda0=0.02),
        maturity=5.0,
        steps=60,
        paths=1_000,
        seed=seed,
    )
    migration_matrix = np.array(
        [
            [0.90, 0.08, 0.015, 0.005],
            [0.04, 0.88, 0.06, 0.02],
            [0.01, 0.07, 0.82, 0.10],
            [0.00, 0.00, 0.00, 1.00],
        ]
    )
    five_year_migration_pd = cumulative_default_probability(migration_matrix, 1, periods=5, default_state=3)
    credit_losses = gaussian_copula_default_losses(
        default_probabilities=np.array([0.01, 0.02, 0.04, 0.08]),
        exposures=np.array([1_000_000.0, 750_000.0, 500_000.0, 250_000.0]),
        recovery_rates=0.4,
        asset_correlation=0.25,
        simulations=2_000,
        confidence=0.95,
        seed=seed,
    )
    mezzanine_tranche = tranche_loss_distribution(
        credit_losses.losses,
        attachment=0.03,
        detachment=0.10,
        portfolio_notional=2_500_000.0,
        confidence=0.95,
    )
    exposure_times = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
    exposure_rng = np.random.default_rng(seed + 10_000)
    exposure_base = np.array([180_000.0, 260_000.0, 240_000.0, 180_000.0, 80_000.0])
    exposure_noise = exposure_rng.normal(
        0.0, np.array([80_000.0, 110_000.0, 130_000.0, 120_000.0, 90_000.0]), size=(1_000, len(exposure_times))
    )
    exposure_paths = exposure_base + exposure_noise
    counterparty_profile = exposure_profile(exposure_times, exposure_paths, pfe_quantile=0.95)
    cva = unilateral_cva(counterparty_profile, hazard_curve, rate=0.03)
    credit_stress_factor = np.interp(
        exposure_times, spread_curve["maturity"].to_numpy(), spread_curve["credit_spread"].to_numpy()
    )
    wrong_way_profile = wrong_way_adjusted_profile(counterparty_profile, credit_stress_factor, beta=0.35)
    wrong_way_cva = unilateral_cva(wrong_way_profile, hazard_curve, rate=0.03)

    exposures, capital = synthetic_exposure_network(seed=seed)
    contagion = simulate_contagion(
        exposures.to_numpy().copy(), capital.to_numpy(), initial_defaults=[0], recovery_rate=0.4
    )
    clearing = eisenberg_noe_clearing(exposures.to_numpy(), capital.to_numpy())
    debt_rank_result = debt_rank(
        exposures.to_numpy(), capital.to_numpy(), initial_distress=np.array([1.0, 0.0, 0.0, 0.0, 0.0]), damping=0.7
    )
    centrality = exposure_centrality(exposures.to_numpy(), names=list(exposures.index))
    capital_requirements = systemic_capital_surcharge(
        centrality["systemic_share"], base_ratio=0.08, surcharge_scale=0.04
    )
    capital_result = capital_adequacy(
        capital, exposures.sum(axis=1) + 100.0, minimum_ratio=float(capital_requirements.mean())
    )
    systemic_holdings = np.array([[100.0, 40.0], [80.0, 20.0], [30.0, 90.0], [50.0, 50.0], [25.0, 75.0]])
    stress = external_asset_stress(
        holdings=systemic_holdings,
        asset_returns=np.array([-0.2, -0.08]),
        capital=capital.to_numpy(),
    )
    firesale = simulate_fire_sale(
        holdings=systemic_holdings,
        capital=capital.to_numpy(),
        initial_shock=np.array([-0.2, -0.08]),
        impact=np.array([0.0005, 0.0004]),
        liquidation_fraction=0.4,
    )
    liquidity_spiral = simulate_liquidity_spiral(
        holdings=systemic_holdings,
        capital=capital.to_numpy(),
        initial_returns=np.array([-0.2, -0.08]),
        market_depth=np.array([2_500.0, 2_500.0]),
        leverage_limit=3.0,
        liquidation_speed=0.35,
    )
    systemic_scenarios = run_systemic_stress_scenarios(
        systemic_holdings,
        pd.DataFrame(
            [[-0.05, -0.02], [-0.20, -0.08], [-0.75, -0.45]],
            index=["mild", "recession", "crisis"],
            columns=["asset_1", "asset_2"],
        ),
        capital.to_numpy(),
        scenario_probabilities=pd.Series([0.65, 0.25, 0.10], index=["mild", "recession", "crisis"]),
    )
    systemic_mc = simulate_systemic_monte_carlo(
        systemic_holdings,
        capital.to_numpy(),
        mean_returns=np.array([-0.04, -0.02]),
        covariance=np.array([[0.18**2, 0.18 * 0.12 * 0.35], [0.18 * 0.12 * 0.35, 0.12**2]]),
        simulations=2_000,
        confidence=0.95,
        institution_names=list(capital.index),
        asset_names=["asset_1", "asset_2"],
        seed=seed,
    )

    summaries: dict[str, dict[str, float | int | bool]] = {
        "options": {
            "quotes": int(len(option_chain)),
            "option_chain_valid": bool(option_chain_validation.is_valid),
            "sabr_objective": float(sabr_calibration.objective_value),
            "heston_fourier": float(heston_fourier),
            "heston_monte_carlo": float(heston_mc),
            "heston_calibration_objective": float(heston_calibration.objective_value),
            "heston_calibrated_v0": float(heston_calibration.parameters["v0"]),
            "bates_calibration_objective": float(bates_calibration.objective_value),
            "bates_calibrated_jump_intensity": float(bates_calibration.parameters["jump_intensity"]),
            "atm_delta": float(greeks.delta),
            "local_vol_mean": float(np.nanmean(local_vol)),
            "svi_objective": float(svi_calibration.objective_value),
            "pricing_rmse": float(pricing_diagnostics.rmse),
            "sabr_surface_mean_objective": float(sabr_surface.mean_objective),
            "antithetic_standard_error": float(antithetic.standard_error),
            "control_variate_standard_error": float(control_variate.standard_error),
            "density_mass": float(density.mass),
            "density_mean_strike": float(density.mean_strike),
            "delta_hedge_pnl": float(hedge.final_pnl),
            "ssvi_no_arbitrage_passes": bool(ssvi_check.passes),
            "ssvi_mean_volatility": float(np.mean(ssvi_vols)),
            "option_stress_worst_pnl": float(option_stress.scenario_values["pnl"].min()),
            "option_book_delta": float(option_stress.greek_exposures["delta"]),
        },
        "market_making": {
            "final_pnl": market_making.final_pnl,
            "max_inventory_abs": market_making.max_inventory_abs,
            "rows": int(len(market_making.history)),
            "book_final_pnl": float(book_market_making.final_pnl),
            "book_fill_rate": float(book_market_making.fill_rate),
            "book_average_spread": float(book_market_making.average_spread),
            "book_max_inventory_abs": float(book_market_making.max_inventory_abs),
            "expected_bid_value": float(expected_bid_value),
            "queue_filled": bool(queue.filled),
            "average_signed_move": float(toxicity.average_signed_move),
            "mean_imbalance": float(np.mean(imbalance)),
            "mean_vpin": float(vpin.mean()) if len(vpin) else 0.0,
            "calibrated_fill_intensity": float(fill_calibration.base_intensity),
            "inventory_penalty": float(inventory_report.inventory_penalty),
            "inventory_pnl_correlation": float(inventory_report.pnl_inventory_correlation),
            "latency_total_slippage": float(latency_report.total_slippage),
            "latency_adverse_fill_rate": float(latency_report.adverse_fill_rate),
            "path_fill_rate": float(path_market_making.fill_rate),
            "path_total_slippage": float(path_market_making.total_slippage),
            "path_final_pnl": float(path_market_making.final_pnl),
            "path_spread_capture": float(path_pnl_attribution.spread_capture),
            "path_inventory_mark_to_market": float(path_pnl_attribution.inventory_mark_to_market),
            "hawkes_event_count": int(hawkes_flow.event_count),
            "hawkes_branching_ratio": float(hawkes_flow.branching_ratio),
            "hawkes_order_flow_imbalance": float(hawkes_flow.order_flow_imbalance),
            "hawkes_realized_intensity": float(hawkes_flow.realized_intensity),
        },
        "rl_trading": {
            "total_return": rl_backtest.total_return,
            "max_drawdown": rl_backtest.max_drawdown,
            "sharpe": rl_backtest.sharpe,
            "summary_sharpe": float(rl_summary["sharpe"]),
            "risk_limited_max_drawdown": float(risk_limited_backtest.max_drawdown),
            "volatility_target_weight": float(vol_target),
            "best_constant_weight": float(policy_search.best_weight),
            "q_learning_last_reward": float(q_learning.episode_rewards.iloc[-1]),
            "policy_gradient_last_reward": float(policy_gradient.episode_rewards.iloc[-1]),
            "constrained_pg_last_adjusted_reward": float(constrained_policy_gradient.adjusted_episode_rewards.iloc[-1]),
            "constrained_pg_drawdown_lambda": float(constrained_policy_gradient.lagrange_multipliers["drawdown"]),
            "constrained_pg_turnover_violation": float(
                constrained_policy_gradient.constraint_history["turnover_violation"].iloc[-1]
            ),
            "deep_q_last_reward": float(deep_q.episode_rewards.iloc[-1]),
            "deep_q_last_loss": float(deep_q.training_losses.iloc[-1]),
            "deep_q_initial_action": float(deep_q.policy(TradingEnv(benchmark_prices).reset())),
            "walk_forward_q_return": float(q_walk_forward.mean_test_return),
            "portfolio_constant_return": float(portfolio_constant.total_return),
            "portfolio_momentum_return": float(portfolio_momentum.total_return),
            "portfolio_momentum_turnover": float(portfolio_momentum.average_turnover),
            "portfolio_momentum_max_drawdown": float(portfolio_momentum.max_drawdown),
        },
        "factor_risk": {
            "assets": int(factor_result.exposures.shape[0]),
            "price_panel_valid": bool(price_panel_validation.is_valid),
            "price_panel_return_rows": int(len(price_panel_returns)),
            "factors": int(factor_result.exposures.shape[1]),
            "pca_variance_first3": float(pca_result.explained_variance_ratio.sum()),
            "style_factor_count": int(style_result.factor_returns.shape[1]),
            "cross_sectional_factor_count": int(cross_sectional_result.factor_returns.shape[1]),
            "macro_factor_count": int(macro_model.factor_returns.shape[1]),
            "macro_average_r_squared": float(macro_model.r_squared.mean()),
            "macro_stress_pnl": float(macro_stress_pnl),
            "factor_oos_r_squared": float(factor_validation.mean_oos_r_squared),
            "factor_residual_correlation": float(factor_validation.mean_abs_residual_correlation),
            "factor_specific_risk_share": float(factor_validation.mean_specific_risk_share),
            "factor_covariance_condition": float(factor_validation.max_covariance_condition_number),
            "sector_factor_count": int(sector_exposures.shape[1]),
            "neutralized_style_exposure_norm": float(np.linalg.norm(neutralized_style_exposure.to_numpy())),
            "mimicking_size_exposure": float(mimicking.loc["size"] @ style_exposures["size"]),
            "style_residual_std": float(style_result.residuals.stack().std(ddof=1)),
            "total_variance": float(attribution.total_variance),
            "factor_variance": float(attribution.factor_variance),
            "ledoit_wolf_trace": float(np.trace(lw_covariance)),
        },
        "portfolio": {
            "min_var_first_weight": float(min_var[0]),
            "mean_var_first_weight": float(mean_var[0]),
            "risk_parity_first_weight": float(risk_parity[0]),
            "cdar_first_weight": float(cdar.weights.iloc[0]),
            "cdar_objective": float(cdar.objective_value),
            "robust_first_weight": float(robust[0]),
            "ellipsoid_robust_first_weight": float(ellipsoid_robust.weights[0]),
            "ellipsoid_worst_case_return": float(ellipsoid_robust.worst_case_return),
            "ellipsoid_uncertainty_penalty": float(ellipsoid_robust.uncertainty_penalty),
            "bayesian_first_weight": float(bayesian_weights[0]),
            "turnover_first_weight": float(turnover_weights[0]),
            "risk_budget_first_weight": float(budget_weights[0]),
            "risk_budget_max_error": float(np.max(np.abs(budget_contributions - 1.0 / len(budget_contributions)))),
            "largest_cvar_contribution": float(cvar_contrib.max()),
            "conditional_drawdown_at_risk": float(drawdown_summary.conditional_drawdown_at_risk),
            "posterior_mean_first": float(bayesian_posterior.posterior_mean[0]),
            "black_litterman_first_return": float(black_litterman.posterior_returns[0]),
            "worst_stress_pnl": float(portfolio_stress.scenario_pnl.min()),
            "frontier_points": int(len(frontier.points)),
            "var_exception_rate": float(var_backtest.exception_rate),
            "christoffersen_p_value": float(christoffersen.conditional_coverage_p_value),
            "var_traffic_light_green": bool(traffic_light.zone == "green"),
        },
        "rough_vol": {
            "terminal_spot_mean": float(np.mean(rough_spots[:, -1])),
            "terminal_variance_mean": float(np.mean(rough_variance[:, -1])),
            "estimated_hurst": float(roughness.hurst),
            "atm_skew_power_law_hurst": float(atm_skew_fit.hurst),
            "atm_skew_power_law_r_squared": float(atm_skew_fit.r_squared),
            "rough_proxy_hurst": float(rough_proxy.params.hurst),
            "rough_proxy_eta": float(rough_proxy.params.eta),
            "rough_proxy_xi0": float(rough_proxy.params.xi0),
            "rough_proxy_objective": float(rough_proxy.objective_value),
            "rough_option_price": float(rough_option),
        },
        "stat_arb": {
            "cointegration_p_value": float(coint.p_value),
            "ou_half_life": float(ou["half_life"]),
            "network_edges": int(network.adjacency.to_numpy().sum()),
            "ranked_pair_count": int(len(ranked_pairs)),
            "top_pair_score": float(ranked_pairs[0].score) if ranked_pairs else 0.0,
            "top_pair_gross_exposure": float(top_pair_weights.abs().sum()) if len(top_pair_weights) else 0.0,
            "pair_portfolio_pnl": float(pair_portfolio.total_pnl) if pair_portfolio is not None else 0.0,
            "pair_portfolio_turnover": float(pair_portfolio.turnover) if pair_portfolio is not None else 0.0,
            "pair_portfolio_max_drawdown": float(pair_portfolio.max_drawdown) if pair_portfolio is not None else 0.0,
            "johansen_basket_pnl": float(johansen_basket_backtest.total_pnl),
            "johansen_basket_turnover": float(johansen_basket_backtest.turnover),
            "johansen_basket_max_drawdown": float(johansen_basket_backtest.max_drawdown),
            "active_signal_count": int(np.count_nonzero(signal)),
            "spread_strategy_pnl": float(spread_backtest.total_pnl),
            "johansen_spread_std": float(johansen_spread.std(ddof=1)),
            "kalman_last_hedge_ratio": float(kalman.states["hedge_ratio"].iloc[-1]),
            "dynamic_spread_pnl": float(dynamic_backtest.total_pnl),
        },
        "credit": {
            "merton_default_probability": float(default_probability),
            "spread_curve_valid": bool(spread_curve_validation.is_valid),
            "calibrated_asset_value": float(merton_calibration.asset_value),
            "five_year_survival": float(hazard_curve.survival(5.0)),
            "five_year_spread": float(hazard_curve.spread(5.0)),
            "risky_bond_price": float(risky_bond),
            "risky_coupon_bond_price": float(risky_coupon),
            "coupon_bond_spread_dv01": float(spread_sensitivity.spread_dv01),
            "coupon_bond_spread_duration": float(spread_sensitivity.spread_duration),
            "cds_par_spread": float(cds_spread),
            "fitted_hazard_rate": float(hazard_fit.hazard_rate),
            "logistic_hazard_stressed": float(logistic_hazard.predict_hazard(stressed_credit_covariates).iloc[0]),
            "cox_hazard_ratio_stressed": float(cox_hazard.hazard_ratio(stressed_credit_covariates).iloc[0]),
            "cox_three_year_default_probability": float(
                cox_hazard.default_probability(stressed_credit_covariates, 3.0).iloc[0]
            ),
            "cir_default_probability": float(cir_intensity.default_probability),
            "cir_mean_survival_probability": float(cir_intensity.mean_survival_probability),
            "cir_terminal_intensity": float(cir_intensity.mean_terminal_intensity),
            "five_year_migration_pd": float(five_year_migration_pd),
            "portfolio_expected_loss": float(credit_losses.expected_loss),
            "portfolio_expected_shortfall": float(credit_losses.expected_shortfall),
            "mezzanine_tranche_expected_loss": float(mezzanine_tranche.expected_loss_rate),
            "counterparty_cva": float(cva.cva),
            "wrong_way_cva": float(wrong_way_cva.cva),
            "counterparty_peak_pfe": float(counterparty_profile.peak_pfe),
            "distance_to_default": float(dd.distance_to_default),
            "expected_default_frequency": float(dd.expected_default_frequency),
        },
        "surface_arbitrage": {
            "violations": int(len(arbitrage_violations)),
            "bound_violations": int(sum(violation.kind == "bounds" for violation in arbitrage_violations)),
            "vertical_violations": int(sum(violation.kind == "vertical" for violation in arbitrage_violations)),
            "interpolation_stability_passes": bool(surface_stability.passes),
            "interpolation_dense_violations": int(len(surface_stability.arbitrage_violations)),
            "interpolation_local_vol_nan_fraction": float(surface_stability.local_vol_nan_fraction),
            "interpolation_max_vol_step": float(surface_stability.max_implied_vol_step),
            "maturities": int(len(pivot_prices.index)),
            "strikes": int(len(pivot_prices.columns)),
            "repair_objective": float(surface_repair.objective_value),
            "violations_after_repair": int(len(surface_repair.violations_after)),
        },
        "systemic": {
            "defaults_after_contagion": int(contagion.defaulted.sum()),
            "clearing_defaults": int(clearing.defaulted.sum()),
            "spectral_radius": float(eigenvalue_stability(exposures.to_numpy(), capital.to_numpy())),
            "largest_systemic_share": float(centrality["systemic_share"].max()),
            "stress_defaults": int(stress.defaulted.sum()),
            "firesale_defaults": int(firesale.defaulted.sum()),
            "liquidity_spiral_rounds": int(liquidity_spiral.rounds),
            "liquidity_spiral_min_price": float(liquidity_spiral.prices.min()),
            "scenario_expected_shortfall": float(systemic_scenarios.expected_shortfall),
            "scenario_worst_default_count": int(systemic_scenarios.scenario_results["default_count"].max()),
            "monte_carlo_value_at_risk": float(systemic_mc.value_at_risk),
            "monte_carlo_expected_shortfall": float(systemic_mc.expected_shortfall),
            "monte_carlo_max_default_probability": float(systemic_mc.max_default_probability),
            "debt_rank_impact": float(debt_rank_result.total_impact),
            "capital_shortfall": float(capital_result.total_shortfall),
        },
    }
    return DemoSuiteResult(summaries)
