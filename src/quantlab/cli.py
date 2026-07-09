from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from quantlab.data.loaders import returns_from_prices, validate_credit_spread_curve, validate_option_chain, validate_price_panel
from quantlab.data.synthetic import synthetic_factor_panel, synthetic_option_chain
from quantlab.data.synthetic import synthetic_credit_spreads
from quantlab.market_making.avellaneda_stoikov import AvellanedaStoikovParams
from quantlab.market_making.simulator import simulate_market_maker
from quantlab.options.black_scholes import black_scholes_price, implied_volatility
from quantlab.options.surface import build_volatility_surface_from_chain
from quantlab.portfolio.backtest import static_weight_backtest
from quantlab.portfolio.optimization import min_variance_weights
from quantlab.portfolio.robust import robust_mean_variance_weights
from quantlab.reporting.markdown import save_demo_report
from quantlab.risk.backtesting import basel_traffic_light, christoffersen_var_backtest, kupiec_var_backtest
from quantlab.risk.var import component_var, gaussian_var, historical_cvar, historical_var
from quantlab.workflows.demo_suite import run_full_demo


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quantlab", description="Run quant systems lab workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    price_parser = subparsers.add_parser("price-option", help="Price a Black-Scholes European option.")
    price_parser.add_argument("--spot", type=float, required=True)
    price_parser.add_argument("--strike", type=float, required=True)
    price_parser.add_argument("--maturity", type=float, required=True)
    price_parser.add_argument("--rate", type=float, default=0.03)
    price_parser.add_argument("--volatility", type=float, required=True)
    price_parser.add_argument("--dividend", type=float, default=0.0)
    price_parser.add_argument("--option-type", choices=["call", "put"], default="call")

    iv_parser = subparsers.add_parser("implied-vol", help="Invert a Black-Scholes European option price.")
    iv_parser.add_argument("--price", type=float, required=True)
    iv_parser.add_argument("--spot", type=float, required=True)
    iv_parser.add_argument("--strike", type=float, required=True)
    iv_parser.add_argument("--maturity", type=float, required=True)
    iv_parser.add_argument("--rate", type=float, default=0.03)
    iv_parser.add_argument("--dividend", type=float, default=0.0)
    iv_parser.add_argument("--option-type", choices=["call", "put"], default="call")

    mm_parser = subparsers.add_parser("market-maker-demo", help="Run a deterministic market-making simulation.")
    mm_parser.add_argument("--seed", type=int, default=7)
    mm_parser.add_argument("--steps", type=int, default=100)

    demo_parser = subparsers.add_parser("demo-suite", help="Run all project-family smoke workflows.")
    demo_parser.add_argument("--seed", type=int, default=7)

    report_parser = subparsers.add_parser("demo-report", help="Write a Markdown report for the full demo suite.")
    report_parser.add_argument("--seed", type=int, default=7)
    report_parser.add_argument("--output", type=Path, required=True)

    surface_parser = subparsers.add_parser("surface-demo", help="Build and summarize a synthetic volatility surface.")
    surface_parser.add_argument("--seed", type=int, default=7)

    risk_parser = subparsers.add_parser("risk-demo", help="Run historical and parametric risk metrics.")
    risk_parser.add_argument("--seed", type=int, default=7)
    risk_parser.add_argument("--confidence", type=float, default=0.95)

    portfolio_parser = subparsers.add_parser("portfolio-demo", help="Run portfolio optimization and backtest examples.")
    portfolio_parser.add_argument("--seed", type=int, default=7)

    data_parser = subparsers.add_parser("data-demo", help="Validate synthetic data schemas and return construction.")
    data_parser.add_argument("--seed", type=int, default=7)

    args = parser.parse_args(argv)

    if args.command == "price-option":
        payload = {
            "price": black_scholes_price(
                args.spot,
                args.strike,
                args.maturity,
                args.rate,
                args.volatility,
                args.dividend,
                args.option_type,
            )
        }
    elif args.command == "implied-vol":
        payload = {
            "implied_volatility": implied_volatility(
                args.price,
                args.spot,
                args.strike,
                args.maturity,
                args.rate,
                args.dividend,
                args.option_type,
            )
        }
    elif args.command == "market-maker-demo":
        result = simulate_market_maker(
            100.0,
            AvellanedaStoikovParams(risk_aversion=0.1, volatility=0.2, order_book_liquidity=1.2, horizon=1.0),
            steps=args.steps,
            seed=args.seed,
        )
        payload = {
            "final_pnl": result.final_pnl,
            "max_inventory_abs": result.max_inventory_abs,
            "rows": int(len(result.history)),
        }
    elif args.command == "demo-suite":
        payload = run_full_demo(seed=args.seed).as_dict()
    elif args.command == "demo-report":
        result = run_full_demo(seed=args.seed)
        output = save_demo_report(result, args.output)
        payload = {"output": str(output), "sections": len(result.as_dict())}
    elif args.command == "surface-demo":
        surface = build_volatility_surface_from_chain(synthetic_option_chain())
        local_vol = surface.local_volatility()
        payload = {
            "maturities": int(len(surface.maturities)),
            "strikes": int(len(surface.strikes)),
            "atm_implied_volatility": surface.implied_volatility(1.0, 100.0),
            "atm_call_price": surface.price(1.0, 100.0, "call"),
            "local_vol_mean": float(np.nanmean(local_vol)),
            "arbitrage_violations": int(len(surface.arbitrage_violations())),
        }
    elif args.command == "risk-demo":
        panel = synthetic_factor_panel(seed=args.seed)
        equal_weights = np.full(panel.asset_returns.shape[1], 1.0 / panel.asset_returns.shape[1])
        portfolio_returns = panel.asset_returns.to_numpy() @ equal_weights
        covariance = panel.asset_returns.cov().to_numpy()
        components = component_var(equal_weights, covariance, confidence=args.confidence)
        var_estimate = historical_var(portfolio_returns, confidence=args.confidence)
        var_series = np.full_like(portfolio_returns, var_estimate)
        kupiec = kupiec_var_backtest(portfolio_returns, var_series, confidence=args.confidence)
        christoffersen = christoffersen_var_backtest(portfolio_returns, var_series, confidence=args.confidence)
        traffic_light = basel_traffic_light(kupiec.exceptions, observations=kupiec.observations, confidence=args.confidence)
        payload = {
            "historical_var": var_estimate,
            "historical_cvar": historical_cvar(portfolio_returns, confidence=args.confidence),
            "gaussian_var": gaussian_var(float(portfolio_returns.mean()), float(portfolio_returns.std(ddof=1)), args.confidence),
            "component_var_sum": float(components.sum()),
            "kupiec_p_value": kupiec.kupiec_p_value,
            "christoffersen_p_value": christoffersen.conditional_coverage_p_value,
            "traffic_light_zone": traffic_light.zone,
            "observations": int(len(portfolio_returns)),
        }
    elif args.command == "portfolio-demo":
        panel = synthetic_factor_panel(seed=args.seed)
        covariance = panel.asset_returns.cov().to_numpy()
        expected_returns = panel.asset_returns.mean().to_numpy()
        min_var = min_variance_weights(covariance)
        robust = robust_mean_variance_weights(expected_returns, covariance, risk_aversion=8.0, uncertainty_penalty=0.05)
        backtest = static_weight_backtest(panel.asset_returns, min_var, transaction_cost_bps=1.0)
        payload = {
            "assets": int(panel.asset_returns.shape[1]),
            "min_var_first_weight": float(min_var[0]),
            "robust_first_weight": float(robust[0]),
            "backtest_total_return": backtest.total_return,
            "backtest_max_drawdown": backtest.max_drawdown,
        }
    else:
        chain = synthetic_option_chain()
        panel = synthetic_factor_panel(seed=args.seed)
        spread_curve = synthetic_credit_spreads()
        price_returns = returns_from_prices(panel.prices)
        payload = {
            "option_chain_valid": validate_option_chain(chain).is_valid,
            "price_panel_valid": validate_price_panel(panel.prices).is_valid,
            "credit_spread_curve_valid": validate_credit_spread_curve(spread_curve).is_valid,
            "price_return_rows": int(len(price_returns)),
            "assets": int(panel.prices.shape[1]),
        }

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
