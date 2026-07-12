from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quantlab.data.loaders import load_price_panel_csv, returns_from_prices
from quantlab.portfolio.backtest import static_weight_backtest
from quantlab.portfolio.optimization import min_variance_weights, risk_parity_weights
from quantlab.risk.covariance import ledoit_wolf_covariance
from quantlab.risk.var import historical_cvar, historical_var


def analyze_price_panel(path: str | Path, date_column: str = "date") -> dict[str, float | int | dict[str, float]]:
    prices = load_price_panel_csv(path, date_column=date_column)
    returns = returns_from_prices(prices)
    covariance = ledoit_wolf_covariance(returns)
    covariance_values = (
        covariance.to_numpy() if hasattr(covariance, "to_numpy") else np.asarray(covariance, dtype=float)
    )

    min_var = min_variance_weights(covariance_values)
    risk_parity = risk_parity_weights(covariance_values)
    portfolio_returns = returns.to_numpy() @ min_var
    backtest = static_weight_backtest(returns, min_var, transaction_cost_bps=1.0)

    return {
        "rows": int(len(prices)),
        "assets": int(prices.shape[1]),
        "return_rows": int(len(returns)),
        "historical_var_95": float(historical_var(portfolio_returns, confidence=0.95)),
        "historical_cvar_95": float(historical_cvar(portfolio_returns, confidence=0.95)),
        "min_variance_total_return": float(backtest.total_return),
        "min_variance_max_drawdown": float(backtest.max_drawdown),
        "min_variance_weights": _weights_payload(prices.columns, min_var),
        "risk_parity_weights": _weights_payload(prices.columns, risk_parity),
    }


def _weights_payload(labels, weights: np.ndarray) -> dict[str, float]:
    return {str(label): float(weight) for label, weight in zip(labels, weights)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a real-data-compatible wide price-panel CSV.")
    parser.add_argument(
        "--prices",
        default="examples/price_panel_template.csv",
        help="CSV with a date column and one price column per asset.",
    )
    parser.add_argument("--date-column", default="date")
    args = parser.parse_args()
    print(json.dumps(analyze_price_panel(args.prices, args.date_column), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
