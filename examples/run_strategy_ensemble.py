from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantlab.research.bitcoin_trend import load_bitcoin_coinbase_csv, run_frozen_bitcoin_candidate
from quantlab.research.defensive_momentum import (
    load_defensive_momentum_csv,
    run_frozen_defensive_monthly_candidate,
)
from quantlab.research.strategy_ensemble import FixedEnsembleConfig, FixedEnsembleResult, run_fixed_strategy_ensemble


def load_config(path: str | Path) -> FixedEnsembleConfig:
    return FixedEnsembleConfig(**json.loads(Path(path).read_text(encoding="utf-8")))


def render_report(
    result: FixedEnsembleResult,
    bitcoin_metadata: dict[str, object],
    defensive_metadata: dict[str, object],
) -> str:
    evaluation = result.period_metrics.loc["evaluation"]
    full = result.period_metrics.loc["full"]
    bootstrap = result.bootstrap
    lines = [
        "# Fixed Cross-Asset Strategy Ensemble",
        "",
        "Status: **exploratory historical evidence; frozen for prospective evaluation**.",
        "",
        f"Bitcoin snapshot: `{bitcoin_metadata['sha256']}` through {bitcoin_metadata['as_of_session']}.",
        f"Defensive snapshot: `{defensive_metadata['sha256']}` through {defensive_metadata['as_of_session']}.",
        "",
        "## Frozen Rule",
        "",
        "Allocate 50% to `bitcoin-trend-v1` and 50% to `defensive-momentum-monthly-v1`. Rebalance on the first observed defensive-market session of each month and charge 10 bps times two-way ensemble turnover. Component returns already include their published trading costs, financing, and leverage drag.",
        "",
        "No ensemble weight was selected on performance. The 50/50 allocation was fixed before this combined evaluation as the neutral diversification rule.",
        "",
        "## Historical Results",
        "",
        "| Period | CAGR | Sharpe | Max drawdown | Observations |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Evaluation (2021-present) | {evaluation['cagr']:.2%} | {evaluation['sharpe']:.2f} | {evaluation['max_drawdown']:.2%} | {int(evaluation['observations']):,} |",
        f"| Full (2016-present) | {full['cagr']:.2%} | {full['sharpe']:.2f} | {full['max_drawdown']:.2%} | {int(full['observations']):,} |",
        "",
        f"Evaluation component-return correlation is {result.component_correlation.iloc[0, 1]:.3f}.",
        "",
        "## Cost Sensitivity",
        "",
        "| Ensemble turnover cost | Evaluation CAGR | Full CAGR |",
        "| ---: | ---: | ---: |",
    ]
    for _, row in result.cost_sensitivity.iterrows():
        lines.append(f"| {row['cost_bps']:.0f} bps | {row['evaluation_cagr']:.2%} | {row['full_cagr']:.2%} |")
    lines.extend(
        [
            "",
            "## Allocation Diagnostics",
            "",
            "These rows are sensitivity diagnostics, not a selection grid.",
            "",
            "| Bitcoin weight | Defensive weight | Evaluation CAGR | Sharpe | Max drawdown |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in result.allocation_diagnostics.iterrows():
        lines.append(
            f"| {row['bitcoin_weight']:.0%} | {row['defensive_weight']:.0%} | {row['cagr']:.2%} | {row['sharpe']:.2f} | {row['max_drawdown']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Path Robustness",
            "",
            f"The 30-day moving-block bootstrap gives a {bootstrap['probability_cagr_at_or_above_target']:.2%} probability of CAGR at or above 20%. Its 5th/median/95th percentiles are {bootstrap['cagr_5th_percentile']:.2%}, {bootstrap['cagr_median']:.2%}, and {bootstrap['cagr_95th_percentile']:.2%}.",
            "",
            "| Evaluation year | Return |",
            "| ---: | ---: |",
        ]
    )
    for year, value in result.calendar_returns.items():
        lines.append(f"| {year} | {value:.2%} |")
    lines.extend(
        [
            "",
            "| Rolling horizon | Minimum CAGR | Median CAGR | Fraction >= 20% |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for years, row in result.rolling_summary.iterrows():
        lines.append(
            f"| {years} years | {row['minimum_cagr']:.2%} | {row['median_cagr']:.2%} | {row['fraction_at_or_above_target']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "The historical point estimate and most resampled paths clear 20%, but this is not untouched ensemble validation. Both component evaluation histories were visible before the combination was tested, and 2022 lost money. The ensemble is therefore frozen as a prospective candidate, not represented as verified future profitability or a 20% return every year.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the fixed Bitcoin/defensive strategy ensemble.")
    parser.add_argument("--bitcoin-data", default="data/real/coinbase_btc_dff.csv")
    parser.add_argument("--bitcoin-metadata", default="data/real/coinbase_btc_dff.metadata.json")
    parser.add_argument("--defensive-data", default="data/real/defensive_momentum_ohlc.csv")
    parser.add_argument("--defensive-metadata", default="data/real/defensive_momentum_ohlc.metadata.json")
    parser.add_argument("--config", default="config/strategy_ensemble.json")
    parser.add_argument("--output", default="reports/strategy_ensemble.md")
    args = parser.parse_args()
    bitcoin = run_frozen_bitcoin_candidate(load_bitcoin_coinbase_csv(args.bitcoin_data))["strategy_return"]
    defensive = run_frozen_defensive_monthly_candidate(load_defensive_momentum_csv(args.defensive_data))[
        "strategy_return"
    ]
    result = run_fixed_strategy_ensemble(bitcoin, defensive, load_config(args.config))
    bitcoin_metadata = json.loads(Path(args.bitcoin_metadata).read_text(encoding="utf-8"))
    defensive_metadata = json.loads(Path(args.defensive_metadata).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(result, bitcoin_metadata, defensive_metadata), encoding="utf-8", newline="")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
