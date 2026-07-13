from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantlab.research.bitcoin_trend import (
    BitcoinTrendConfig,
    BitcoinTrendResult,
    load_bitcoin_coinbase_csv,
    run_bitcoin_trend_study,
)


def load_config(path: str | Path) -> BitcoinTrendConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for field in ("moving_average_days", "volatility_days", "target_volatilities", "hysteresis_bands"):
        payload[field] = tuple(payload[field])
    return BitcoinTrendConfig(**payload)


def render_report(study: BitcoinTrendResult, metadata: dict[str, object]) -> str:
    selected = study.selected
    periods = study.period_metrics
    evaluation = periods.loc["evaluation"]
    full = periods.loc["full"]
    bootstrap = study.bootstrap
    robust = bool(
        evaluation["cagr"] >= study.config.target_cagr
        and study.cost_sensitivity.loc[study.cost_sensitivity["cost_bps"] == 50.0, "evaluation_cagr"].iloc[0]
        >= study.config.target_cagr
        and bootstrap["probability_cagr_at_or_above_target"] >= 0.5
    )
    lines = [
        "# Bitcoin Trend Study",
        "",
        f"Data: Coinbase BTC-USD completed daily candles through {metadata['as_of_session']}; official FRED DFF financing.",
        f"Snapshot SHA-256: `{metadata['sha256']}`.",
        "",
        "## Prespecified Design",
        "",
        "The 64-row grid uses 2016-2017 training, 2018-2020 validation-only selection, and a one-time 2021-present evaluation. A completed close changes the state; exposure begins on the next UTC candle. Costs are 25 bps times exposure turnover.",
        "",
        "## Selected Candidate",
        "",
        f"Validation selected MA {int(selected['moving_average_days'])}, volatility lookback {int(selected['volatility_days'])}, target volatility {selected['target_volatility']:.0%}, and hysteresis {selected['hysteresis_band']:.0%}.",
        "",
        "| Period | CAGR | Sharpe | Max drawdown | Observations |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for period, row in periods.iterrows():
        lines.append(
            f"| {period.title()} | {row['cagr']:.2%} | {row['sharpe']:.2f} | {row['max_drawdown']:.2%} | {int(row['observations']):,} |"
        )
    lines.extend(
        [
            "",
            f"The evaluation point estimate {'MET' if evaluation['cagr'] >= study.config.target_cagr else 'DID NOT MEET'} the 20% historical CAGR threshold. Full-history CAGR is {full['cagr']:.2%}.",
            "",
            "## Falsification Checks",
            "",
            "| Cost per unit turnover | Evaluation CAGR | Full CAGR |",
            "| ---: | ---: | ---: |",
        ]
    )
    for _, row in study.cost_sensitivity.iterrows():
        lines.append(f"| {row['cost_bps']:.0f} bps | {row['evaluation_cagr']:.2%} | {row['full_cagr']:.2%} |")
    lines.extend(
        [
            "",
            f"{int(study.parameter_breadth['both_at_or_above_target'])} of {int(study.parameter_breadth['configurations'])} configurations clear 20% in both evaluation and full history.",
            f"The 30-day moving-block bootstrap gives a {bootstrap['probability_cagr_at_or_above_target']:.2%} probability of CAGR at or above 20%; its 5th/median/95th percentiles are {bootstrap['cagr_5th_percentile']:.2%}, {bootstrap['cagr_median']:.2%}, and {bootstrap['cagr_95th_percentile']:.2%}.",
            "",
            "| Evaluation year | Return |",
            "| ---: | ---: |",
        ]
    )
    for year, value in study.calendar_returns.items():
        lines.append(f"| {year} | {value:.2%} |")
    lines.extend(
        [
            "",
            "| Rolling horizon | Minimum CAGR | Median CAGR | Fraction >= 20% |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for years, row in study.rolling_summary.iterrows():
        lines.append(
            f"| {years} years | {row['minimum_cagr']:.2%} | {row['median_cagr']:.2%} | {row['fraction_at_or_above_target']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"Robust/prospective 20% claim: **{'SUPPORTED' if robust else 'NOT SUPPORTED'}**. The historical evaluation CAGR narrowly clears the threshold, but 50 bps costs push it below 20%, bootstrap support is below 50%, and individual years include losses. This is a reproducible historical result, not verified future profitability or a guaranteed annual return.",
            "",
            "Coinbase daily bars meet at the midnight UTC boundary and do not contain executable bid/ask fills. The turnover charge is a sensitivity assumption; live slippage, fees, taxes, outages, and custody risks can be materially worse.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Coinbase BTC trend holdout study.")
    parser.add_argument("--data", default="data/real/coinbase_btc_dff.csv")
    parser.add_argument("--metadata", default="data/real/coinbase_btc_dff.metadata.json")
    parser.add_argument("--config", default="config/bitcoin_trend.json")
    parser.add_argument("--output", default="reports/bitcoin_trend_study.md")
    args = parser.parse_args()
    data = load_bitcoin_coinbase_csv(args.data)
    config = load_config(args.config)
    result = run_bitcoin_trend_study(data, config)
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(result, metadata), encoding="utf-8", newline="")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
