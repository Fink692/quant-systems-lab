from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantlab.research.valuation_regime import (
    ValuationRegimeConfig,
    load_shiller_sp500_csv,
    run_valuation_regime_walk_forward,
)


def run_study(
    data_path: str | Path = "data/real/shiller_sp500_monthly.csv",
    output: str | Path | None = None,
    config_path: str | Path | None = None,
) -> dict[str, float | int]:
    data = load_shiller_sp500_csv(data_path)
    config = _load_config(config_path) if config_path is not None else None
    result = run_valuation_regime_walk_forward(data, config)
    payload = {
        "observations": int(len(data)),
        "out_of_sample_months": int(len(result.history)),
        "folds": int(len(result.folds)),
        "strategy_cagr": float(result.tear_sheet.metrics["cagr"]),
        "benchmark_cagr": float(result.tear_sheet.metrics["benchmark_cagr"]),
        "strategy_sharpe": float(result.tear_sheet.metrics["sharpe"]),
        "max_drawdown": float(result.tear_sheet.metrics["max_drawdown"]),
        "beta": float(result.tear_sheet.metrics["beta"]),
        "alpha_annual": float(result.tear_sheet.metrics["alpha_annual"]),
        "average_turnover": float(result.tear_sheet.metrics["average_turnover"]),
        "total_cost": float(result.tear_sheet.metrics["total_cost"]),
        "positive_cost_scenarios": int(result.robustness.positive_cost_scenarios),
        "deflated_sharpe_probability": float(result.diagnostics.overfitting_metrics["deflated_sharpe_probability"]),
        "probability_backtest_overfitting": float(
            result.diagnostics.overfitting_metrics["probability_backtest_overfitting"]
        ),
    }
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(_render_markdown(result), encoding="utf-8")
    return payload


def _render_markdown(result) -> str:
    lines = [
        "# Valuation-Regime Walk-Forward Study",
        "",
        "Real monthly S&P 500, dividend, rate, and Shiller CAPE/PE10 data from the DataHub `s-and-p-500` package.",
        "",
        "## Tear Sheet",
        "",
    ]
    for key, value in result.tear_sheet.metrics.items():
        lines.append(f"- **{key.replace('_', ' ').title()}**: {value:.6g}")
    lines.extend(
        [
            "",
            "## Walk-Forward Folds",
            "",
            _table(result.folds),
            "",
            "## Cost Robustness",
            "",
            _table(result.robustness.grid),
            "",
            "## Risk-Matched and Simple Baselines",
            "",
            _table(result.diagnostics.baseline_comparison.reset_index()),
            "",
            "## Bootstrap Confidence Intervals",
            "",
            _table(result.diagnostics.bootstrap_confidence_intervals.reset_index()),
            "",
            "## Overfitting Diagnostics",
            "",
        ]
    )
    for key, value in result.diagnostics.overfitting_metrics.items():
        lines.append(f"- **{key.replace('_', ' ').title()}**: {value:.6g}")
    lines.extend(
        [
            "",
            "## Parameter Stability",
            "",
            _table(result.diagnostics.parameter_stability),
            "",
        ]
    )
    lines.extend(
        [
            "## Regime Breakdown",
            "",
            _table(result.tear_sheet.regime_breakdown.reset_index()),
            "",
            "## Stress Tests",
            "",
            _table(result.tear_sheet.stress_tests.reset_index()),
            "",
        ]
    )
    return "\n".join(lines)


def _load_config(path: str | Path) -> ValuationRegimeConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    tuple_fields = {"cheap_percentiles", "expensive_percentiles"}
    for field in tuple_fields & payload.keys():
        payload[field] = tuple(payload[field])
    return ValuationRegimeConfig(**payload)


def _table(frame) -> str:
    display = frame.copy()
    for column in display.columns:
        display[column] = display[column].map(_format_cell)
    headers = [str(column) for column in display.columns]
    rows = display.astype(str).values.tolist()
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _format_cell(value) -> str:
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the real-data S&P 500 valuation-regime walk-forward study.")
    parser.add_argument("--data", default="data/real/shiller_sp500_monthly.csv")
    parser.add_argument("--output", default="reports/valuation_regime_study.md")
    parser.add_argument("--config", default="config/valuation_regime.json")
    args = parser.parse_args()
    print(json.dumps(run_study(args.data, args.output, args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
