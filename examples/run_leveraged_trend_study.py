from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantlab.research.leveraged_trend import LeveragedTrendConfig, load_leveraged_etf_csv, run_leveraged_trend_study


def run_study(
    data_path: str | Path = "data/real/leveraged_etf_adjusted.csv",
    output: str | Path | None = None,
    config_path: str | Path | None = None,
) -> dict[str, float | int | bool]:
    prices = load_leveraged_etf_csv(data_path)
    config = _load_config(config_path) if config_path is not None else None
    result = run_leveraged_trend_study(prices, config)
    payload: dict[str, float | int | bool] = {
        "observations": len(prices),
        "holdout_observations": len(result.history),
        "holdout_cagr": float(result.metrics["cagr"]),
        "holdout_sharpe": float(result.metrics["sharpe"]),
        "holdout_max_drawdown": float(result.metrics["max_drawdown"]),
        "holdout_target_met": result.holdout_target_met,
        "bootstrap_probability_target_met": float(result.bootstrap["probability_cagr_at_least_target"]),
    }
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(_render_markdown(result), encoding="utf-8")
    return payload


def _render_markdown(result) -> str:
    target = result.config.target_cagr
    status = "MET" if result.holdout_target_met else "NOT MET"
    lines = [
        "# Leveraged Trend Holdout Study",
        "",
        "A validation-selected daily trend and volatility-targeting strategy using real TQQQ, QQQ, and BIL adjusted prices.",
        "",
        f"**Historical holdout threshold ({target:.0%} CAGR): {status}. This is not a forecast or a guaranteed return.**",
        "",
        "## Selected Parameters",
        "",
    ]
    for key, value in result.selected_parameters.items():
        lines.append(f"- **{key.replace('_', ' ').title()}**: {_format_cell(value)}")
    lines.extend(["", "## Holdout Tear Sheet", ""])
    for key, value in result.metrics.items():
        lines.append(f"- **{key.replace('_', ' ').title()}**: {_format_cell(value)}")
    lines.extend(
        [
            "",
            "## Bootstrap Uncertainty",
            "",
            *[
                f"- **{key.replace('_', ' ').title()}**: {_format_cell(value)}"
                for key, value in result.bootstrap.items()
            ],
            "",
            "## Calendar Returns",
            "",
            _table(result.annual_returns.reset_index()),
            "",
            "## Cost Sensitivity",
            "",
            _table(result.cost_sensitivity),
            "",
            "## Parameter Sensitivity Summary",
            "",
            f"{int(result.parameter_sensitivity['target_met'].sum())} of {len(result.parameter_sensitivity)} prespecified parameter combinations exceeded the historical holdout target.",
            "",
            _table(result.parameter_sensitivity),
            "",
            "## Validation Selection Grid",
            "",
            _table(result.selection_grid),
            "",
        ]
    )
    return "\n".join(lines)


def _load_config(path: str | Path) -> LeveragedTrendConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for field in ("moving_average_candidates", "band_candidates", "target_volatility_candidates"):
        if field in payload:
            payload[field] = tuple(payload[field])
    return LeveragedTrendConfig(**payload)


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
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (float, int)):
        return f"{value:.6g}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the real-data leveraged ETF trend holdout study.")
    parser.add_argument("--data", default="data/real/leveraged_etf_adjusted.csv")
    parser.add_argument("--output", default="reports/leveraged_trend_study.md")
    parser.add_argument("--config", default="config/leveraged_trend.json")
    args = parser.parse_args()
    print(json.dumps(run_study(args.data, args.output, args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
