from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantlab.research.leveraged_trend import load_leveraged_etf_csv
from quantlab.research.leveraged_trend_stress import (
    LeveragedTrendStressConfig,
    load_qqq_fred_csv,
    run_leveraged_trend_stress,
)


def run_study(
    data_path: str | Path = "data/real/qqq_fred_stress_daily.csv",
    actual_path: str | Path = "data/real/leveraged_etf_adjusted.csv",
    output: str | Path | None = None,
    config_path: str | Path | None = None,
) -> dict[str, float | bool]:
    config = _load_config(config_path) if config_path is not None else None
    actual = load_leveraged_etf_csv(actual_path)["tqqq"]
    result = run_leveraged_trend_stress(load_qqq_fred_csv(data_path), actual, config)
    payload = {
        "full_history_cagr": float(result.period_metrics.loc["full_history", "cagr"]),
        "early_crisis_cagr": float(result.period_metrics.loc["dotcom_and_gfc", "cagr"]),
        "published_holdout_cagr": float(result.period_metrics.loc["published_holdout", "cagr"]),
        "long_history_target_met": result.long_history_target_met,
        "reconstruction_correlation": float(result.reconciliation["daily_return_correlation"]),
    }
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(_render(result), encoding="utf-8")
    return payload


def _render(result) -> str:
    status = "MET" if result.long_history_target_met else "NOT MET"
    lines = [
        "# Leveraged Trend Long-History Falsification",
        "",
        "Real QQQ adjusted closes and FRED DFF rates; pre-inception 3x returns are explicitly reconstructed, not observed.",
        "",
        f"**Long-history 20% CAGR threshold: {status}.**",
        "",
        "## Period Results",
        "",
        _table(result.period_metrics.reset_index()),
        "",
        "## Reconstruction Reconciliation",
        "",
    ]
    lines.extend(f"- **{key.replace('_', ' ').title()}**: {value:.6g}" for key, value in result.reconciliation.items())
    lines.extend(["", "## Drag Sensitivity", "", _table(result.drag_sensitivity), ""])
    return "\n".join(lines)


def _load_config(path: str | Path) -> LeveragedTrendStressConfig:
    return LeveragedTrendStressConfig(**json.loads(Path(path).read_text(encoding="utf-8")))


def _table(frame) -> str:
    display = frame.copy()
    for column in display.columns:
        display[column] = display[column].map(lambda value: f"{value:.6g}" if isinstance(value, float) else str(value))
    lines = [
        "| " + " | ".join(map(str, display.columns)) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in display.astype(str).values.tolist())
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the long-history leveraged-trend falsification study.")
    parser.add_argument("--data", default="data/real/qqq_fred_stress_daily.csv")
    parser.add_argument("--actual", default="data/real/leveraged_etf_adjusted.csv")
    parser.add_argument("--output", default="reports/leveraged_trend_long_history.md")
    parser.add_argument("--config", default="config/leveraged_trend_stress.json")
    args = parser.parse_args()
    print(json.dumps(run_study(args.data, args.actual, args.output, args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
