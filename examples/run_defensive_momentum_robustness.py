from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantlab.research.defensive_momentum import (
    load_defensive_momentum_csv,
    run_frozen_monthly_robustness,
)


def _table(frame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.6f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def run_report(data_path: str | Path, output: str | Path) -> dict[str, float]:
    result = run_frozen_monthly_robustness(load_defensive_momentum_csv(data_path))
    baseline = result.baseline_metrics
    consistency = result.consistency
    bootstrap = result.bootstrap
    grid = result.monthly_grid_summary
    calendar = result.calendar_returns.rename("return").reset_index()
    text = "\n".join(
        [
            "# Frozen Monthly Defensive-Momentum Robustness",
            "",
            "The candidate is fixed at 252-session momentum, 200-session trend, 25% volatility target, "
            "1.5x cap, monthly rebalancing, and next-open execution.",
            "",
            f"Baseline full-sample CAGR is **{baseline['cagr']:.2%}**, but the 20% threshold is **not robust**.",
            "",
            "## Consistency",
            "",
            f"- Calendar years at or above 20%: {int(consistency['calendar_years_at_or_above_target'])} "
            f"of {int(consistency['calendar_years'])}.",
            f"- Negative calendar years: {int(consistency['negative_calendar_years'])}.",
            f"- Rolling five-year windows at or above 20%: "
            f"{consistency['rolling_windows_at_or_above_target_fraction']:.2%}.",
            f"- Worst rolling five-year CAGR: {consistency['worst_rolling_five_year_cagr']:.2%}.",
            f"- Median rolling five-year CAGR: {consistency['median_rolling_five_year_cagr']:.2%}.",
            "",
            "## Calendar Returns",
            "",
            _table(calendar),
            "",
            "## Cost Sensitivity",
            "",
            _table(result.cost_sensitivity),
            "",
            "At 25 bps per unit of turnover, full-sample CAGR falls below 20%.",
            "",
            "## Excess-Leverage Drag Sensitivity",
            "",
            _table(result.drag_sensitivity),
            "",
            "## Moving-Block Bootstrap",
            "",
            f"- Samples: {int(bootstrap['samples'])}; block size: {int(bootstrap['block_size'])} sessions.",
            f"- 5th / median / 95th percentile CAGR: {bootstrap['cagr_5th_percentile']:.2%} / "
            f"{bootstrap['cagr_median']:.2%} / {bootstrap['cagr_95th_percentile']:.2%}.",
            f"- Probability CAGR is at least 20%: {bootstrap['probability_cagr_at_or_above_target']:.2%}.",
            "",
            "## Monthly Grid Breadth",
            "",
            f"Within the original monthly subfamily, {int(grid['both_at_or_above_target'])} of "
            f"{int(grid['configurations'])} configurations clear 20% in both evaluation and full periods.",
            "",
            "## Interpretation",
            "",
            "The observed 20.73% CAGR is reproducible, but it is marginal to costs, absent from most rolling "
            "five-year windows, and only slightly more likely than not under the block bootstrap. The candidate "
            "remains suitable for frozen paper observation, not for a verified 20% profitability claim.",
            "",
        ]
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return {
        "baseline_cagr": float(baseline["cagr"]),
        "rolling_five_year_target_fraction": float(consistency["rolling_windows_at_or_above_target_fraction"]),
        "bootstrap_target_probability": float(bootstrap["probability_cagr_at_or_above_target"]),
        "monthly_grid_both_count": float(grid["both_at_or_above_target"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stress the frozen monthly defensive-momentum candidate.")
    parser.add_argument("--data", default="data/real/defensive_momentum_ohlc.csv")
    parser.add_argument("--output", default="reports/defensive_momentum_robustness.md")
    args = parser.parse_args()
    print(json.dumps(run_report(args.data, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
