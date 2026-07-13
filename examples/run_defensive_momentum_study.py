from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantlab.research.defensive_momentum import (
    DefensiveMomentumConfig,
    load_defensive_momentum_csv,
    run_defensive_momentum_study,
)


def _table(frame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.6f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def run_study(data_path: str | Path, config_path: str | Path, output: str | Path) -> dict[str, object]:
    raw_config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    for field in (
        "momentum_days",
        "trend_days",
        "target_volatilities",
        "max_leverages",
        "rebalance_frequencies",
    ):
        raw_config[field] = tuple(raw_config[field])
    result = run_defensive_momentum_study(load_defensive_momentum_csv(data_path), DefensiveMomentumConfig(**raw_config))
    selected = result.selected_parameters
    monthly = result.grid_metrics[result.grid_metrics["rebalance_frequency"] == "monthly"].iloc[0]
    text = "\n".join(
        [
            "# Defensive Multi-Asset Momentum Exploration",
            "",
            "Real adjusted QQQ/GLD/TLT OHLC and official FRED rates; completed-close signals execute at the next open.",
            "",
            "**All-grid selected evaluation 20% CAGR threshold: "
            + ("MET" if result.evaluation_target_met else "NOT MET")
            + ".**",
            "",
            "## Development-Selected Parameters",
            "",
            f"- Momentum: {int(selected['momentum_days'])} sessions.",
            f"- Trend: {int(selected['trend_days'])} sessions.",
            f"- Target volatility: {float(selected['target_volatility']):.0%}.",
            f"- Maximum leverage: {float(selected['max_leverage']):.1f}x.",
            f"- Rebalance: {selected['rebalance_frequency']}.",
            "",
            "## Selected-Model Period Results",
            "",
            _table(result.period_metrics.reset_index()),
            "",
            "## Best Monthly Subfamily Row",
            "",
            _table(monthly.to_frame().T),
            "",
            "The monthly row is exploratory: it was observed after the broader weekly/monthly evaluation was inspected. "
            "It must not replace the all-grid selection or be called prospective evidence.",
            "",
            "## Full-History Cost Sensitivity",
            "",
            _table(result.cost_sensitivity),
            "",
        ]
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return {
        "selected_evaluation_cagr": float(result.period_metrics.loc["evaluation", "cagr"]),
        "selected_full_cagr": float(result.period_metrics.loc["full", "cagr"]),
        "selected_evaluation_target_met": result.evaluation_target_met,
        "best_monthly_evaluation_cagr": float(monthly["evaluation_cagr"]),
        "best_monthly_full_cagr": float(monthly["full_cagr"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run defensive multi-asset momentum research.")
    parser.add_argument("--data", default="data/real/defensive_momentum_ohlc.csv")
    parser.add_argument("--config", default="config/defensive_momentum.json")
    parser.add_argument("--output", default="reports/defensive_momentum_study.md")
    args = parser.parse_args()
    print(json.dumps(run_study(args.data, args.config, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
