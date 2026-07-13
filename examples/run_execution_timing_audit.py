from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantlab.research.execution_timing import load_adjusted_ohlc_csv, run_execution_timing_audit


def _markdown_table(frame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.6f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def run_audit(data_path: str | Path, output: str | Path | None = None) -> dict[str, float]:
    result = run_execution_timing_audit(load_adjusted_ohlc_csv(data_path))
    payload = {
        "holdout_next_open_cagr": float(result.period_metrics.loc[("holdout", "next_open"), "cagr"]),
        "holdout_close_to_close_cagr": float(result.period_metrics.loc[("holdout", "close_to_close"), "cagr"]),
        "holdout_next_open_max_drawdown": float(result.period_metrics.loc[("holdout", "next_open"), "max_drawdown"]),
    }
    if output is not None:
        table = _markdown_table(result.period_metrics.reset_index())
        text = "\n".join(
            [
                "# Leveraged Trend Execution-Timing Audit",
                "",
                "Frozen parameters; signals use completed closes and next-open returns use adjusted OHLC.",
                "",
                table,
                "",
            ]
        )
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare next-open and close-to-close execution assumptions.")
    parser.add_argument("--data", default="data/paper/execution_timing_ohlc_2026-07-13.csv")
    parser.add_argument("--output", default="reports/leveraged_trend_execution_timing.md")
    args = parser.parse_args()
    print(json.dumps(run_audit(args.data, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
