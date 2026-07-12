from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_dashboard_data(report_dir: str | Path) -> dict[str, object]:
    root = Path(report_dir)
    required = {
        "summary": root / "summary.json",
        "quality": root / "data_quality.json",
        "comparison": root / "strategy_comparison.csv",
        "sensitivity": root / "sensitivity.csv",
        "validation": root / "validation_selection.csv",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing dashboard artifacts: {missing}; run make reproduce-market-making-sample")
    return {
        "summary": json.loads(required["summary"].read_text(encoding="utf-8")),
        "quality": json.loads(required["quality"].read_text(encoding="utf-8")),
        "comparison": pd.read_csv(required["comparison"]),
        "sensitivity": pd.read_csv(required["sensitivity"]),
        "validation": pd.read_csv(required["validation"]),
    }
