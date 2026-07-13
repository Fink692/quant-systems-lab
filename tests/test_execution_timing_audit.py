from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from quantlab.research.execution_timing import load_adjusted_ohlc_csv, run_execution_timing_audit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "paper" / "execution_timing_ohlc_2026-07-13.csv"


def test_execution_timing_snapshot_and_return_accounting():
    data = load_adjusted_ohlc_csv(DATA)
    result = run_execution_timing_audit(data)
    metadata = json.loads(DATA.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    canonical = DATA.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert metadata["sha256"] == hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    row = result.history.dropna(subset=["overnight_return", "intraday_return"]).iloc[-1]
    expected = (1.0 + row["overnight_return"]) * (1.0 + row["intraday_return"]) - 1.0 - row["trading_cost"]
    np.testing.assert_allclose(row["next_open_return"], expected, atol=1e-12)


def test_frozen_next_open_holdout_result_clears_recent_threshold():
    result = run_execution_timing_audit(load_adjusted_ohlc_csv(DATA))
    next_open = result.period_metrics.loc[("holdout", "next_open")]
    close_to_close = result.period_metrics.loc[("holdout", "close_to_close")]
    assert next_open["cagr"] > 0.20
    assert next_open["max_drawdown"] < 0.25
    assert next_open["cagr"] > close_to_close["cagr"]
