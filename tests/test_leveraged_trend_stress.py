from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from quantlab.research.leveraged_trend import load_leveraged_etf_csv
from quantlab.research.leveraged_trend_stress import load_qqq_fred_csv, run_leveraged_trend_stress

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "real" / "qqq_fred_stress_daily.csv"
ACTUAL_PATH = ROOT / "data" / "real" / "leveraged_etf_adjusted.csv"


def test_long_history_inputs_and_reconstruction_are_auditable():
    inputs = load_qqq_fred_csv(DATA_PATH)
    result = run_leveraged_trend_stress(inputs, load_leveraged_etf_csv(ACTUAL_PATH)["tqqq"])
    metadata = json.loads(DATA_PATH.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    canonical = DATA_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")

    assert len(inputs) >= 6_800
    assert metadata["rows"] == len(inputs)
    assert metadata["sha256"] == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    expected = (
        3.0 * result.history["qqq_return"]
        - 2.0 * result.history["cash_return"]
        - result.config.annual_fund_drag / 252.0
    )
    np.testing.assert_allclose(result.history["levered_return"], expected, atol=1e-12)


def test_long_history_falsifies_twenty_percent_while_reconciling_actual_etf():
    result = run_leveraged_trend_stress(
        load_qqq_fred_csv(DATA_PATH),
        load_leveraged_etf_csv(ACTUAL_PATH)["tqqq"],
    )

    assert result.long_history_target_met is False
    assert result.period_metrics.loc["full_history", "cagr"] < 0.20
    assert result.period_metrics.loc["dotcom_and_gfc", "cagr"] < 0.10
    assert result.period_metrics.loc["published_holdout", "cagr"] > 0.20
    assert result.reconciliation["daily_return_correlation"] > 0.99
    assert result.reconciliation["actual_minus_synthetic_annual_mean"] < 0.0


def test_long_history_report_records_failed_threshold(tmp_path):
    script = ROOT / "examples" / "run_leveraged_trend_stress.py"
    spec = importlib.util.spec_from_file_location("run_leveraged_trend_stress", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "stress.md"
    payload = module.run_study(DATA_PATH, ACTUAL_PATH, output, ROOT / "config" / "leveraged_trend_stress.json")
    text = output.read_text(encoding="utf-8")

    assert payload["long_history_target_met"] is False
    assert "Long-history 20% CAGR threshold: NOT MET" in text
    assert "pre-inception 3x returns are explicitly reconstructed, not observed" in text
