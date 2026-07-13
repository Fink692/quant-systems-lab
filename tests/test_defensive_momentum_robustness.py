from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from quantlab.research.defensive_momentum import (
    load_defensive_momentum_csv,
    run_frozen_monthly_robustness,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "real" / "defensive_momentum_ohlc.csv"


def test_frozen_monthly_candidate_is_marginal_not_consistent_twenty_percent():
    result = run_frozen_monthly_robustness(load_defensive_momentum_csv(DATA))
    assert result.baseline_metrics["cagr"] == pytest.approx(0.2073289611)
    assert result.consistency["rolling_windows_at_or_above_target_fraction"] < 0.50
    assert result.consistency["worst_rolling_five_year_cagr"] < 0.0
    assert result.cost_sensitivity.set_index("cost_bps").loc[25.0, "cagr"] < 0.20
    assert 0.50 < result.bootstrap["probability_cagr_at_or_above_target"] < 0.60
    assert result.monthly_grid_summary["both_at_or_above_target"] == 19.0


def test_frozen_monthly_robustness_report_rejects_verified_claim(tmp_path):
    script = ROOT / "examples" / "run_defensive_momentum_robustness.py"
    spec = importlib.util.spec_from_file_location("run_defensive_momentum_robustness", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "robustness.md"
    payload = module.run_report(DATA, output)
    text = output.read_text(encoding="utf-8")
    assert payload["baseline_cagr"] > 0.20
    assert payload["bootstrap_target_probability"] < 0.60
    assert "20% threshold is **not robust**" in text
    assert "not for a verified 20% profitability claim" in text
