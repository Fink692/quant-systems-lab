from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from quantlab.research.defensive_momentum import load_defensive_momentum_csv, run_defensive_momentum_study

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "real" / "defensive_momentum_ohlc.csv"


def test_defensive_momentum_snapshot_and_next_open_accounting():
    inputs = load_defensive_momentum_csv(DATA)
    result = run_defensive_momentum_study(inputs)
    metadata = json.loads(DATA.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    canonical = DATA.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert metadata["sha256"] == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert metadata["rows"] == len(inputs)

    history = result.selected_history
    row = history.dropna(subset=["overnight_return", "intraday_return"]).iloc[-1]
    expected = (
        (1.0 + row["overnight_return"]) * (1.0 + row["intraday_return"])
        - 1.0
        - row["trading_cost"]
        - row["leverage_drag"]
    )
    np.testing.assert_allclose(row["strategy_return"], expected, atol=1e-12)


def test_broad_grid_selection_rejects_twenty_percent_claim():
    result = run_defensive_momentum_study(load_defensive_momentum_csv(DATA))
    selected = result.selected_parameters
    assert selected["rebalance_frequency"] == "weekly"
    assert selected["momentum_days"] == 63
    assert result.evaluation_target_met is False
    assert result.period_metrics.loc["evaluation", "cagr"] < 0.10

    monthly = result.grid_metrics[result.grid_metrics["rebalance_frequency"] == "monthly"].iloc[0]
    assert monthly["evaluation_cagr"] > 0.20
    assert monthly["full_cagr"] > 0.20


def test_defensive_momentum_selection_ignores_evaluation_prices():
    inputs = load_defensive_momentum_csv(DATA)
    baseline = run_defensive_momentum_study(inputs).selected_parameters
    mutated = inputs.copy()
    price_columns = [column for column in mutated if column != "dff"]
    mutated.loc[mutated.index >= "2017-01-01", price_columns] *= 100.0
    selected = run_defensive_momentum_study(mutated).selected_parameters
    fields = ["momentum_days", "trend_days", "target_volatility", "max_leverage", "rebalance_frequency"]
    assert selected[fields].to_dict() == baseline[fields].to_dict()


def test_defensive_momentum_report_discloses_exploratory_subfamily(tmp_path):
    script = ROOT / "examples" / "run_defensive_momentum_study.py"
    spec = importlib.util.spec_from_file_location("run_defensive_momentum_study", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "report.md"
    payload = module.run_study(DATA, ROOT / "config" / "defensive_momentum.json", output)
    text = output.read_text(encoding="utf-8")
    assert payload["selected_evaluation_target_met"] is False
    assert "All-grid selected evaluation 20% CAGR threshold: NOT MET" in text
    assert "must not replace the all-grid selection" in text
