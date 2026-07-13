from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.research.leveraged_trend import LeveragedTrendConfig, load_leveraged_etf_csv, run_leveraged_trend_study

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "real" / "leveraged_etf_adjusted.csv"


def test_real_price_loader_and_holdout_are_point_in_time():
    prices = load_leveraged_etf_csv(DATA_PATH)
    result = run_leveraged_trend_study(prices, LeveragedTrendConfig(bootstrap_samples=100))

    assert len(prices) >= 4_000
    assert prices.index.max() < pd.Timestamp("2027-01-01")
    assert result.history.index.min() >= pd.Timestamp(result.config.holdout_start)
    assert result.history.index.min() > pd.Timestamp(result.config.validation_end)
    assert result.history.index.is_monotonic_increasing
    assert result.history["exposure"].between(0.0, result.config.max_exposure).all()
    expected_return = (
        result.history["exposure"] * result.history["tqqq_return"]
        + (1.0 - result.history["exposure"]) * result.history["cash_return"]
        - result.history["trading_cost"]
    )
    np.testing.assert_allclose(result.history["strategy_return"], expected_return, atol=1e-12)
    desired_exposure = (
        result.history["signal"]
        * (result.config.target_volatility_candidates[0] / result.history["realized_volatility"]).clip(
            upper=result.config.max_exposure
        )
    ).fillna(0.0)
    np.testing.assert_allclose(result.history["exposure"].iloc[1:], desired_exposure.shift(1).iloc[1:], atol=1e-12)

    metadata = json.loads((DATA_PATH.with_suffix(".metadata.json")).read_text(encoding="utf-8"))
    assert metadata["rows"] == len(prices)
    assert metadata["end"] == prices.index[-1].strftime("%Y-%m-%d")
    canonical_csv = DATA_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert metadata["sha256"] == hashlib.sha256(canonical_csv.encode("utf-8")).hexdigest()


def test_validation_selection_and_snapshot_holdout_clear_target_after_costs():
    result = run_leveraged_trend_study(
        load_leveraged_etf_csv(DATA_PATH),
        LeveragedTrendConfig(bootstrap_samples=100),
    )

    assert int(result.selected_parameters["moving_average_days"]) == 200
    assert float(result.selected_parameters["target_volatility"]) == 0.30
    assert result.holdout_target_met
    assert result.metrics["cagr"] >= 0.20
    assert result.metrics["total_cost"] > 0.0
    assert result.metrics["max_drawdown"] < 0.30
    assert (result.cost_sensitivity["holdout_cagr"] > 0.0).all()
    assert len(result.parameter_sensitivity) == 48
    assert 0.0 <= result.bootstrap["probability_cagr_at_least_target"] <= 1.0


def test_report_script_reproduces_leveraged_trend_markdown(tmp_path):
    script_path = ROOT / "examples" / "run_leveraged_trend_study.py"
    spec = importlib.util.spec_from_file_location("run_leveraged_trend_study", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "leveraged_trend.md"
    payload = module.run_study(DATA_PATH, output, ROOT / "config" / "leveraged_trend.json")
    text = output.read_text(encoding="utf-8")

    assert payload["holdout_target_met"] is True
    assert payload["holdout_cagr"] >= 0.20
    assert "Historical holdout threshold (20% CAGR): MET" in text
    assert "not a forecast or a guaranteed return" in text
    assert "## Bootstrap Uncertainty" in text
