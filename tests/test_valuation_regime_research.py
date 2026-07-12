from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import numpy as np

from quantlab.research.valuation_regime import (
    ValuationRegimeConfig,
    load_shiller_sp500_csv,
    run_valuation_regime_walk_forward,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "real" / "shiller_sp500_monthly.csv"


def test_shiller_real_data_loader_and_walk_forward_have_no_date_leakage():
    data = load_shiller_sp500_csv(DATA_PATH)
    result = run_valuation_regime_walk_forward(
        data,
        ValuationRegimeConfig(train_months=120, validation_months=36, test_months=36, step_months=36),
    )

    assert len(data) >= 500
    assert len(result.folds) >= 8
    assert result.history.index.is_monotonic_increasing
    assert result.folds["train_end"].lt(result.folds["validation_start"]).all()
    assert result.folds["validation_end"].lt(result.folds["test_start"]).all()
    assert result.folds["test_return"].notna().all()
    assert result.history.index[-1] == data.index[-1]
    assert result.history["exposure"].between(0.0, result.config.max_leverage).all()


def test_real_data_tear_sheet_contains_costs_risk_and_robustness():
    data = load_shiller_sp500_csv(DATA_PATH)
    result = run_valuation_regime_walk_forward(data)
    metrics = result.tear_sheet.metrics

    for key in (
        "cagr",
        "benchmark_cagr",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "hit_rate",
        "profit_factor",
        "beta",
        "alpha_annual",
        "value_at_risk_95",
        "conditional_var_95",
        "total_cost",
    ):
        assert key in metrics
        assert np.isfinite(metrics[key])
    assert 0.0 <= metrics["hit_rate"] <= 1.0
    assert metrics["conditional_var_95"] >= metrics["value_at_risk_95"]
    assert metrics["total_cost"] > 0.0
    assert len(result.tear_sheet.monthly_returns) > 0
    assert {"cheap", "neutral", "expensive"}.issubset(set(result.tear_sheet.regime_breakdown.index))
    assert result.robustness.positive_cost_scenarios >= 1
    assert {
        "valuation_regime",
        "valuation_regime_bond_sleeve",
        "buy_and_hold",
        "volatility_targeted_equity",
        "sixty_forty_proxy",
        "volatility_matched_equity",
        "beta_matched_equity",
    } == set(result.diagnostics.baseline_comparison.index)
    assert set(result.diagnostics.bootstrap_confidence_intervals.columns) == {"estimate", "lower_95", "upper_95"}
    assert (
        result.diagnostics.bootstrap_confidence_intervals["lower_95"]
        .le(result.diagnostics.bootstrap_confidence_intervals["upper_95"])
        .all()
    )
    assert 0.0 <= result.diagnostics.overfitting_metrics["probability_backtest_overfitting"] <= 1.0
    assert 0.0 <= result.diagnostics.overfitting_metrics["deflated_sharpe_probability"] <= 1.0
    assert result.diagnostics.overfitting_metrics["selection_trials_per_fold"] == 9.0
    assert len(result.diagnostics.parameter_stability) == len(result.folds)


def test_valuation_regime_report_script_reproduces_markdown(tmp_path):
    script_path = ROOT / "examples" / "run_valuation_regime_study.py"
    spec = importlib.util.spec_from_file_location("run_valuation_regime_study", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    output = tmp_path / "valuation_report.md"
    payload = module.run_study(DATA_PATH, output)
    text = output.read_text(encoding="utf-8")

    assert payload["observations"] >= 500
    assert payload["out_of_sample_months"] >= 300
    assert payload["folds"] >= 8
    assert payload["total_cost"] > 0.0
    assert "# Valuation-Regime Walk-Forward Study" in text
    assert "## Cost Robustness" in text
    assert "## Risk-Matched and Simple Baselines" in text
    assert "## Bootstrap Confidence Intervals" in text
    assert "## Overfitting Diagnostics" in text
    assert "## Parameter Stability" in text


def test_shiller_fetcher_writes_dataset_provenance_manifest(tmp_path, monkeypatch):
    script_path = ROOT / "examples" / "fetch_shiller_sp500_data.py"
    spec = importlib.util.spec_from_file_location("fetch_shiller_sp500_data", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    csv = (
        "Date,SP500,Dividend,Earnings,Consumer Price Index,Long Interest Rate,Real Price,PE10\n"
        "2023-08-01,4500,70,190,305,4.1,4500,30\n"
        "2023-09-01,4550,71,191,306,4.2,4550,31\n"
    ).encode()
    monkeypatch.setattr(module, "urlopen", lambda *_args, **_kwargs: io.BytesIO(csv))
    output = tmp_path / "shiller.csv"

    module.fetch_shiller_sp500_monthly(output)

    manifest = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_url"] == module.DATA_URL
    assert manifest["rows"] == 2
    assert manifest["last_observation"] == "2023-09-01"
    assert manifest["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert "not label it live" in manifest["freshness_warning"]
