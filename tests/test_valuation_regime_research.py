from __future__ import annotations

import importlib.util
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
