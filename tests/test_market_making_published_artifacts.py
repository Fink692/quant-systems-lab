import json
from pathlib import Path

import pandas as pd

from quantlab.reporting.market_making_dashboard import load_dashboard_data
from quantlab.research import load_market_making_experiment_config

ROOT = Path(__file__).resolve().parents[1]


def test_published_sample_artifacts_match_frozen_config_and_accounting() -> None:
    report_dir = ROOT / "reports" / "market_making_sample"
    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    quality = json.loads((report_dir / "data_quality.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(report_dir / "strategy_comparison.csv")
    config = load_market_making_experiment_config(ROOT / "config" / "lobster_sample_experiment.json")

    assert summary["config_fingerprint"] == config.fingerprint
    assert summary["dataset_fingerprint"] == config.dataset.sha256
    assert quality["status"] == "pass"
    assert quality["messages"] == 301_587
    assert quality["reconstruction_match_rate"] > 0.85
    assert set(comparison["strategy"]) == set(config.strategies)
    assert comparison["accounting_error"].abs().max() < 1e-8


def test_notebook_is_executed_without_error_outputs() -> None:
    notebook = json.loads((ROOT / "notebooks" / "start_here_market_making.ipynb").read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert notebook["nbformat"] == 4
    assert code_cells
    assert all(cell["execution_count"] is not None for cell in code_cells)
    assert not [
        output for cell in code_cells for output in cell.get("outputs", []) if output.get("output_type") == "error"
    ]


def test_pdf_artifacts_have_pdf_signatures_and_nontrivial_size() -> None:
    for name in ("queue_aware_market_making_sample_paper.pdf", "queue_aware_market_making_tear_sheet.pdf"):
        path = ROOT / "output" / "pdf" / name
        assert path.stat().st_size > 20_000
        assert path.read_bytes().startswith(b"%PDF-")


def test_demo_video_has_mp4_signature_and_nontrivial_size() -> None:
    path = ROOT / "output" / "video" / "queue_aware_market_making_demo.mp4"
    header = path.read_bytes()[:32]
    assert path.stat().st_size > 1_000_000
    assert b"ftyp" in header


def test_dashboard_loader_uses_only_published_aggregate_artifacts() -> None:
    dashboard = load_dashboard_data(ROOT / "reports" / "market_making_sample")
    assert set(dashboard) == {"summary", "quality", "comparison", "sensitivity", "validation"}
    assert not dashboard["comparison"].empty
