from __future__ import annotations

import importlib.util
from pathlib import Path


def test_resume_artifact_generator_writes_svg_charts(tmp_path):
    script_path = Path(__file__).resolve().parents[1] / "examples" / "generate_resume_artifacts.py"
    spec = importlib.util.spec_from_file_location("generate_resume_artifacts", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    paths = module.generate_artifacts(tmp_path, seed=7)

    assert {path.name for path in paths} == {
        "market_making_pnl_inventory.svg",
        "volatility_surface_slices.svg",
        "factor_risk_contributions.svg",
    }
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert text.startswith("<svg")
        assert "</svg>" in text
    assert "Queue-Aware Market Making" in (tmp_path / "market_making_pnl_inventory.svg").read_text(encoding="utf-8")
    assert "Synthetic Volatility Surface Slices" in (tmp_path / "volatility_surface_slices.svg").read_text(encoding="utf-8")
    assert "Factor Risk Contributions" in (tmp_path / "factor_risk_contributions.svg").read_text(encoding="utf-8")


def test_price_panel_example_analyzes_template_csv():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "examples" / "run_price_panel_example.py"
    spec = importlib.util.spec_from_file_location("run_price_panel_example", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module.analyze_price_panel(root / "examples" / "price_panel_template.csv")

    assert result["rows"] == 20
    assert result["assets"] == 5
    assert result["return_rows"] == 19
    assert result["historical_var_95"] > 0.0
    assert result["historical_cvar_95"] >= result["historical_var_95"]
    assert abs(sum(result["min_variance_weights"].values()) - 1.0) < 1e-12
    assert abs(sum(result["risk_parity_weights"].values()) - 1.0) < 1e-12
