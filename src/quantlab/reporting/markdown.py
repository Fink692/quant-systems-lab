from __future__ import annotations

from pathlib import Path

from quantlab.workflows.demo_suite import DemoSuiteResult


def render_demo_markdown(result: DemoSuiteResult) -> str:
    """Render a demo-suite result as a compact Markdown report."""
    lines = ["# Quant Systems Lab Demo Report", ""]
    for section, metrics in result.as_dict().items():
        title = _humanize_label(section)
        lines.extend([f"## {title}", ""])
        for key, value in metrics.items():
            label = _humanize_label(key)
            if isinstance(value, float):
                rendered = f"{value:.6g}"
            else:
                rendered = str(value)
            lines.append(f"- **{label}**: {rendered}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def save_demo_report(result: DemoSuiteResult, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_demo_markdown(result), encoding="utf-8")
    return path


def _humanize_label(value: str) -> str:
    acronym_map = {
        "atm": "ATM",
        "bates": "Bates",
        "cdar": "CDaR",
        "cds": "CDS",
        "cir": "CIR",
        "cva": "CVA",
        "cvar": "CVaR",
        "dv01": "DV01",
        "heston": "Heston",
        "kmv": "KMV",
        "monte": "Monte",
        "nan": "NaN",
        "ou": "OU",
        "pca": "PCA",
        "pd": "PD",
        "pfe": "PFE",
        "pg": "PG",
        "pnl": "PnL",
        "q": "Q",
        "rl": "RL",
        "rmse": "RMSE",
        "sabr": "SABR",
        "ssvi": "SSVI",
        "svi": "SVI",
        "var": "VaR",
        "vpin": "VPIN",
    }
    words = []
    for word in value.replace("_", " ").split():
        words.append(acronym_map.get(word.lower(), word.capitalize()))
    return " ".join(words)
