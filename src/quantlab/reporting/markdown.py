from __future__ import annotations

from pathlib import Path

from quantlab.workflows.demo_suite import DemoSuiteResult


def render_demo_markdown(result: DemoSuiteResult) -> str:
    """Render a demo-suite result as a compact Markdown report."""
    lines = ["# Quant Systems Lab Demo Report", ""]
    for section, metrics in result.as_dict().items():
        title = section.replace("_", " ").title()
        lines.extend([f"## {title}", ""])
        for key, value in metrics.items():
            label = key.replace("_", " ")
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
