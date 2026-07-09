from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.data.synthetic import synthetic_factor_panel, synthetic_option_chain
from quantlab.market_making.avellaneda_stoikov import AvellanedaStoikovParams
from quantlab.market_making.book_simulator import simulate_order_book_market_maker
from quantlab.options.surface import build_volatility_surface_from_chain
from quantlab.portfolio.optimization import min_variance_weights
from quantlab.risk.attribution import factor_risk_attribution
from quantlab.risk.factor_model import fit_factor_model


def generate_artifacts(output_dir: str | Path = "examples/artifacts", seed: int = 7) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    market_path = out / "market_making_pnl_inventory.svg"
    surface_path = out / "volatility_surface_slices.svg"
    factor_path = out / "factor_risk_contributions.svg"

    _write_market_making_chart(market_path, seed)
    _write_vol_surface_chart(surface_path)
    _write_factor_risk_chart(factor_path, seed)
    return [market_path, surface_path, factor_path]


def _write_market_making_chart(path: Path, seed: int) -> None:
    result = simulate_order_book_market_maker(
        100.0,
        AvellanedaStoikovParams(risk_aversion=0.08, volatility=0.18, order_book_liquidity=1.2, horizon=1.0),
        steps=80,
        dt=1.0 / 80.0,
        levels=4,
        depth_per_level=2.0,
        market_order_intensity=700.0,
        seed=seed,
    )
    history = result.history
    svg = _line_chart(
        "Queue-Aware Market Making",
        [
            ("PnL", history["pnl"].to_numpy(dtype=float), "#1f77b4"),
            ("Inventory", history["inventory"].to_numpy(dtype=float), "#d62728"),
        ],
        subtitle=f"fill rate {result.fill_rate:.1%}; final PnL {result.final_pnl:.2f}; max inventory {result.max_inventory_abs:.0f}",
        x_label="simulation step",
        y_label="normalized value",
    )
    path.write_text(svg, encoding="utf-8")


def _write_vol_surface_chart(path: Path) -> None:
    surface = build_volatility_surface_from_chain(synthetic_option_chain())
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    series = [
        (f"T={maturity:g}y", surface.implied_volatilities[idx], colors[idx % len(colors)])
        for idx, maturity in enumerate(surface.maturities)
    ]
    svg = _line_chart(
        "Synthetic Volatility Surface Slices",
        series,
        x_values=surface.strikes,
        subtitle="SABR-generated smiles used for calibration and no-arbitrage diagnostics",
        x_label="strike",
        y_label="implied volatility",
        normalize=False,
    )
    path.write_text(svg, encoding="utf-8")


def _write_factor_risk_chart(path: Path, seed: int) -> None:
    panel = synthetic_factor_panel(seed=seed)
    model = fit_factor_model(panel.asset_returns, panel.factor_returns)
    covariance = panel.asset_returns.cov().to_numpy()
    weights = pd.Series(min_variance_weights(covariance), index=panel.asset_returns.columns)
    attribution = factor_risk_attribution(weights, model.exposures, model.factor_covariance, model.specific_variance)
    values = attribution.factor_contributions.abs().sort_values(ascending=False)
    values.loc["Specific"] = attribution.specific_variance
    svg = _bar_chart(
        "Factor Risk Contributions",
        values,
        subtitle=f"factor variance share {attribution.factor_variance / attribution.total_variance:.1%}",
        y_label="variance contribution",
    )
    path.write_text(svg, encoding="utf-8")


def _line_chart(
    title: str,
    series: list[tuple[str, np.ndarray, str]],
    subtitle: str,
    x_label: str,
    y_label: str,
    x_values: np.ndarray | None = None,
    normalize: bool = True,
) -> str:
    width, height = 920, 520
    left, right, top, bottom = 84, 36, 72, 76
    plot_w, plot_h = width - left - right, height - top - bottom
    processed: list[tuple[str, np.ndarray, str]] = []
    for name, values, color in series:
        clean = np.asarray(values, dtype=float)
        if normalize:
            span = np.max(clean) - np.min(clean)
            clean = np.zeros_like(clean) if span == 0 else (clean - np.mean(clean)) / span
        processed.append((name, clean, color))
    y_values = np.concatenate([values for _name, values, _color in processed])
    y_min, y_max = _expanded_bounds(float(np.min(y_values)), float(np.max(y_values)))
    if x_values is None:
        xs = np.arange(len(processed[0][1]), dtype=float)
    else:
        xs = np.asarray(x_values, dtype=float)
    x_min, x_max = _axis_bounds(float(np.min(xs)), float(np.max(xs)))

    parts = [_svg_header(width, height), _style(), f'<text class="title" x="{left}" y="34">{_esc(title)}</text>']
    parts.append(f'<text class="subtitle" x="{left}" y="56">{_esc(subtitle)}</text>')
    parts.append(_axes(left, top, plot_w, plot_h, x_label, y_label))
    for idx, tick in enumerate(np.linspace(x_min, x_max, 5)):
        x = _scale(tick, x_min, x_max, left, left + plot_w)
        parts.append(f'<text class="tick" x="{x:.1f}" y="{height - 46}" text-anchor="middle">{_fmt_tick(tick)}</text>')
    for tick in np.linspace(y_min, y_max, 5):
        y = _scale(tick, y_min, y_max, top + plot_h, top)
        parts.append(f'<line class="grid" x1="{left}" x2="{left + plot_w}" y1="{y:.1f}" y2="{y:.1f}" />')
        parts.append(f'<text class="tick" x="{left - 12}" y="{y + 4:.1f}" text-anchor="end">{_fmt_tick(tick)}</text>')
    for name, values, color in processed:
        points = []
        for idx, value in enumerate(values):
            x_raw = xs[idx] if idx < len(xs) else float(idx)
            x = _scale(float(x_raw), x_min, x_max, left, left + plot_w)
            y = _scale(float(value), y_min, y_max, top + plot_h, top)
            points.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<polyline class="series" stroke="{color}" points="{" ".join(points)}" />')
    parts.extend(_legend(processed, left + 18, top + 18))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _bar_chart(title: str, values: pd.Series, subtitle: str, y_label: str) -> str:
    width, height = 920, 520
    left, right, top, bottom = 88, 36, 72, 100
    plot_w, plot_h = width - left - right, height - top - bottom
    labels = [str(label) for label in values.index]
    data = values.to_numpy(dtype=float)
    y_min, y_max = 0.0, max(float(np.max(data)) * 1.08, 1e-12)
    bar_gap = 18
    bar_w = (plot_w - bar_gap * (len(data) + 1)) / len(data)
    parts = [_svg_header(width, height), _style(), f'<text class="title" x="{left}" y="34">{_esc(title)}</text>']
    parts.append(f'<text class="subtitle" x="{left}" y="56">{_esc(subtitle)}</text>')
    parts.append(_axes(left, top, plot_w, plot_h, "component", y_label))
    for tick in np.linspace(y_min, y_max, 5):
        y = _scale(tick, y_min, y_max, top + plot_h, top)
        parts.append(f'<line class="grid" x1="{left}" x2="{left + plot_w}" y1="{y:.1f}" y2="{y:.1f}" />')
        parts.append(f'<text class="tick" x="{left - 12}" y="{y + 4:.1f}" text-anchor="end">{_fmt_tick(tick)}</text>')
    for idx, value in enumerate(data):
        x = left + bar_gap + idx * (bar_w + bar_gap)
        y = _scale(float(value), y_min, y_max, top + plot_h, top)
        h = top + plot_h - y
        color = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"][idx % 4]
        parts.append(f'<rect class="bar" x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}" />')
        parts.append(f'<text class="tick" x="{x + bar_w / 2:.1f}" y="{height - 68}" text-anchor="middle">{_esc(labels[idx])}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _axes(left: int, top: int, plot_w: int, plot_h: int, x_label: str, y_label: str) -> str:
    return "\n".join(
        [
            f'<line class="axis" x1="{left}" x2="{left}" y1="{top}" y2="{top + plot_h}" />',
            f'<line class="axis" x1="{left}" x2="{left + plot_w}" y1="{top + plot_h}" y2="{top + plot_h}" />',
            f'<text class="label" x="{left + plot_w / 2:.1f}" y="{top + plot_h + 56}" text-anchor="middle">{_esc(x_label)}</text>',
            f'<text class="label" transform="translate(24 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle">{_esc(y_label)}</text>',
        ]
    )


def _legend(series: list[tuple[str, np.ndarray, str]], x: int, y: int) -> list[str]:
    parts = []
    for idx, (name, _values, color) in enumerate(series):
        yy = y + idx * 22
        parts.append(f'<line class="series" stroke="{color}" x1="{x}" x2="{x + 28}" y1="{yy}" y2="{yy}" />')
        parts.append(f'<text class="legend" x="{x + 38}" y="{yy + 5}">{_esc(name)}</text>')
    return parts


def _style() -> str:
    return """<style>
    .title { font: 700 24px Arial, sans-serif; fill: #111827; }
    .subtitle, .label, .legend { font: 14px Arial, sans-serif; fill: #374151; }
    .tick { font: 12px Arial, sans-serif; fill: #4b5563; }
    .axis { stroke: #111827; stroke-width: 1.4; }
    .grid { stroke: #e5e7eb; stroke-width: 1; }
    .series { fill: none; stroke-width: 2.6; stroke-linejoin: round; stroke-linecap: round; }
    .bar { opacity: 0.9; }
  </style>"""


def _svg_header(width: int, height: int) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">'


def _expanded_bounds(low: float, high: float) -> tuple[float, float]:
    if low == high:
        return low - 1.0, high + 1.0
    pad = 0.08 * (high - low)
    return low - pad, high + pad


def _axis_bounds(low: float, high: float) -> tuple[float, float]:
    if low == high:
        return low - 1.0, high + 1.0
    return low, high


def _fmt_tick(value: float) -> str:
    if abs(value) >= 1 and abs(value) < 1000:
        rounded = round(value)
        if abs(value - rounded) < 1e-9:
            return str(int(rounded))
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.2g}"


def _scale(value: float, src_low: float, src_high: float, dst_low: float, dst_high: float) -> float:
    return dst_low + (value - src_low) * (dst_high - dst_low) / (src_high - src_low)


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate resume-facing SVG artifacts for Quant Systems Lab.")
    parser.add_argument("--output-dir", default="examples/artifacts")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    for path in generate_artifacts(args.output_dir, args.seed):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
