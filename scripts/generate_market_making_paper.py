from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

NAVY = colors.HexColor("#0f2747")
BLUE = colors.HexColor("#2563eb")
LIGHT_BLUE = colors.HexColor("#e8f0fe")
SLATE = colors.HexColor("#475569")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the market-making sample paper and tear sheet PDFs.")
    parser.add_argument("--report-dir", type=Path, default=Path("reports/market_making_sample"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/pdf"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path("tmp/pdfs")
    temp_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((args.report_dir / "summary.json").read_text(encoding="utf-8"))
    quality = json.loads((args.report_dir / "data_quality.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(args.report_dir / "strategy_comparison.csv")
    sensitivity = pd.read_csv(args.report_dir / "sensitivity.csv")
    validation = pd.read_csv(args.report_dir / "validation_selection.csv")
    chart_path = _strategy_chart(comparison, temp_dir / "strategy_comparison.png")
    sensitivity_path = _sensitivity_chart(sensitivity, temp_dir / "sensitivity.png")
    _build_paper(
        args.output_dir / "queue_aware_market_making_sample_paper.pdf",
        summary,
        quality,
        comparison,
        validation,
        chart_path,
        sensitivity_path,
    )
    _build_tear_sheet(
        args.output_dir / "queue_aware_market_making_tear_sheet.pdf",
        summary,
        quality,
        comparison,
        chart_path,
    )


def _styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "PaperTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=27,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=16,
        )
    )
    styles.add(
        ParagraphStyle(
            "Section",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=NAVY,
            spaceBefore=12,
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            "BodySmall",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13.5,
            textColor=colors.HexColor("#1f2937"),
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            "Callout",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=14,
            textColor=NAVY,
            backColor=LIGHT_BLUE,
            borderColor=BLUE,
            borderWidth=0.6,
            borderPadding=8,
            spaceAfter=12,
        )
    )
    return styles


def _build_paper(path, summary, quality, comparison, validation, chart_path, sensitivity_path):
    styles = _styles()
    document = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.65 * inch,
        title="Queue-Aware Market Making on a Real NASDAQ Order-Book Sample",
        author="Charles Backman",
    )
    story = [
        Paragraph("Queue-Aware Market Making on a Real NASDAQ Order-Book Sample", styles["PaperTitle"]),
        Paragraph("A reproducible pipeline-validation study", styles["Heading3"]),
        Spacer(1, 8),
        Paragraph("Charles Backman - Quant Systems Lab", styles["BodySmall"]),
        Paragraph(
            "This paper validates a real-data research pipeline; it does not claim persistent profitability. "
            "The public sample contains one AAPL session and truncated Level-5 state.",
            styles["Callout"],
        ),
        Paragraph("Abstract", styles["Section"]),
        Paragraph(
            "We evaluate fixed-spread, Avellaneda-Stoikov, queue-aware, toxicity-aware, and latency-aware "
            "market-making policies using a single shared replay engine and a strictly chronological train, "
            "validation, embargo, and test design. The pipeline ingests 301,587 synchronized LOBSTER messages "
            "and book states, converts exchange-local time to UTC nanoseconds, records immutable data and "
            "configuration fingerprints, reconciles reconstructed state, models queue depletion and latency, "
            "forces terminal liquidation, and independently recomputes cash from fills. All policies lose money "
            "on the held-out sample. This negative result is retained and interpreted as evidence about execution "
            "sensitivity, not as a failed software demonstration.",
            styles["BodySmall"],
        ),
        Paragraph("1. Research question and hypotheses", styles["Section"]),
        Paragraph(
            "The primary research question is whether explicit queue state improves executable outcomes after "
            "fees, latency, inventory risk, and adverse selection. The central comparison is queue-aware quoting "
            "against fixed-spread and Avellaneda-Stoikov baselines. Toxicity- and latency-aware policies test "
            "whether contemporaneous signed flow and stale-state risk reduce adverse fills.",
            styles["BodySmall"],
        ),
        Paragraph("2. Data and provenance", styles["Section"]),
        _key_value_table(
            [
                ("Provider", "LOBSTER public demonstration sample"),
                ("Instrument", "AAPL, NASDAQ, 2012-06-21"),
                ("Rows", f"{quality['messages']:,} canonical messages / {quality['snapshots']:,} snapshots"),
                ("Dataset fingerprint", summary["dataset_fingerprint"]),
                ("Config fingerprint", summary["config_fingerprint"]),
                ("Reconstruction match", f"{quality['reconstruction_match_rate']:.2%}"),
            ],
            widths=(1.45 * inch, 5.55 * inch),
        ),
        Paragraph(
            "The repository never redistributes raw provider files. File hashes and provenance are recorded in a "
            "local manifest. Level-5 truncation means a price can enter the visible range without its original "
            "resting history. The reconstructor counts such mismatches and reseeds from synchronized provider state.",
            styles["BodySmall"],
        ),
        PageBreak(),
        Paragraph("3. Chronological design", styles["Section"]),
        Paragraph(
            "The session is divided into training, validation, and untouched test intervals with one-second embargoes. "
            "Training estimates descriptive intensities, spread transitions, volatility, and short-horizon signed price "
            "response. Validation selects queue-ahead fraction. The test interval is evaluated with the frozen choice. "
            f"Validation selected a queue-ahead fraction of {summary['selected_queue_ahead_fraction']:.2f}.",
            styles["BodySmall"],
        ),
        _dataframe_table(validation, ["queue_ahead_fraction", "net_pnl", "risk_score"]),
        Paragraph("4. Shared execution model", styles["Section"]),
        Paragraph(
            "Every policy receives identical historical messages, inventory limits, order size, quote cadence, fees, "
            "queue assumptions, and liquidation rules. Historical cancels reduce queue-ahead depth; visible executions "
            "consume remaining queue before filling the counterfactual agent. Observations are delayed by the configured "
            "latency. The model assumes a small agent whose orders do not alter subsequent market events.",
            styles["BodySmall"],
        ),
        Paragraph("5. Held-out results", styles["Section"]),
        Image(str(chart_path), width=6.8 * inch, height=3.0 * inch),
        _dataframe_table(
            comparison,
            ["strategy", "net_pnl", "fill_rate", "adverse_selection_cost", "max_abs_inventory", "max_drawdown"],
            font_size=7.2,
        ),
        Paragraph(
            "All five policies have negative net PnL. The toxicity-aware and latency-aware specifications lose the "
            "least in the base scenario, while queue-aware quoting improves on fixed spread but remains unprofitable. "
            "These rankings are descriptive for one held-out intraday interval and are not evidence of persistent alpha.",
            styles["Callout"],
        ),
        Paragraph("6. Sensitivity and failure analysis", styles["Section"]),
        Image(str(sensitivity_path), width=6.8 * inch, height=3.0 * inch),
        Paragraph(
            "Queue position, latency, and fee multipliers are varied over the frozen grid. The policy comparison is "
            "therefore not conditional on a single favorable queue assumption. Failure cases include inventory build-up, "
            "adverse post-fill price movement, and profitability that is dominated by fee rebates or terminal liquidation.",
            styles["BodySmall"],
        ),
        Paragraph("7. Accounting and governance", styles["Section"]),
        Paragraph(
            f"The largest absolute cash-ledger reconciliation error is {comparison['accounting_error'].abs().max():.3e}. "
            "Each publishable run records the data fingerprint, frozen configuration fingerprint, Git commit, dirty-tree "
            "status, seed, command, environment, and artifact hashes in an append-only experiment directory.",
            styles["BodySmall"],
        ),
        Paragraph("8. Limitations", styles["Section"]),
        Paragraph(
            "The public sample covers one 2012 AAPL session, only five visible levels, no distinct receive timestamp, "
            "and an illustrative fee tier. Original queue priority outside the visible range and hidden liquidity are not "
            "fully observed. Intraday block bootstrap intervals cannot replace session-level inference. A final paper "
            "requires licensed multi-session L2 or L3 data, true receive timestamps, provider-specific reconciliation, "
            "and a later untouched test interval.",
            styles["BodySmall"],
        ),
        Paragraph("9. Conclusion", styles["Section"]),
        Paragraph(
            "The contribution of this study is an auditable empirical research system rather than a profitable result. "
            "It demonstrates that real event data can be ingested, reconstructed, split chronologically, replayed through "
            "five comparable policies, stress-tested, and reconciled without hiding negative outcomes. The system is ready "
            "to receive a larger licensed dataset.",
            styles["BodySmall"],
        ),
        Paragraph("References", styles["Section"]),
        Paragraph(
            "Avellaneda, M. and Stoikov, S. (2008). High-frequency trading in a limit order book. Quantitative Finance."
            "<br/>Huang, W. and Polak, T. LOBSTER output and public NASDAQ sample documentation."
            "<br/>Cartea, A., Jaimungal, S., and Penalva, J. (2015). Algorithmic and High-Frequency Trading.",
            styles["BodySmall"],
        ),
    ]
    document.build(story, onFirstPage=_page_decoration, onLaterPages=_page_decoration)


def _build_tear_sheet(path, summary, quality, comparison, chart_path):
    styles = _styles()
    document = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        leftMargin=0.55 * inch,
        rightMargin=0.55 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )
    best = comparison.sort_values("net_pnl", ascending=False).iloc[0]
    story = [
        Paragraph("Queue-Aware Market Making - Sample Tear Sheet", styles["PaperTitle"]),
        Paragraph(
            "PIPELINE VALIDATION - one real NASDAQ-derived AAPL session; no persistent-alpha claim",
            styles["Callout"],
        ),
        _key_value_table(
            [
                ("Messages", f"{quality['messages']:,}"),
                ("Reconstruction", f"{quality['reconstruction_match_rate']:.2%}"),
                ("Selected queue ahead", f"{summary['selected_queue_ahead_fraction']:.2f}"),
                ("Least-negative policy", f"{best['strategy']} ({best['net_pnl']:.2f})"),
                ("Max accounting error", f"{comparison['accounting_error'].abs().max():.2e}"),
            ],
            widths=(2.1 * inch, 4.7 * inch),
        ),
        Spacer(1, 6),
        Image(str(chart_path), width=6.8 * inch, height=3.0 * inch),
        Paragraph("Held-out comparison", styles["Section"]),
        _dataframe_table(
            comparison, ["strategy", "net_pnl", "fill_rate", "max_abs_inventory", "max_drawdown"], font_size=7.5
        ),
        Paragraph(
            "Verdict: the real-data machinery is credible; the public sample is not sufficient for a strategy claim. "
            "Next gate: licensed multi-session L2/L3 data with true receive timestamps.",
            styles["Callout"],
        ),
    ]
    document.build(story, onFirstPage=_page_decoration)


def _strategy_chart(frame: pd.DataFrame, path: Path) -> Path:
    ordered = frame.sort_values("net_pnl")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(ordered["strategy"], ordered["net_pnl"], color="#2563eb")
    ax.axvline(0, color="#111827", linewidth=0.8)
    ax.set_title("Net PnL - chronological held-out interval")
    ax.set_xlabel("Currency units after fees and liquidation")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _sensitivity_chart(frame: pd.DataFrame, path: Path) -> Path:
    view = frame.loc[frame["strategy"].eq("queue_aware") & frame["fee_multiplier"].eq(1.0)]
    fig, ax = plt.subplots(figsize=(9, 4))
    for latency, group in view.groupby("latency_ns"):
        group = group.sort_values("queue_ahead_fraction")
        ax.plot(group["queue_ahead_fraction"], group["net_pnl"], marker="o", label=f"{latency / 1e6:g} ms")
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_title("Queue-aware PnL sensitivity")
    ax.set_xlabel("Queue-ahead fraction")
    ax.set_ylabel("Net PnL")
    ax.legend(title="Observation latency")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _key_value_table(rows, widths):
    table = Table(
        [
            [Paragraph(str(left), _styles()["BodySmall"]), Paragraph(str(right), _styles()["BodySmall"])]
            for left, right in rows
        ],
        colWidths=widths,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE),
                ("TEXTCOLOR", (0, 0), (0, -1), NAVY),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def _dataframe_table(frame, columns, font_size=8):
    selected = frame[columns].copy()
    for column in selected.select_dtypes(include="number"):
        selected[column] = selected[column].map(lambda value: f"{value:.6f}")
    data = [columns] + selected.astype(str).values.tolist()
    table = Table(data, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _page_decoration(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
    canvas.line(0.7 * inch, 0.48 * inch, 7.8 * inch, 0.48 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(SLATE)
    canvas.drawString(0.7 * inch, 0.3 * inch, "Quant Systems Lab - reproducible research artifact")
    canvas.drawRightString(7.8 * inch, 0.3 * inch, f"Page {document.page}")
    canvas.restoreState()


if __name__ == "__main__":
    main()
