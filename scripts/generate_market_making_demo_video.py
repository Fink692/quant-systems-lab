from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pandas as pd
from imageio_ffmpeg import get_ffmpeg_exe
from PIL import Image, ImageDraw, ImageFont

WIDTH = 1920
HEIGHT = 1080
NAVY = "#081a33"
BLUE = "#3b82f6"
WHITE = "#f8fafc"
SLATE = "#cbd5e1"
GREEN = "#34d399"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a narrated market-making research demo video.")
    parser.add_argument("--report-dir", type=Path, default=Path("reports/market_making_sample"))
    parser.add_argument("--output", type=Path, default=Path("output/video/queue_aware_market_making_demo.mp4"))
    parser.add_argument("--voice", default="Samantha")
    parser.add_argument("--rate", type=int, default=225, help="macOS say words per minute")
    args = parser.parse_args()
    if shutil.which("say") is None:
        raise RuntimeError("This generator currently requires the macOS 'say' command for narration")

    summary = json.loads((args.report_dir / "summary.json").read_text(encoding="utf-8"))
    quality = json.loads((args.report_dir / "data_quality.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(args.report_dir / "strategy_comparison.csv")
    best = comparison.sort_values("net_pnl", ascending=False).iloc[0]
    slides = _slides(summary, quality, comparison, best)
    work_dir = Path("tmp/video")
    work_dir.mkdir(parents=True, exist_ok=True)
    segments = []
    ffmpeg = get_ffmpeg_exe()

    for index, slide in enumerate(slides, start=1):
        image_path = work_dir / f"slide-{index:02d}.png"
        audio_path = work_dir / f"slide-{index:02d}.aiff"
        segment_path = work_dir / f"segment-{index:02d}.mp4"
        _render_slide(image_path, slide["title"], slide["bullets"], index, len(slides))
        subprocess.run(
            ["say", "-v", args.voice, "-r", str(args.rate), "-o", str(audio_path), slide["narration"]],
            check=True,
        )
        duration = _audio_duration_seconds(ffmpeg, audio_path) + 0.5
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loop",
                "1",
                "-framerate",
                "2",
                "-i",
                str(image_path),
                "-i",
                str(audio_path),
                "-af",
                "apad",
                "-c:v",
                "libx264",
                "-tune",
                "stillimage",
                "-crf",
                "18",
                "-g",
                "1",
                "-keyint_min",
                "1",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                "-t",
                f"{duration:.3f}",
                str(segment_path),
            ],
            check=True,
            capture_output=True,
        )
        segments.append(segment_path)

    concat_path = work_dir / "segments.txt"
    concat_path.write_text(
        "".join(f"file '{path.resolve()}'\n" for path in segments),
        encoding="utf-8",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-c",
            "copy",
            str(args.output),
        ],
        check=True,
        capture_output=True,
    )
    print(args.output)


def _audio_duration_seconds(ffmpeg: str, path: Path) -> float:
    probe = subprocess.run([ffmpeg, "-i", str(path)], check=False, capture_output=True, text=True)
    match = re.search(r"Duration: (\d+):(\d+):([\d.]+)", probe.stderr)
    if match is None:
        raise RuntimeError(f"Could not determine narration duration for {path}")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _slides(summary, quality, comparison, best):
    results = [
        f"{row.strategy.replace('_', ' ').title()}: {row.net_pnl:.2f} net PnL"
        for row in comparison.itertuples(index=False)
    ]
    return [
        {
            "title": "Queue-Aware Market Making",
            "bullets": [
                "Real NASDAQ-derived order-book messages",
                "Five policies, one execution engine",
                "Chronological validation and honest negative results",
            ],
            "narration": (
                "Welcome to Quant Systems Lab. This demonstration asks a narrow quantitative question: does explicit "
                "queue state improve market-making outcomes after latency, fees, inventory risk, and adverse selection? "
                "The repository compares five policies using one shared historical replay engine. The purpose of this "
                "public-sample study is to validate the full empirical pipeline. It does not claim a profitable trading "
                "strategy. In fact, every base policy loses money on the held-out interval, and those negative results are "
                "kept visible. Over the next few minutes I will show the data contract, reconstruction controls, research "
                "chronology, execution assumptions, results, sensitivity analysis, and immutable audit trail."
            ),
        },
        {
            "title": "Real Data and Provenance",
            "bullets": [
                f"{quality['messages']:,} synchronized AAPL messages and Level-5 states",
                f"Dataset fingerprint: {summary['dataset_fingerprint'][:20]}...",
                "Raw data is hashed, read-only, ignored by Git, and never redistributed",
            ],
            "narration": (
                "The input is the public LOBSTER AAPL Level-five demonstration sample, based on official Nasdaq "
                "Historical TotalView ITCH data. It contains three hundred one thousand five hundred eighty-seven "
                "synchronized message and book rows across the regular session on June twenty-first, twenty twelve. "
                "The downloader verifies two published SHA two-fifty-six hashes. A stable manifest records provider, "
                "dataset identifier, canonical source, actual download route, file sizes, and file hashes. Raw data is "
                "never committed. LOBSTER reports seconds after exchange-local midnight, so ingestion converts New York "
                "time into UTC nanoseconds and preserves source-row order. The public sample does not provide a distinct "
                "network receive time, which is why latency is treated as a controlled sensitivity rather than estimated."
            ),
        },
        {
            "title": "Reconstruction Before Strategy",
            "bullets": [
                f"Exact synchronized match rate: {quality['reconstruction_match_rate']:.2%}",
                f"Boundary mismatches recorded: {quality['reconstruction_mismatches']:,}",
                "Zero crossed snapshots, duplicate sequences, or invalid cancels",
            ],
            "narration": (
                "The project reconstructs market-by-price state event by event and compares it with the synchronized "
                "provider snapshot after every message. Exact matching is eighty-nine point one seven percent. The "
                "remaining thirty-two thousand six hundred seventy-three mismatches are expected boundary effects of "
                "Level-five truncation: a price can enter the visible range even though its original resting history was "
                "outside the sample. Instead of hiding that limitation, the reconstructor counts the mismatch, records the "
                "row, and explicitly reseeds from provider state. There are no crossed snapshots, duplicate sequences, "
                "out-of-order timestamps, or invalid cancel attempts. The quality gate must pass before any strategy PnL "
                "is interpreted. A complete licensed feed would replace boundary reseeding with deeper reconstruction."
            ),
        },
        {
            "title": "Locked Chronological Research",
            "bullets": [
                "Train: descriptive calibration only",
                "Validation: queue assumption selected before test",
                "Embargoes separate every interval; test remains untouched",
                f"Config fingerprint: {summary['config_fingerprint'][:20]}...",
            ],
            "narration": (
                "The experiment configuration is immutable and schema validated. It freezes the dataset fingerprint, "
                "train, validation, embargo, and test boundaries, all five strategy names, latency and queue grids, fee "
                "multipliers, order size, inventory cap, quote cadence, feature windows, model parameters, seed, and "
                "bootstrap settings. Training estimates descriptive arrival, cancel, visible and hidden trade intensity, "
                "spread transitions, midpoint volatility, and one-second signed price response. Validation selects one "
                "queue-ahead fraction using queue-aware risk-adjusted PnL. It selected one hundred percent queue ahead. "
                "Only then are the five frozen policies evaluated on the final interval. Any test-driven model change "
                "must create a new experiment and use a later untouched period."
            ),
        },
        {
            "title": "One Shared Replay Engine",
            "bullets": [
                "Fixed spread, Avellaneda-Stoikov, queue, toxicity, and latency aware",
                "Queue depletion, partial fills, stale state, fees, limits, and liquidation",
                "Independent cash ledger invalidates unexplained PnL",
            ],
            "narration": (
                "All policies use the same counterfactual small-agent replay engine. Historical cancellations reduce "
                "queue-ahead depth. Visible executions must consume the remaining queue before the agent fills. Orders can "
                "fill partially, and quotes that would breach the inventory cap are suppressed. Observation latency means "
                "the policy sees the most recent snapshot available before the delayed decision time. Maker rebates, taker "
                "fees, and fixed order charges are applied consistently. Remaining inventory is forcibly liquidated at "
                "the displayed touch at the end of the test. Every fill is written to a separate ledger, and cash is then "
                "recomputed independently from signed quantity, price, and fee. The largest accounting difference across "
                "the five results is less than four times ten to the minus eleven."
            ),
        },
        {
            "title": "Held-Out Result: No Profitable Policy",
            "bullets": results,
            "narration": (
                f"The held-out result is deliberately unflattering. Fixed spread loses one hundred thirty point six three. "
                f"Avellaneda-Stoikov loses ninety-eight point eight eight. Queue-aware loses sixty-seven point eight zero. "
                f"Toxicity-aware and latency-aware each lose forty-seven point three four in the zero-latency base case. "
                f"The least-negative policy is {best['strategy'].replace('_', ' ')}, but that is not a profitability claim. "
                "Queue awareness improves the base result relative to fixed spread while reducing maximum inventory and "
                "drawdown, yet its block-bootstrap difference has a wide interval crossing zero. The correct conclusion is "
                "that explicit state changes execution outcomes on this sample, while one session supplies no evidence of "
                "persistent alpha. The repository promotes neither a best model nor a cherry-picked scenario."
            ),
        },
        {
            "title": "Sensitivity and Failure Cases",
            "bullets": [
                "Latency: 0, 1, and 5 milliseconds",
                "Queue ahead: 0%, 50%, and 100%",
                "Fees: base and 1.5-times scenarios",
                "Some isolated positive cells do not override the locked base result",
            ],
            "narration": (
                "The test is rerun across zero, one, and five milliseconds of observation latency; zero, fifty, and one "
                "hundred percent queue ahead; and base or one-point-five times fees. Results vary materially. A few "
                "latency-aware cells are positive, but those cells are sensitivity diagnostics, not the preselected base "
                "result. The report retains inventory build-up, late cancellation, adverse post-fill moves, fee dependence, "
                "and terminal liquidation as failure cases. Intraday block bootstrap intervals resample contiguous PnL "
                "changes and consistently remain too wide for a strong comparison. They describe path uncertainty within "
                "this session only and cannot replace session-level inference across many days and regimes."
            ),
        },
        {
            "title": "Audit Trail and Next Empirical Gate",
            "bullets": [
                "Append-only run: config, dataset manifest, commit, seed, command, artifact hashes",
                "Executed notebook, dashboard, paper, tear sheet, and CI reproduction button",
                "Next: licensed multi-session L2/L3 with true receive timestamps",
            ],
            "narration": (
                "Every publishable run is registered in an append-only directory named by experiment identifier and "
                "configuration fingerprint. The record includes the dataset fingerprint, Git commit, dirty-tree status, "
                "random seed, exact command, Python and platform information, and a hash for every report, notebook, PDF, "
                "and table. A duplicate registration fails rather than overwriting history. The repository also includes an "
                "executed start-here notebook, interactive dashboard, architecture diagram, research paper, tear sheet, "
                "strict documentation build, Python three-eleven through three-thirteen CI, eighty-five percent coverage "
                "floor, static typing, dependency audit, and a manual reproduction workflow. The next empirical gate is "
                "licensed multi-session Level-two or Level-three data with true receive timestamps and a later untouched "
                "test period. The software and governance are ready; the public sample is not being mistaken for proof."
            ),
        },
    ]


def _render_slide(path: Path, title: str, bullets: list[str], index: int, total: int) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), NAVY)
    draw = ImageDraw.Draw(image)
    title_font = _font(72, bold=True)
    body_font = _font(38)
    small_font = _font(24)
    draw.rounded_rectangle((90, 80, 1830, 960), radius=32, fill="#0f2747", outline="#28476f", width=3)
    draw.text((150, 140), title, font=title_font, fill=WHITE)
    draw.rectangle((150, 245, 410, 255), fill=BLUE)
    y = 330
    for bullet in bullets:
        wrapped = textwrap.wrap(bullet, width=72) or [bullet]
        draw.ellipse((160, y + 10, 178, y + 28), fill=GREEN)
        draw.multiline_text((210, y), "\n".join(wrapped), font=body_font, fill=SLATE, spacing=12)
        y += 85 + max(len(wrapped) - 1, 0) * 52
    draw.text((150, 900), "Quant Systems Lab | real-data pipeline validation", font=small_font, fill="#94a3b8")
    draw.text((1680, 900), f"{index}/{total}", font=small_font, fill="#94a3b8")
    progress_width = int(1680 * index / total)
    draw.rectangle((120, 1010, 120 + progress_width, 1024), fill=BLUE)
    image.save(path)


def _font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


if __name__ == "__main__":
    main()
