from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.market_data.lobster import LobsterDataset
from quantlab.market_data.reconstruction import ReconstructionReport


@dataclass(frozen=True)
class MarketDataQualityReport:
    symbol: str
    session_date: str
    messages: int
    snapshots: int
    duration_hours: float
    sequence_gaps: int
    duplicate_sequences: int
    non_monotonic_timestamps: int
    duplicate_timestamps: int
    crossed_snapshots: int
    median_spread: float
    p99_spread: float
    median_top_depth: float
    event_counts: dict[str, int]
    reconstruction_match_rate: float
    reconstruction_mismatches: int
    invalid_cancels: int
    status: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def write_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return output

    def write_markdown(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        events = "\n".join(f"| {name} | {count:,} |" for name, count in sorted(self.event_counts.items()))
        text = f"""# Market-Data Quality Report

## Verdict

**{self.status.upper()}** — {self.symbol} on {self.session_date}. This report describes data integrity and reconstruction only; it is not strategy evidence.

## Integrity

| Metric | Value |
| --- | ---: |
| Canonical messages | {self.messages:,} |
| Synchronized snapshots | {self.snapshots:,} |
| Duration | {self.duration_hours:.2f} hours |
| Sequence gaps | {self.sequence_gaps:,} |
| Duplicate sequences | {self.duplicate_sequences:,} |
| Non-monotonic timestamps | {self.non_monotonic_timestamps:,} |
| Duplicate timestamps | {self.duplicate_timestamps:,} |
| Crossed snapshots | {self.crossed_snapshots:,} |
| Reconstruction match rate | {self.reconstruction_match_rate:.2%} |
| Reconstruction mismatches | {self.reconstruction_mismatches:,} |
| Invalid cancel/execution attempts | {self.invalid_cancels:,} |

## Market description

| Metric | Value |
| --- | ---: |
| Median spread | {self.median_spread:.4f} |
| 99th-percentile spread | {self.p99_spread:.4f} |
| Median top-of-book depth | {self.median_top_depth:.2f} |

## Event counts

| Event | Count |
| --- | ---: |
{events}

## Interpretation

LOBSTER Level-N samples contain synchronized top-N snapshots but not the full depth outside the requested range. A level can therefore enter the visible range without its original resting order having appeared in the truncated message history. The replay deliberately records and reseeds those boundary mismatches instead of silently treating the truncated feed as a complete L3 book.
"""
        output.write_text(text, encoding="utf-8")
        return output


def build_quality_report(
    dataset: LobsterDataset,
    reconstruction: ReconstructionReport,
    *,
    minimum_match_rate: float = 0.85,
) -> MarketDataQualityReport:
    events = dataset.events
    snapshots = dataset.snapshots
    sequence = events["sequence_number"].to_numpy(dtype=np.int64)
    timestamps = events["timestamp_ns"].to_numpy(dtype=np.int64)
    spreads = snapshots["ask_price_1"].to_numpy(float) - snapshots["bid_price_1"].to_numpy(float)
    crossed = int((spreads <= 0).sum())
    top_depth = snapshots["ask_quantity_1"].to_numpy(float) + snapshots["bid_quantity_1"].to_numpy(float)
    duration = float((timestamps[-1] - timestamps[0]) / 3.6e12) if len(timestamps) > 1 else 0.0
    status = "pass"
    if reconstruction.match_rate < minimum_match_rate or crossed or events.empty:
        status = "fail"
    return MarketDataQualityReport(
        symbol=dataset.symbol,
        session_date=dataset.session_date.isoformat(),
        messages=len(events),
        snapshots=len(snapshots),
        duration_hours=duration,
        sequence_gaps=int((np.diff(sequence) > 1).sum()),
        duplicate_sequences=int(pd.Series(sequence).duplicated().sum()),
        non_monotonic_timestamps=int((np.diff(timestamps) < 0).sum()),
        duplicate_timestamps=int(pd.Series(timestamps).duplicated().sum()),
        crossed_snapshots=crossed,
        median_spread=float(np.median(spreads)),
        p99_spread=float(np.quantile(spreads, 0.99)),
        median_top_depth=float(np.median(top_depth)),
        event_counts={str(key): int(value) for key, value in events["source_event_name"].value_counts().items()},
        reconstruction_match_rate=reconstruction.match_rate,
        reconstruction_mismatches=reconstruction.mismatches,
        invalid_cancels=reconstruction.invalid_cancels,
        status=status,
    )
