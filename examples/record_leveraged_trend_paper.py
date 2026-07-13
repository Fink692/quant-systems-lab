from __future__ import annotations

import argparse
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quantlab.research.paper_trading import (
    FrozenPaperConfig,
    append_paper_decision,
    canonical_file_sha256,
    compute_paper_decision,
    verify_paper_ledger,
    write_immutable_text,
)

SYMBOLS = ("TQQQ", "QQQ", "BIL")
NASDAQ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nasdaq.com",
}


def fetch_completed_snapshot() -> tuple[pd.DataFrame, dict[str, object]]:
    closes: dict[str, float] = {}
    close_dates = set()
    timestamps: dict[str, str] = {}
    for symbol in SYMBOLS:
        url = f"https://api.nasdaq.com/api/quote/{symbol}/info?assetclass=etf"
        request = urllib.request.Request(
            url,
            headers={**NASDAQ_HEADERS, "Referer": f"https://www.nasdaq.com/market-activity/etf/{symbol.lower()}"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.load(response)["data"]
        secondary = data["secondaryData"]
        timestamp = str(secondary["lastTradeTimestamp"])
        match = re.search(r"Closed at ([A-Z][a-z]{2} \d{1,2}, \d{4})", timestamp)
        if match is None:
            raise ValueError(f"Nasdaq did not report a completed close for {symbol}: {timestamp}")
        close_dates.add(pd.Timestamp(match.group(1)).normalize())
        closes[symbol.lower()] = float(str(secondary["lastSalePrice"]).replace("$", "").replace(",", ""))
        timestamps[symbol.lower()] = timestamp
    if len(close_dates) != 1:
        raise ValueError("independent symbols do not share one completed session")
    as_of = close_dates.pop()

    series = []
    period2 = int(datetime.now(timezone.utc).timestamp()) + 86_400
    for symbol in SYMBOLS:
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            f"?period1=1230768000&period2={period2}&interval=1d&events=div%2Csplits"
        )
        request = urllib.request.Request(url, headers={"User-Agent": "quant-systems-lab/0.1"})
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)["chart"]["result"][0]
        dates = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_localize(None).normalize()
        adjusted = result["indicators"]["adjclose"][0]["adjclose"]
        series.append(pd.Series(adjusted, index=dates, name=symbol.lower(), dtype=float).dropna())
    frame = pd.concat(series, axis=1).dropna().sort_index().loc[:as_of].tail(260)
    if frame.index[-1] != as_of:
        raise ValueError("Yahoo signal data does not include the independent completed session")
    differences = {symbol: abs(float(frame[symbol].iloc[-1]) / closes[symbol] - 1.0) * 10_000.0 for symbol in closes}
    metadata: dict[str, object] = {
        "as_of_session": as_of.strftime("%Y-%m-%d"),
        "independent_source": "Nasdaq.com quote info secondaryData",
        "independent_closes": closes,
        "independent_timestamps": timestamps,
        "source_difference_bps": differences,
        "signal_source": "Yahoo Finance adjusted close",
        "rows": len(frame),
    }
    frame.index.name = "date"
    return frame, metadata


def record_decision(
    snapshot_path: str | Path,
    metadata_path: str | Path,
    ledger_path: str | Path,
    effective_session: str,
    config_path: str | Path,
) -> dict[str, object]:
    frame, metadata = fetch_completed_snapshot()
    snapshot = Path(snapshot_path)
    write_immutable_text(
        snapshot,
        frame.to_csv(date_format="%Y-%m-%d", float_format="%.10f", lineterminator="\n"),
    )
    metadata["snapshot_sha256"] = canonical_file_sha256(snapshot)
    metadata_file = Path(metadata_path)
    write_immutable_text(metadata_file, json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    config = FrozenPaperConfig(**json.loads(Path(config_path).read_text(encoding="utf-8")))
    ledger = verify_paper_ledger(ledger_path)
    previous = ledger[-1]["record_hash"] if ledger else "GENESIS"
    decision = compute_paper_decision(
        frame,
        str(metadata["as_of_session"]),
        effective_session,
        datetime.now(timezone.utc).isoformat(),
        config,
        str(metadata["snapshot_sha256"]),
        dict(metadata["independent_closes"]),
        previous,
    )
    append_paper_decision(ledger_path, decision)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description="Append a frozen leveraged-trend paper decision.")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--ledger", default="paper/leveraged_trend_decisions.jsonl")
    parser.add_argument("--effective-session", required=True)
    parser.add_argument("--config", default="config/leveraged_trend_paper.json")
    args = parser.parse_args()
    decision = record_decision(args.snapshot, args.metadata, args.ledger, args.effective_session, args.config)
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
