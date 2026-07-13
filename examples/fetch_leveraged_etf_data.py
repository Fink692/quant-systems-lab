from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SYMBOLS = ("TQQQ", "QQQ", "BIL")
CHART_ENDPOINT = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


def fetch_adjusted_prices(start: str = "2010-01-01") -> pd.DataFrame:
    start_epoch = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp())
    end_epoch = int(datetime.now(timezone.utc).timestamp()) + 86_400
    series = [_fetch_symbol(symbol, start_epoch, end_epoch) for symbol in SYMBOLS]
    frame = pd.concat(series, axis=1).dropna().sort_index()
    # The chart endpoint can expose the current, still-open session as a daily bar.
    completed_before = pd.Timestamp(datetime.now(timezone.utc).date())
    frame = frame.loc[frame.index < completed_before]
    frame.index.name = "date"
    if frame.empty:
        raise ValueError("Yahoo Finance returned no common adjusted-price observations")
    return frame


def _fetch_symbol(symbol: str, start_epoch: int, end_epoch: int) -> pd.Series:
    query = urllib.parse.urlencode(
        {
            "period1": start_epoch,
            "period2": end_epoch,
            "interval": "1d",
            "events": "div,splits",
        }
    )
    request = urllib.request.Request(
        f"{CHART_ENDPOINT.format(symbol=symbol)}?{query}",
        headers={"User-Agent": "quant-systems-lab/0.1 research@example.invalid"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    chart = payload.get("chart", {})
    if chart.get("error") is not None or not chart.get("result"):
        raise ValueError(f"Yahoo Finance chart request failed for {symbol}: {chart.get('error')}")
    result = chart["result"][0]
    timestamps = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_localize(None)
    adjusted = result["indicators"]["adjclose"][0]["adjclose"]
    values = pd.Series(adjusted, index=timestamps, name=symbol.lower(), dtype=float)
    return values.dropna()[~values.dropna().index.duplicated(keep="last")]


def write_snapshot(output: str | Path, metadata_output: str | Path | None = None) -> dict[str, object]:
    frame = fetch_adjusted_prices()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_text = frame.to_csv(date_format="%Y-%m-%d", float_format="%.10f", lineterminator="\n")
    output_path.write_text(csv_text, encoding="utf-8", newline="")
    digest = hashlib.sha256(csv_text.encode("utf-8")).hexdigest()
    metadata = {
        "symbols": list(SYMBOLS),
        "field": "adjusted close",
        "rows": len(frame),
        "start": frame.index[0].strftime("%Y-%m-%d"),
        "end": frame.index[-1].strftime("%Y-%m-%d"),
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": digest,
        "source": "Yahoo Finance chart API",
    }
    if metadata_output is not None:
        metadata_path = Path(metadata_output)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch adjusted TQQQ, QQQ, and BIL daily prices.")
    parser.add_argument("--output", default="data/real/leveraged_etf_adjusted.csv")
    parser.add_argument("--metadata", default="data/real/leveraged_etf_adjusted.metadata.json")
    args = parser.parse_args()
    print(json.dumps(write_snapshot(args.output, args.metadata), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
