from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def fetch_adjusted_ohlc() -> tuple[pd.DataFrame, dict[str, object]]:
    as_of, nasdaq_close = _completed_nasdaq_close("TQQQ")
    columns = []
    for symbol in ("TQQQ", "BIL"):
        period2 = int(datetime.now(timezone.utc).timestamp()) + 86_400
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            f"?period1=1230768000&period2={period2}&interval=1d&events=div%2Csplits"
        )
        request = urllib.request.Request(url, headers={"User-Agent": "quant-systems-lab/0.1"})
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)["chart"]["result"][0]
        dates = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_localize(None).normalize()
        quote = result["indicators"]["quote"][0]
        adjusted = result["indicators"]["adjclose"][0]["adjclose"]
        raw = pd.DataFrame(
            {"open": quote["open"], "close": quote["close"], "adjusted_close": adjusted}, index=dates
        ).dropna()
        adjustment = raw["adjusted_close"] / raw["close"]
        columns.append(
            pd.DataFrame(
                {
                    f"{symbol.lower()}_open": raw["open"] * adjustment,
                    f"{symbol.lower()}_close": raw["adjusted_close"],
                },
                index=raw.index,
            )
        )
    frame = pd.concat(columns, axis=1).dropna().sort_index().loc[:as_of]
    frame.index.name = "date"
    difference_bps = abs(float(frame["tqqq_close"].iloc[-1]) / nasdaq_close - 1.0) * 10_000.0
    if frame.index[-1] != as_of or difference_bps > 5.0:
        raise ValueError("adjusted OHLC does not reconcile to the completed Nasdaq session")
    metadata = {
        "as_of_session": as_of.strftime("%Y-%m-%d"),
        "rows": len(frame),
        "fields": list(frame.columns),
        "source": "Yahoo Finance OHLC adjusted by Yahoo adjusted-close factor",
        "independent_tqqq_close": nasdaq_close,
        "source_difference_bps": difference_bps,
    }
    return frame, metadata


def _completed_nasdaq_close(symbol: str) -> tuple[pd.Timestamp, float]:
    url = f"https://api.nasdaq.com/api/quote/{symbol}/info?assetclass=etf"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://www.nasdaq.com",
            "Referer": f"https://www.nasdaq.com/market-activity/etf/{symbol.lower()}",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        secondary = json.load(response)["data"]["secondaryData"]
    match = re.search(r"Closed at ([A-Z][a-z]{2} \d{1,2}, \d{4})", secondary["lastTradeTimestamp"])
    if match is None:
        raise ValueError("Nasdaq does not report a completed session")
    close = float(str(secondary["lastSalePrice"]).replace("$", "").replace(",", ""))
    return pd.Timestamp(match.group(1)).normalize(), close


def write_snapshot(output: str | Path, metadata_output: str | Path) -> dict[str, object]:
    frame, metadata = fetch_adjusted_ohlc()
    csv_text = frame.to_csv(date_format="%Y-%m-%d", float_format="%.10f", lineterminator="\n")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(csv_text, encoding="utf-8", newline="")
    metadata["sha256"] = hashlib.sha256(csv_text.encode("utf-8")).hexdigest()
    metadata["retrieved_utc"] = datetime.now(timezone.utc).isoformat()
    metadata_path = Path(metadata_output)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch adjusted TQQQ/BIL OHLC for execution timing research.")
    parser.add_argument("--output", default="data/paper/execution_timing_ohlc_2026-07-13.csv")
    parser.add_argument("--metadata", default="data/paper/execution_timing_ohlc_2026-07-13.metadata.json")
    args = parser.parse_args()
    print(json.dumps(write_snapshot(args.output, args.metadata), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
