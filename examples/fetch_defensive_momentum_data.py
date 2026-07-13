from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def fetch_snapshot(rates_path: str | Path) -> tuple[pd.DataFrame, dict[str, object]]:
    frames = []
    independent_closes: dict[str, float] = {}
    completed_sessions: list[pd.Timestamp] = []
    for symbol in ("QQQ", "GLD", "TLT"):
        frames.append(_yahoo_adjusted_ohlc(symbol))
        session, close = _nasdaq_completed_close(symbol)
        completed_sessions.append(session)
        independent_closes[symbol.lower()] = close
    if len(set(completed_sessions)) != 1:
        raise ValueError("independent sources do not share a completed session")
    as_of = completed_sessions[0]
    frame = pd.concat(frames, axis=1).dropna().sort_index().loc[:as_of]
    rates = pd.read_csv(rates_path, parse_dates=["date"]).sort_values("date").set_index("date")["dff"]
    frame["dff"] = rates.reindex(frame.index).ffill().bfill()
    differences = {
        symbol: abs(float(frame[f"{symbol}_close"].iloc[-1]) / close - 1.0) * 10_000.0
        for symbol, close in independent_closes.items()
    }
    if frame.index[-1] != as_of or max(differences.values()) > 5.0:
        raise ValueError("Yahoo prices do not reconcile to the completed Nasdaq session")
    frame.index.name = "date"
    metadata: dict[str, object] = {
        "as_of_session": as_of.strftime("%Y-%m-%d"),
        "rows": len(frame),
        "source": "Yahoo Finance adjusted OHLC plus official FRED DFF snapshot",
        "independent_source": "Nasdaq.com completed-session closes",
        "independent_closes": independent_closes,
        "source_difference_bps": differences,
    }
    return frame, metadata


def _yahoo_adjusted_ohlc(symbol: str) -> pd.DataFrame:
    period2 = int(datetime.now(timezone.utc).timestamp()) + 86_400
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1=1009843200&period2={period2}&interval=1d&events=div%2Csplits"
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
    factor = raw["adjusted_close"] / raw["close"]
    prefix = symbol.lower()
    return pd.DataFrame(
        {f"{prefix}_open": raw["open"] * factor, f"{prefix}_close": raw["adjusted_close"]}, index=raw.index
    )


def _nasdaq_completed_close(symbol: str) -> tuple[pd.Timestamp, float]:
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
        raise ValueError(f"Nasdaq does not report a completed session for {symbol}")
    close = float(str(secondary["lastSalePrice"]).replace("$", "").replace(",", ""))
    return pd.Timestamp(match.group(1)).normalize(), close


def write_snapshot(rates_path: str | Path, output: str | Path, metadata_output: str | Path) -> dict[str, object]:
    frame, metadata = fetch_snapshot(rates_path)
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
    parser = argparse.ArgumentParser(description="Fetch adjusted multi-asset OHLC and align official rates.")
    parser.add_argument("--rates", default="data/real/qqq_fred_stress_daily.csv")
    parser.add_argument("--output", default="data/real/defensive_momentum_ohlc.csv")
    parser.add_argument("--metadata", default="data/real/defensive_momentum_ohlc.metadata.json")
    args = parser.parse_args()
    print(json.dumps(write_snapshot(args.rates, args.output, args.metadata), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
