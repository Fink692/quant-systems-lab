from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

COINBASE_CANDLES = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
FRED_DFF = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF"


def fetch_snapshot() -> tuple[pd.DataFrame, dict[str, object]]:
    as_of = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1)
    start = pd.Timestamp("2015-01-01", tz="UTC")
    rows: list[list[float]] = []
    cursor = start
    while cursor <= as_of:
        end = min(cursor + pd.Timedelta(days=250), as_of + pd.Timedelta(days=1))
        url = f"{COINBASE_CANDLES}?granularity=86400" f"&start={cursor.isoformat()}&end={end.isoformat()}"
        request = urllib.request.Request(
            url, headers={"User-Agent": "quant-systems-lab/0.1", "Accept": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            rows.extend(json.load(response))
        cursor = end
        time.sleep(0.05)
    frame = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    frame = frame.drop_duplicates("time")
    frame["date"] = pd.to_datetime(frame["time"], unit="s", utc=True).dt.tz_localize(None).dt.normalize()
    frame = frame.set_index("date").sort_index().loc[: as_of.tz_localize(None)]
    frame = frame[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce").dropna()
    if frame.empty or frame.index[-1] != as_of.tz_localize(None):
        raise ValueError("Coinbase snapshot does not end on the last completed UTC day")
    if (frame <= 0).any().any():
        raise ValueError("Coinbase OHLCV values must be positive")

    request = urllib.request.Request(FRED_DFF, headers={"User-Agent": "quant-systems-lab/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        rates = pd.read_csv(io.StringIO(response.read().decode("utf-8")), parse_dates=[0])
    rates.columns = ["date", "dff"]
    rates["dff"] = pd.to_numeric(rates["dff"], errors="coerce")
    rate_series = rates.dropna().set_index("date")["dff"]
    frame["dff"] = rate_series.reindex(frame.index).ffill().bfill()

    yahoo_close = _fetch_yahoo_close(as_of)
    joined = pd.concat(
        [frame["close"].pct_change().rename("coinbase"), yahoo_close.pct_change().rename("yahoo")], axis=1
    ).dropna()
    residual = joined["coinbase"] - joined["yahoo"]
    metadata: dict[str, object] = {
        "as_of_session": as_of.strftime("%Y-%m-%d"),
        "rows": len(frame),
        "source": "Coinbase Exchange BTC-USD daily candles",
        "source_url": COINBASE_CANDLES,
        "rate_source": "Federal Reserve Bank of St. Louis DFF",
        "rate_source_url": FRED_DFF,
        "independent_source": "Yahoo Finance BTC-USD daily closes",
        "overlap_observations": len(joined),
        "daily_return_correlation": float(joined.corr().iloc[0, 1]),
        "coinbase_minus_yahoo_annual_mean": float(residual.mean() * 365.0),
        "annualized_tracking_error": float(residual.std(ddof=1) * np.sqrt(365.0)),
    }
    frame.index.name = "date"
    return frame, metadata


def _fetch_yahoo_close(as_of: pd.Timestamp) -> pd.Series:
    period2 = int((as_of + pd.Timedelta(days=2)).timestamp())
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/BTC-USD" f"?period1=1420070400&period2={period2}&interval=1d"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "quant-systems-lab/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)["chart"]["result"][0]
    dates = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_localize(None).normalize()
    return (
        pd.Series(result["indicators"]["quote"][0]["close"], index=dates, dtype=float)
        .dropna()
        .loc[: as_of.tz_localize(None)]
    )


def write_snapshot(output: str | Path, metadata_output: str | Path) -> dict[str, object]:
    frame, metadata = fetch_snapshot()
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
    parser = argparse.ArgumentParser(description="Fetch completed Coinbase BTC candles and official DFF rates.")
    parser.add_argument("--output", default="data/real/coinbase_btc_dff.csv")
    parser.add_argument("--metadata", default="data/real/coinbase_btc_dff.metadata.json")
    args = parser.parse_args()
    print(json.dumps(write_snapshot(args.output, args.metadata), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
