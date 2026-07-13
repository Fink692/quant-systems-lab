from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

QQQ_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/QQQ"
    "?period1=915148800&period2={period2}&interval=1d&events=div%2Csplits"
)
DFF_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF"


def fetch_stress_inputs() -> pd.DataFrame:
    end_epoch = int(datetime.now(timezone.utc).timestamp()) + 86_400
    request = urllib.request.Request(
        QQQ_URL.format(period2=end_epoch),
        headers={"User-Agent": "quant-systems-lab/0.1 research@example.invalid"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)["chart"]["result"][0]
    dates = pd.to_datetime(payload["timestamp"], unit="s", utc=True).tz_localize(None).normalize()
    adjusted = payload["indicators"]["adjclose"][0]["adjclose"]
    qqq = pd.Series(adjusted, index=dates, name="qqq", dtype=float).dropna()
    qqq = qqq.loc[qqq.index < pd.Timestamp(datetime.now(timezone.utc).date())]

    dff = pd.read_csv(DFF_URL, parse_dates=["observation_date"]).set_index("observation_date")["DFF"]
    dff = pd.to_numeric(dff.replace(".", np.nan), errors="coerce")
    calendar = pd.date_range(dff.index.min(), qqq.index.max(), freq="D")
    dff = dff.reindex(calendar).ffill().reindex(qqq.index).rename("dff")
    frame = pd.concat([qqq, dff], axis=1).dropna().sort_index()
    frame.index.name = "date"
    if frame.empty or (frame <= 0).any().any():
        raise ValueError("stress inputs must contain complete positive observations")
    return frame


def write_snapshot(output: str | Path, metadata_output: str | Path) -> dict[str, object]:
    frame = fetch_stress_inputs()
    csv_text = frame.to_csv(date_format="%Y-%m-%d", float_format="%.10f", lineterminator="\n")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(csv_text, encoding="utf-8", newline="")
    metadata = {
        "fields": ["QQQ Yahoo Finance adjusted close", "FRED DFF percent per annum"],
        "rows": len(frame),
        "start": frame.index[0].strftime("%Y-%m-%d"),
        "end": frame.index[-1].strftime("%Y-%m-%d"),
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(csv_text.encode("utf-8")).hexdigest(),
        "sources": ["Yahoo Finance chart API", DFF_URL],
    }
    metadata_path = Path(metadata_output)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch real QQQ and FRED rate inputs for long-history stress testing.")
    parser.add_argument("--output", default="data/real/qqq_fred_stress_daily.csv")
    parser.add_argument("--metadata", default="data/real/qqq_fred_stress_daily.metadata.json")
    args = parser.parse_args()
    print(json.dumps(write_snapshot(args.output, args.metadata), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
