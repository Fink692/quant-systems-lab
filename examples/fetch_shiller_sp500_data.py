from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


DATA_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"
KEEP_COLUMNS = ["Date", "SP500", "Dividend", "Earnings", "Consumer Price Index", "Long Interest Rate", "Real Price", "PE10"]


def fetch_shiller_sp500_monthly(output: str | Path = "data/real/shiller_sp500_monthly.csv") -> Path:
    request = Request(DATA_URL, headers={"User-Agent": "quant-systems-lab/0.1"})
    with urlopen(request, timeout=30) as response:
        raw = pd.read_csv(response)
    cleaned = _clean(raw)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(path, index=False)
    return path


def _clean(raw: pd.DataFrame) -> pd.DataFrame:
    missing = set(KEEP_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"missing expected columns: {sorted(missing)}")
    frame = raw[KEEP_COLUMNS].copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    for column in KEEP_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["Date", "SP500", "PE10", "Long Interest Rate"]).sort_values("Date")
    frame = frame[frame["SP500"] > 0]
    frame = frame[frame["PE10"] > 0]
    frame = frame[(frame["Date"] >= "1980-01-01") & (frame["Date"] <= "2025-12-31")]
    frame = frame.rename(
        columns={
            "Date": "date",
            "SP500": "sp500",
            "Dividend": "dividend",
            "Earnings": "earnings",
            "Consumer Price Index": "cpi",
            "Long Interest Rate": "long_rate",
            "Real Price": "real_price",
            "PE10": "pe10",
        }
    )
    return frame.reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Shiller/DataHub monthly S&P 500 data.")
    parser.add_argument("--output", default="data/real/shiller_sp500_monthly.csv")
    args = parser.parse_args()
    print(fetch_shiller_sp500_monthly(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
