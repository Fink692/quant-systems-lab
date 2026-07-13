from __future__ import annotations

import argparse
import io
import json
import re
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from quantlab.research.defensive_momentum import (
    ASSETS,
    FrozenDefensiveMomentumConfig,
    append_defensive_outcome,
    compute_defensive_momentum_outcome,
    verify_defensive_outcome_ledger,
)
from quantlab.research.paper_trading import verify_paper_ledger

NASDAQ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nasdaq.com",
}


def fetch_completed_bar(session: str) -> dict[str, object]:
    requested = pd.Timestamp(session).normalize()
    values: dict[str, object] = {"session": requested.strftime("%Y-%m-%d")}
    independent_closes: dict[str, float] = {}
    differences: dict[str, float] = {}
    for asset in ASSETS:
        symbol = asset.upper()
        if _latest_completed_session(symbol) < requested:
            raise ValueError(f"Nasdaq does not report {session} as completed for {symbol}")
        adjusted_open, adjusted_close, raw_close = _yahoo_session(symbol, requested)
        nasdaq_close = _nasdaq_historical_close(symbol, requested)
        difference = abs(raw_close / nasdaq_close - 1.0) * 10_000.0
        if difference > 5.0:
            raise ValueError(f"Yahoo/Nasdaq close difference exceeds 5 bps for {symbol}")
        values[f"{asset}_open"] = adjusted_open
        values[f"{asset}_close"] = adjusted_close
        independent_closes[asset] = nasdaq_close
        differences[asset] = difference
    dff_date, dff_percent = _prior_dff(requested)
    values.update(
        {
            "dff_percent": dff_percent,
            "dff_observation_date": dff_date.strftime("%Y-%m-%d"),
            "independent_closes": independent_closes,
            "source_difference_bps": differences,
            "source": "Yahoo adjusted OHLC; Nasdaq completed closes; official FRED DFF",
        }
    )
    return values


def _latest_completed_session(symbol: str) -> pd.Timestamp:
    url = f"https://api.nasdaq.com/api/quote/{symbol}/info?assetclass=etf"
    request = urllib.request.Request(
        url,
        headers={**NASDAQ_HEADERS, "Referer": f"https://www.nasdaq.com/market-activity/etf/{symbol.lower()}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        timestamp = str(json.load(response)["data"]["secondaryData"]["lastTradeTimestamp"])
    match = re.search(r"Closed at ([A-Z][a-z]{2} \d{1,2}, \d{4})", timestamp)
    if match is None:
        raise ValueError(f"Nasdaq does not report a completed close for {symbol}")
    return pd.Timestamp(match.group(1)).normalize()


def _yahoo_session(symbol: str, session: pd.Timestamp) -> tuple[float, float, float]:
    start = int((session - timedelta(days=7)).tz_localize("UTC").timestamp())
    end = int((session + timedelta(days=2)).tz_localize("UTC").timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={start}&period2={end}&interval=1d&events=div%2Csplits"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "quant-systems-lab/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)["chart"]["result"][0]
    dates = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_localize(None).normalize()
    quote = result["indicators"]["quote"][0]
    adjusted = result["indicators"]["adjclose"][0]["adjclose"]
    frame = pd.DataFrame(
        {"open": quote["open"], "close": quote["close"], "adjusted_close": adjusted}, index=dates
    ).dropna()
    if session not in frame.index:
        raise ValueError(f"Yahoo has no completed OHLC row for {symbol} on {session:%Y-%m-%d}")
    row = frame.loc[session]
    factor = float(row["adjusted_close"] / row["close"])
    return float(row["open"] * factor), float(row["adjusted_close"]), float(row["close"])


def _nasdaq_historical_close(symbol: str, session: pd.Timestamp) -> float:
    date = session.strftime("%Y-%m-%d")
    url = (
        f"https://api.nasdaq.com/api/quote/{symbol}/historical"
        f"?assetclass=etf&fromdate={date}&todate={date}&limit=10"
    )
    request = urllib.request.Request(
        url,
        headers={
            **NASDAQ_HEADERS,
            "Referer": f"https://www.nasdaq.com/market-activity/etf/{symbol.lower()}/historical",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    rows = ((payload.get("data") or {}).get("tradesTable") or {}).get("rows") or []
    matching = [row for row in rows if pd.Timestamp(row["date"]).normalize() == session]
    if len(matching) != 1:
        raise ValueError(f"Nasdaq has no unique historical row for {symbol} on {date}")
    return float(str(matching[0]["close"]).replace("$", "").replace(",", ""))


def _prior_dff(session: pd.Timestamp) -> tuple[pd.Timestamp, float]:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF"
    request = urllib.request.Request(url, headers={"User-Agent": "quant-systems-lab/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        text = response.read().decode("utf-8")
    frame = pd.read_csv(io.StringIO(text), parse_dates=[0])
    frame.columns = ["date", "dff"]
    frame["dff"] = pd.to_numeric(frame["dff"], errors="coerce")
    eligible = frame.loc[(frame["date"] < session) & frame["dff"].notna()]
    if eligible.empty:
        raise ValueError("FRED has no prior DFF observation")
    row = eligible.iloc[-1]
    return pd.Timestamp(row["date"]).normalize(), float(row["dff"])


def score_session(
    session: str,
    decision_ledger: str | Path,
    outcome_ledger: str | Path,
    config_path: str | Path,
) -> dict[str, object]:
    decisions = verify_paper_ledger(decision_ledger)
    config = FrozenDefensiveMomentumConfig(**json.loads(Path(config_path).read_text(encoding="utf-8")))
    outcomes = verify_defensive_outcome_ledger(outcome_ledger, decisions, config)
    requested = pd.Timestamp(session).normalize()
    if any(pd.Timestamp(outcome["session"]) == requested for outcome in outcomes):
        raise ValueError("the requested defensive outcome session is already scored")
    eligible = [decision for decision in decisions if pd.Timestamp(decision["effective_session"]) <= requested]
    if not eligible:
        raise ValueError("there is no effective defensive decision for the requested session")
    decision = max(eligible, key=lambda item: pd.Timestamp(item["effective_session"]))
    if not outcomes and requested != pd.Timestamp(decision["effective_session"]):
        raise ValueError("the genesis outcome must score the decision effective session first")
    completed_bar = fetch_completed_bar(session)
    outcome = compute_defensive_momentum_outcome(
        decision,
        completed_bar,
        datetime.now(timezone.utc).isoformat(),
        config,
        outcomes[-1] if outcomes else None,
    )
    append_defensive_outcome(outcome_ledger, outcome, decisions, config)
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser(description="Score a completed defensive-momentum paper session.")
    parser.add_argument("--session", required=True)
    parser.add_argument("--decisions", default="paper/defensive_momentum_decisions.jsonl")
    parser.add_argument("--outcomes", default="paper/defensive_momentum_outcomes.jsonl")
    parser.add_argument("--config", default="config/defensive_momentum_paper.json")
    args = parser.parse_args()
    print(json.dumps(score_session(args.session, args.decisions, args.outcomes, args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
