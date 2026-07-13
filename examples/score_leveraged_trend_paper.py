from __future__ import annotations

import argparse
import json
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from quantlab.research.paper_trading import (
    append_paper_outcome,
    compute_paper_outcome,
    verify_outcome_ledger,
    verify_paper_ledger,
)

NASDAQ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nasdaq.com",
}


def fetch_completed_bar(session: str) -> dict[str, float | str]:
    requested = pd.Timestamp(session).normalize()
    values: dict[str, float | str] = {
        "session": requested.strftime("%Y-%m-%d"),
        "source": "Nasdaq.com historical OHLC",
    }
    for symbol in ("TQQQ", "BIL"):
        date = requested.strftime("%Y-%m-%d")
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
        matching = [row for row in rows if pd.Timestamp(row["date"]).normalize() == requested]
        if len(matching) != 1:
            raise ValueError(f"Nasdaq has no unique completed OHLC row for {symbol} on {date}")
        row = matching[0]
        values[f"{symbol.lower()}_open"] = _number(row["open"])
        values[f"{symbol.lower()}_close"] = _number(row["close"])
    return values


def score_next(decision_ledger: str, outcome_ledger: str, total_cost_bps: float = 10.0) -> dict[str, object]:
    decisions = verify_paper_ledger(decision_ledger)
    outcomes = verify_outcome_ledger(outcome_ledger, decisions)
    scored_hashes = {outcome["decision_hash"] for outcome in outcomes}
    pending = [decision for decision in decisions if decision["record_hash"] not in scored_hashes]
    if not pending:
        raise ValueError("there are no unscored paper decisions")
    decision = pending[0]
    completed_bar = fetch_completed_bar(decision["effective_session"])
    previous = outcomes[-1] if outcomes else None
    outcome = compute_paper_outcome(
        decision,
        completed_bar,
        datetime.now(timezone.utc).isoformat(),
        previous,
        total_cost_bps,
    )
    append_paper_outcome(outcome_ledger, outcome, decisions)
    return outcome


def _number(value: object) -> float:
    return float(str(value).replace("$", "").replace(",", ""))


def main() -> int:
    parser = argparse.ArgumentParser(description="Score the oldest completed leveraged-trend paper decision.")
    parser.add_argument("--decisions", default="paper/leveraged_trend_decisions.jsonl")
    parser.add_argument("--outcomes", default="paper/leveraged_trend_outcomes.jsonl")
    parser.add_argument("--total-cost-bps", type=float, default=10.0)
    args = parser.parse_args()
    print(json.dumps(score_next(args.decisions, args.outcomes, args.total_cost_bps), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
