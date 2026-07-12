from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from quantlab.data.loaders import validate_order_book_events

LOBSTER_EVENT_NAMES = {
    1: "submission",
    2: "partial_cancel",
    3: "delete",
    4: "visible_execution",
    5: "hidden_execution",
    6: "cross_trade",
    7: "trading_halt",
}


@dataclass(frozen=True)
class LobsterDataset:
    events: pd.DataFrame
    snapshots: pd.DataFrame
    symbol: str
    session_date: date
    levels: int


def load_lobster_sample(
    message_path: str | Path,
    orderbook_path: str | Path,
    *,
    symbol: str,
    session_date: date,
    levels: int,
    tick_size: float = 0.01,
    exchange_timezone: str = "America/New_York",
    max_rows: int | None = None,
) -> LobsterDataset:
    """Load synchronized LOBSTER messages and snapshots into canonical frames."""
    if levels < 1 or tick_size <= 0:
        raise ValueError("levels and tick_size must be positive")
    raw_messages = pd.read_csv(message_path, header=None, nrows=max_rows)
    raw_books = pd.read_csv(orderbook_path, header=None, nrows=max_rows)
    if len(raw_messages) != len(raw_books):
        raise ValueError("LOBSTER message and order-book rows must be synchronized")
    if raw_messages.shape[1] != 6:
        raise ValueError("LOBSTER message file must contain six columns")
    if raw_books.shape[1] != 4 * levels:
        raise ValueError(f"LOBSTER order-book file must contain {4 * levels} columns for level {levels}")

    raw_messages.columns = ["seconds", "source_event_type", "order_id", "quantity", "price_raw", "direction"]
    event_type = raw_messages["source_event_type"].map(
        {1: "add", 2: "cancel", 3: "cancel", 4: "trade", 5: "trade", 6: "trade"}
    )
    keep = event_type.notna() & raw_messages["direction"].isin([-1, 1]) & (raw_messages["price_raw"] > 0)
    messages = raw_messages.loc[keep].copy()
    row_number = messages.index.to_numpy(dtype=np.int64)
    local_midnight = datetime.combine(session_date, datetime.min.time(), tzinfo=ZoneInfo(exchange_timezone))
    midnight_ns = int(local_midnight.astimezone(timezone.utc).timestamp() * 1e9)
    timestamp_ns = midnight_ns + np.rint(messages["seconds"].to_numpy(float) * 1e9).astype(np.int64)
    events = pd.DataFrame(
        {
            "timestamp_ns": timestamp_ns,
            "receive_timestamp_ns": timestamp_ns,
            "sequence_number": row_number,
            "event_type": event_type.loc[keep].to_numpy(),
            "side": np.where(messages["direction"].to_numpy(int) == 1, "bid", "ask"),
            "price": messages["price_raw"].to_numpy(float) / 10_000.0,
            "quantity": messages["quantity"].to_numpy(float),
            "order_id": messages["order_id"].astype(str).to_numpy(),
            "source_event_type": messages["source_event_type"].to_numpy(int),
            "source_event_name": messages["source_event_type"].map(LOBSTER_EVENT_NAMES).to_numpy(),
            "applies_to_visible_book": messages["source_event_type"].isin([1, 2, 3, 4]).to_numpy(),
            "source_row": row_number,
            "symbol": symbol,
            "venue": "NASDAQ",
        }
    )
    validation = validate_order_book_events(events, tick_size=tick_size)
    if not validation.is_valid:
        raise ValueError("; ".join(validation.issues))

    snapshots = _parse_orderbook(raw_books, levels=levels)
    snapshots.insert(0, "source_row", np.arange(len(snapshots), dtype=np.int64))
    snapshots.insert(
        1, "timestamp_ns", midnight_ns + np.rint(raw_messages["seconds"].to_numpy(float) * 1e9).astype(np.int64)
    )
    return LobsterDataset(events=events, snapshots=snapshots, symbol=symbol, session_date=session_date, levels=levels)


def _parse_orderbook(raw: pd.DataFrame, *, levels: int) -> pd.DataFrame:
    result: dict[str, np.ndarray] = {}
    for level in range(1, levels + 1):
        offset = (level - 1) * 4
        result[f"ask_price_{level}"] = raw.iloc[:, offset].to_numpy(float) / 10_000.0
        result[f"ask_quantity_{level}"] = raw.iloc[:, offset + 1].to_numpy(float)
        result[f"bid_price_{level}"] = raw.iloc[:, offset + 2].to_numpy(float) / 10_000.0
        result[f"bid_quantity_{level}"] = raw.iloc[:, offset + 3].to_numpy(float)
    return pd.DataFrame(result)
