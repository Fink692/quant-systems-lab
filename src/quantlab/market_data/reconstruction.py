from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BookSnapshot:
    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]

    @property
    def best_bid(self) -> float | None:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0][0] if self.asks else None

    @property
    def mid(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return 0.5 * (self.best_bid + self.best_ask)


@dataclass(frozen=True)
class ReconstructionReport:
    observations: int
    compared: int
    exact_matches: int
    mismatches: int
    invalid_cancels: int
    crossed_books: int
    match_rate: float
    mismatch_rows: tuple[int, ...]


class ReconstructedBook:
    """Market-by-price reconstruction with explicit invalid-cancel accounting."""

    def __init__(self) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.invalid_cancels = 0

    def seed(self, snapshot: BookSnapshot) -> None:
        self.bids = dict(snapshot.bids)
        self.asks = dict(snapshot.asks)

    def apply(self, event: pd.Series) -> None:
        if not bool(event.get("applies_to_visible_book", True)):
            return
        side = str(event["side"])
        levels = self.bids if side == "bid" else self.asks
        price = float(event["price"])
        quantity = float(event["quantity"])
        kind = str(event["event_type"])
        if kind == "add":
            levels[price] = levels.get(price, 0.0) + quantity
            return
        if kind in {"cancel", "trade"}:
            available = levels.get(price, 0.0)
            if available + 1e-9 < quantity:
                self.invalid_cancels += 1
            remaining = max(available - quantity, 0.0)
            if remaining <= 1e-9:
                levels.pop(price, None)
            else:
                levels[price] = remaining

    def snapshot(self, depth: int) -> BookSnapshot:
        bids = tuple((price, self.bids[price]) for price in sorted(self.bids, reverse=True)[:depth])
        asks = tuple((price, self.asks[price]) for price in sorted(self.asks)[:depth])
        return BookSnapshot(bids=bids, asks=asks)

    @property
    def is_crossed(self) -> bool:
        return bool(self.bids and self.asks and max(self.bids) >= min(self.asks))


def snapshot_from_row(row: pd.Series, levels: int) -> BookSnapshot:
    asks = tuple(
        (float(row[f"ask_price_{level}"]), float(row[f"ask_quantity_{level}"]))
        for level in range(1, levels + 1)
        if float(row[f"ask_quantity_{level}"]) > 0
    )
    bids = tuple(
        (float(row[f"bid_price_{level}"]), float(row[f"bid_quantity_{level}"]))
        for level in range(1, levels + 1)
        if float(row[f"bid_quantity_{level}"]) > 0
    )
    return BookSnapshot(bids=bids, asks=asks)


def reconstruct_and_reconcile(
    events: pd.DataFrame,
    snapshots: pd.DataFrame,
    *,
    levels: int,
    reseed_on_mismatch: bool = True,
    max_mismatch_rows: int = 100,
) -> ReconstructionReport:
    """Replay visible events and compare each synchronized provider snapshot."""
    if snapshots.empty:
        raise ValueError("snapshots cannot be empty")
    event_by_row = events.set_index("source_row", drop=False)
    book = ReconstructedBook()
    book.seed(snapshot_from_row(snapshots.iloc[0], levels))
    matches = 0
    compared = 0
    mismatches: list[int] = []
    crossed = int(book.is_crossed)
    for position in range(1, len(snapshots)):
        if position in event_by_row.index:
            event = event_by_row.loc[position]
            if isinstance(event, pd.DataFrame):
                for _, item in event.iterrows():
                    book.apply(item)
            else:
                book.apply(event)
        expected = snapshot_from_row(snapshots.iloc[position], levels)
        actual = book.snapshot(levels)
        compared += 1
        if _snapshots_equal(actual, expected):
            matches += 1
        else:
            if len(mismatches) < max_mismatch_rows:
                mismatches.append(position)
            if reseed_on_mismatch:
                book.seed(expected)
        crossed += int(book.is_crossed)
    mismatch_count = compared - matches
    return ReconstructionReport(
        observations=len(snapshots),
        compared=compared,
        exact_matches=matches,
        mismatches=mismatch_count,
        invalid_cancels=book.invalid_cancels,
        crossed_books=crossed,
        match_rate=float(matches / compared) if compared else 1.0,
        mismatch_rows=tuple(mismatches),
    )


def _snapshots_equal(left: BookSnapshot, right: BookSnapshot) -> bool:
    if len(left.bids) != len(right.bids) or len(left.asks) != len(right.asks):
        return False
    return all(np.allclose(a, b, atol=1e-9, rtol=0.0) for a, b in zip(left.bids + left.asks, right.bids + right.asks))
