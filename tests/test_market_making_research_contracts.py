from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pandas as pd
import pytest

from quantlab.data import load_order_book_events_csv, validate_order_book_events
from quantlab.research import load_market_making_experiment_config

ROOT = Path(__file__).resolve().parents[1]


def _valid_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_ns": [1_000_000_000, 1_000_000_000, 1_000_000_050, 1_000_000_100],
            "receive_timestamp_ns": [1_000_000_010, 1_000_000_020, 1_000_000_060, 1_000_000_120],
            "sequence_number": [100, 101, 102, 103],
            "event_type": ["add", "add", "trade", "cancel"],
            "side": ["bid", "ask", "ask", "bid"],
            "price": [99.99, 100.01, 100.01, 99.99],
            "quantity": [20.0, 18.0, 2.0, 5.0],
            "order_id": ["b-1", "a-1", "a-1", "b-1"],
        }
    )


def test_order_book_contract_accepts_ordered_l3_events_and_csv_round_trip(tmp_path: Path) -> None:
    events = _valid_events()
    assert validate_order_book_events(events, tick_size=0.01).is_valid

    path = tmp_path / "events.csv"
    events.to_csv(path, index=False)
    loaded = load_order_book_events_csv(path, tick_size=0.01)
    assert loaded["timestamp_ns"].dtype == "int64"
    assert loaded["sequence_number"].tolist() == [100, 101, 102, 103]


@pytest.mark.parametrize(
    ("mutation", "expected_issue"),
    [
        (lambda frame: frame.assign(sequence_number=[100, 102, 101, 103]), "strictly increasing"),
        (lambda frame: frame.assign(timestamp_ns=[1, 3, 2, 4]), "non-decreasing"),
        (lambda frame: frame.assign(event_type=["add", "quote", "trade", "cancel"]), "event_type"),
        (lambda frame: frame.assign(side=["bid", "buy", "ask", "bid"]), "side"),
        (lambda frame: frame.assign(price=[99.99, 100.005, 100.01, 99.99]), "tick grid"),
        (
            lambda frame: frame.assign(receive_timestamp_ns=frame["timestamp_ns"] - 1),
            "must not precede",
        ),
    ],
)
def test_order_book_contract_rejects_corrupt_or_ambiguous_events(mutation, expected_issue: str) -> None:
    result = validate_order_book_events(mutation(_valid_events()), tick_size=0.01)
    assert not result.is_valid
    assert any(expected_issue in issue for issue in result.issues)


def test_order_book_contract_allows_sub_tick_hidden_executions() -> None:
    events = _valid_events()
    events.loc[2, "price"] = 100.005
    events["applies_to_visible_book"] = [True, True, False, True]
    assert validate_order_book_events(events, tick_size=0.01).is_valid


def test_flagship_config_is_frozen_validated_and_fingerprinted() -> None:
    path = ROOT / "config" / "market_making_flagship.example.json"
    first = load_market_making_experiment_config(path)
    second = load_market_making_experiment_config(path)

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64
    assert first.dataset.book_level == "L3"
    assert first.strategies[0] == "fixed_spread"
    with pytest.raises(FrozenInstanceError):
        first.random_seed = 11  # type: ignore[misc]


def test_flagship_config_rejects_overlapping_periods(tmp_path: Path) -> None:
    source = (ROOT / "config" / "market_making_flagship.example.json").read_text(encoding="utf-8")
    broken = source.replace(
        '"start": "2024-04-01T00:00:00Z", "end": "2024-05-01T00:00:00Z"',
        '"start": "2024-03-01T00:00:00Z", "end": "2024-05-01T00:00:00Z"',
    )
    path = tmp_path / "overlap.json"
    path.write_text(broken, encoding="utf-8")
    with pytest.raises(ValueError, match="chronological and non-overlapping"):
        load_market_making_experiment_config(path)


def test_flagship_config_rejects_unregistered_fields(tmp_path: Path) -> None:
    source = (ROOT / "config" / "market_making_flagship.example.json").read_text(encoding="utf-8")
    broken = source.replace('"random_seed": 7,', '"random_seed": 7,\n  "analyst_note": "not part of the schema",')
    path = tmp_path / "unknown-field.json"
    path.write_text(broken, encoding="utf-8")
    with pytest.raises(ValueError, match="unknown=.*analyst_note"):
        load_market_making_experiment_config(path)
