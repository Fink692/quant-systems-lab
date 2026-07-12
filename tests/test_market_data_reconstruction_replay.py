from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from quantlab.market_data import (
    build_dataset_manifest,
    build_quality_report,
    load_lobster_sample,
    reconstruct_and_reconcile,
)
from quantlab.market_making.replay import ReplayConfig, replay_market_maker
from quantlab.market_making.strategies import FixedSpreadPolicy
from quantlab.research import chronological_split
from quantlab.research.registry import register_experiment


def _lobster_fixture(tmp_path: Path):
    messages = pd.DataFrame(
        [
            [34200.0, 1, 1, 10, 1_000_000, 1],
            [34201.0, 1, 2, 10, 1_000_100, -1],
            [34202.0, 4, 2, 15, 1_000_100, -1],
            [34203.0, 2, 1, 5, 1_000_000, 1],
            [34204.0, 5, 0, 3, 1_000_050, -1],
            [34205.0, 1, 3, 5, 1_000_000, 1],
            [34206.0, 4, 3, 8, 1_000_000, 1],
            [34207.0, 3, 2, 15, 1_000_100, -1],
        ]
    )
    books = pd.DataFrame(
        [
            [1_000_100, 20, 1_000_000, 10, 1_000_200, 40, 999_900, 40],
            [1_000_100, 30, 1_000_000, 10, 1_000_200, 40, 999_900, 40],
            [1_000_100, 15, 1_000_000, 10, 1_000_200, 40, 999_900, 40],
            [1_000_100, 15, 1_000_000, 5, 1_000_200, 40, 999_900, 40],
            [1_000_100, 15, 1_000_000, 5, 1_000_200, 40, 999_900, 40],
            [1_000_100, 15, 1_000_000, 10, 1_000_200, 40, 999_900, 40],
            [1_000_100, 15, 1_000_000, 2, 1_000_200, 40, 999_900, 40],
            [1_000_200, 40, 1_000_000, 2, 9_999_999_999, 0, 999_900, 40],
        ]
    )
    message_path = tmp_path / "fixture_message_2.csv"
    book_path = tmp_path / "fixture_orderbook_2.csv"
    messages.to_csv(message_path, header=False, index=False)
    books.to_csv(book_path, header=False, index=False)
    dataset = load_lobster_sample(
        message_path,
        book_path,
        symbol="TEST",
        session_date=date(2012, 6, 21),
        levels=2,
        tick_size=0.01,
    )
    return dataset, message_path, book_path


def test_lobster_ingestion_reconstruction_quality_and_timezone(tmp_path: Path) -> None:
    dataset, _, _ = _lobster_fixture(tmp_path)
    assert str(pd.Timestamp(dataset.events["timestamp_ns"].iloc[0], unit="ns", tz="UTC")) == "2012-06-21 13:30:00+00:00"
    assert dataset.events.loc[dataset.events["source_event_type"] == 5, "price"].iloc[0] == 100.005
    reconstruction = reconstruct_and_reconcile(dataset.events, dataset.snapshots, levels=2)
    assert reconstruction.match_rate == 1.0
    assert reconstruction.invalid_cancels == 0
    report = build_quality_report(dataset, reconstruction, minimum_match_rate=1.0)
    assert report.status == "pass"
    assert report.crossed_snapshots == 0


def test_shared_replay_fills_both_sides_and_reconciles_cash(tmp_path: Path) -> None:
    dataset, _, _ = _lobster_fixture(tmp_path)
    result = replay_market_maker(
        dataset.events,
        dataset.snapshots,
        FixedSpreadPolicy(),
        ReplayConfig(
            tick_size=0.01,
            order_size=5,
            inventory_limit=20,
            maker_fee_bps=0.0,
            taker_fee_bps=0.0,
            quote_interval_ns=1,
            queue_ahead_fraction=0.0,
            toxicity_window=2,
            volatility_window=2,
            record_every=1,
        ),
    )
    strategy_fills = result.fills.loc[~result.fills["liquidation"]]
    assert set(strategy_fills["side"]) == {"bid", "ask"}
    assert abs(result.accounting_error) < 1e-12
    assert result.max_abs_inventory <= 20
    assert result.history["inventory"].iloc[-1] == 0


def test_manifest_fingerprint_is_content_stable_and_registry_is_append_only(tmp_path: Path) -> None:
    _, message_path, book_path = _lobster_fixture(tmp_path)
    kwargs = {
        "provider": "LOBSTER",
        "dataset_id": "fixture",
        "source_url": "https://example.test/sample",
        "license_url": "https://example.test/license",
    }
    first = build_dataset_manifest([message_path, book_path], acquired_at_utc="2024-01-01T00:00:00Z", **kwargs)
    second = build_dataset_manifest([message_path, book_path], acquired_at_utc="2025-01-01T00:00:00Z", **kwargs)
    assert first.fingerprint == second.fingerprint
    manifest_path = first.write(tmp_path / "manifest.json")
    config_path = tmp_path / "config.json"
    artifact_path = tmp_path / "report.md"
    config_path.write_text("{}\n", encoding="utf-8")
    artifact_path.write_text("result\n", encoding="utf-8")
    run = register_experiment(
        tmp_path / "registry",
        experiment_id="fixture",
        config_path=config_path,
        config_fingerprint="a" * 64,
        dataset_manifest_path=manifest_path,
        dataset_fingerprint=first.fingerprint,
        random_seed=7,
        command="pytest",
        artifacts={"report.md": artifact_path},
    )
    assert (run / "record.json").exists()
    with pytest.raises(FileExistsError):
        register_experiment(
            tmp_path / "registry",
            experiment_id="fixture",
            config_path=config_path,
            config_fingerprint="a" * 64,
            dataset_manifest_path=manifest_path,
            dataset_fingerprint=first.fingerprint,
            random_seed=7,
            command="pytest",
            artifacts={"report.md": artifact_path},
        )


def test_chronological_split_has_embargo_and_no_shared_rows(tmp_path: Path) -> None:
    dataset, _, _ = _lobster_fixture(tmp_path)
    split = chronological_split(dataset, train_fraction=0.5, validation_fraction=0.25, embargo_ns=0)
    train_rows = set(split.train_snapshots["source_row"])
    validation_rows = set(split.validation_snapshots["source_row"])
    test_rows = set(split.test_snapshots["source_row"])
    assert train_rows.isdisjoint(validation_rows | test_rows)
    assert validation_rows.isdisjoint(test_rows)
    assert split.train_end_ns <= split.validation_start_ns <= split.validation_end_ns <= split.test_start_ns
