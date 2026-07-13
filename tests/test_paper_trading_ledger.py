from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from quantlab.research.paper_trading import (
    FrozenPaperConfig,
    append_paper_decision,
    append_paper_outcome,
    canonical_file_sha256,
    compute_paper_decision,
    compute_paper_outcome,
    verify_outcome_ledger,
    verify_paper_ledger,
    write_immutable_text,
)

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data" / "paper" / "leveraged_trend_inputs_2026-07-13.csv"
LEDGER = ROOT / "paper" / "leveraged_trend_decisions.jsonl"


def _prices() -> pd.DataFrame:
    return pd.read_csv(SNAPSHOT, parse_dates=["date"]).set_index("date")


def test_committed_paper_decision_is_hash_chained_and_source_reconciled():
    records = verify_paper_ledger(LEDGER)
    assert len(records) == 1
    record = records[0]
    assert record["as_of_session"] == "2026-07-13"
    assert record["effective_session"] == "2026-07-14"
    assert record["input_data_sha256"] == canonical_file_sha256(SNAPSHOT)
    assert record["previous_record_hash"] == "GENESIS"
    assert record["trend_on"] is True
    assert 0.30 < record["target_tqqq_exposure"] < 0.40
    assert max(record["source_difference_bps"].values()) < 1.0


def test_paper_signal_ignores_future_rows():
    prices = _prices()
    future = prices.iloc[-1:].copy()
    future.index = future.index + pd.Timedelta(days=1)
    future.loc[:, "tqqq"] = 1_000_000.0
    extended = pd.concat([prices, future])
    kwargs = {
        "as_of_session": "2026-07-13",
        "effective_session": "2026-07-14",
        "created_utc": "2026-07-13T21:00:00+00:00",
        "config": FrozenPaperConfig(),
        "input_data_sha256": "abc",
        "independent_closes": {"tqqq": 72.64, "qqq": 711.74, "bil": 91.50},
    }
    baseline = compute_paper_decision(prices, **kwargs)
    mutated = compute_paper_decision(extended, **kwargs)
    assert baseline == mutated


def test_ledger_rejects_duplicate_and_tampering(tmp_path):
    original = verify_paper_ledger(LEDGER)[0]
    ledger = tmp_path / "ledger.jsonl"
    append_paper_decision(ledger, original)
    with pytest.raises(ValueError, match="duplicate or older"):
        append_paper_decision(ledger, original)

    tampered = copy.deepcopy(original)
    tampered["target_tqqq_exposure"] = 0.99
    ledger.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_paper_ledger(ledger)

    evidence = tmp_path / "evidence.csv"
    write_immutable_text(evidence, "date,value\n2026-07-13,1\n")
    write_immutable_text(evidence, "date,value\r\n2026-07-13,1\r\n")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_immutable_text(evidence, "date,value\n2026-07-13,2\n")


def test_paper_outcome_links_decision_and_charges_next_open_turnover(tmp_path):
    decision = verify_paper_ledger(LEDGER)[0]
    bar = {
        "session": "2026-07-14",
        "tqqq_open": 70.0,
        "tqqq_close": 71.0,
        "bil_open": 91.5,
        "bil_close": 91.51,
        "source": "test completed OHLC",
    }
    outcome = compute_paper_outcome(decision, bar, "2026-07-14T21:00:00+00:00", total_cost_bps=10.0)
    weight = decision["target_tqqq_exposure"]
    expected_intraday = weight * (71.0 / 70.0 - 1.0) + (1.0 - weight) * (91.51 / 91.5 - 1.0)
    assert outcome["overnight_return"] == 0.0
    assert outcome["turnover"] == weight
    assert outcome["net_return"] == pytest.approx(expected_intraday - weight * 0.001)
    assert outcome["decision_hash"] == decision["record_hash"]

    path = tmp_path / "outcomes.jsonl"
    decisions = [decision]
    append_paper_outcome(path, outcome, decisions)
    assert verify_outcome_ledger(path, decisions) == [outcome]
    with pytest.raises(ValueError, match="duplicate or older"):
        append_paper_outcome(path, outcome, decisions)

    tampered = copy.deepcopy(outcome)
    tampered["net_return"] = 1.0
    path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outcome hash mismatch"):
        verify_outcome_ledger(path, decisions)

    tampered["outcome_hash"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in tampered.items() if key != "outcome_hash"},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="net_return calculation is invalid"):
        verify_outcome_ledger(path, decisions)
