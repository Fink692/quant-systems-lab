from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from quantlab.research.defensive_momentum import (
    FrozenDefensiveMomentumConfig,
    compute_defensive_momentum_decision,
    load_defensive_momentum_csv,
)
from quantlab.research.paper_trading import canonical_file_sha256, verify_paper_ledger

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data" / "paper" / "defensive_momentum_inputs_2026-07-13.csv"
LEDGER = ROOT / "paper" / "defensive_momentum_decisions.jsonl"


def _inputs():
    return load_defensive_momentum_csv(SNAPSHOT)


def test_committed_defensive_momentum_genesis_is_frozen_and_hash_chained():
    records = verify_paper_ledger(LEDGER)
    assert len(records) == 1
    record = records[0]
    assert record["strategy_id"] == "defensive-momentum-monthly-v1"
    assert record["as_of_session"] == "2026-07-13"
    assert record["effective_session"] == "2026-07-14"
    assert record["signal_session"] == "2026-06-30"
    assert record["selected_asset"] == "qqq"
    assert 1.0 < record["selected_leverage"] < 1.1
    assert record["target_weights"]["cash"] < 0.0
    assert record["input_data_sha256"] == canonical_file_sha256(SNAPSHOT)
    assert max(record["source_difference_bps"].values()) < 5.0


def test_monthly_decision_ignores_rows_after_as_of_and_uses_completed_month():
    inputs = _inputs()
    future = inputs.iloc[-1:].copy()
    future.index = future.index + pd.Timedelta(days=1)
    for column in future:
        if column != "dff":
            future.loc[:, column] = 1_000_000.0
    kwargs = {
        "as_of_session": "2026-07-13",
        "effective_session": "2026-07-14",
        "created_utc": "2026-07-13T22:00:00+00:00",
        "config": FrozenDefensiveMomentumConfig(),
        "input_data_sha256": "abc",
        "independent_closes": {"qqq": 711.74, "gld": 367.2, "tlt": 83.97},
    }
    baseline = compute_defensive_momentum_decision(inputs, **kwargs)
    mutated = compute_defensive_momentum_decision(pd.concat([inputs, future]), **kwargs)
    assert baseline == mutated
    assert baseline["signal_session"] == "2026-06-30"


def test_monthly_decision_rejects_source_disagreement_and_ledger_tampering(tmp_path):
    inputs = _inputs()
    kwargs = {
        "as_of_session": "2026-07-13",
        "effective_session": "2026-07-14",
        "created_utc": "2026-07-13T22:00:00+00:00",
        "config": FrozenDefensiveMomentumConfig(),
        "input_data_sha256": "abc",
        "independent_closes": {"qqq": 700.0, "gld": 367.2, "tlt": 83.97},
    }
    with pytest.raises(ValueError, match="independent close difference"):
        compute_defensive_momentum_decision(inputs, **kwargs)

    record = verify_paper_ledger(LEDGER)[0]
    tampered = copy.deepcopy(record)
    tampered["selected_leverage"] = 1.5
    path = tmp_path / "ledger.jsonl"
    path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_paper_ledger(path)
