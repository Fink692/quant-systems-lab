from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FrozenPaperConfig:
    strategy_id: str = "leveraged-trend-v1"
    moving_average_days: int = 200
    volatility_lookback: int = 21
    target_volatility: float = 0.30
    max_exposure: float = 1.0
    source_tolerance_bps: float = 5.0

    def validate(self) -> None:
        if not self.strategy_id:
            raise ValueError("strategy_id is required")
        if min(self.moving_average_days, self.volatility_lookback) <= 1:
            raise ValueError("lookbacks must exceed one session")
        if min(self.target_volatility, self.max_exposure, self.source_tolerance_bps) <= 0:
            raise ValueError("risk and tolerance settings must be positive")


def compute_paper_decision(
    prices: pd.DataFrame,
    as_of_session: str | pd.Timestamp,
    effective_session: str | pd.Timestamp,
    created_utc: str,
    config: FrozenPaperConfig,
    input_data_sha256: str,
    independent_closes: dict[str, float],
    previous_record_hash: str = "GENESIS",
) -> dict[str, Any]:
    config.validate()
    as_of = pd.Timestamp(as_of_session).normalize()
    effective = pd.Timestamp(effective_session).normalize()
    if effective <= as_of:
        raise ValueError("effective_session must be strictly after as_of_session")
    required = ["tqqq", "qqq", "bil"]
    missing = set(required) - set(prices.columns)
    if missing:
        raise ValueError(f"missing price columns: {sorted(missing)}")
    history = prices.loc[prices.index <= as_of, required].copy().sort_index()
    if history.empty or history.index[-1] != as_of:
        raise ValueError("as_of_session must exist in the completed price history")
    if len(history) < max(config.moving_average_days, config.volatility_lookback + 1):
        raise ValueError("not enough completed sessions for the frozen signal")
    if history.isna().any().any() or (history <= 0).any().any():
        raise ValueError("paper signal prices must be complete and positive")

    returns = history["tqqq"].pct_change()
    moving_average = float(history["tqqq"].tail(config.moving_average_days).mean())
    realized_volatility = float(returns.tail(config.volatility_lookback).std(ddof=1) * np.sqrt(252.0))
    signal_price = float(history["tqqq"].iloc[-1])
    trend_on = bool(signal_price > moving_average)
    target_exposure = (
        min(config.max_exposure, config.target_volatility / max(realized_volatility, 1e-12)) if trend_on else 0.0
    )
    normalized_closes = {symbol.lower(): float(value) for symbol, value in independent_closes.items()}
    if set(required) - set(normalized_closes):
        raise ValueError("independent closes must include TQQQ, QQQ, and BIL")
    differences = {
        symbol: abs(float(history[symbol].iloc[-1]) / normalized_closes[symbol] - 1.0) * 10_000.0 for symbol in required
    }
    if max(differences.values()) > config.source_tolerance_bps:
        raise ValueError(f"independent close difference exceeds {config.source_tolerance_bps} bps")

    record: dict[str, Any] = {
        "schema_version": 1,
        "strategy_id": config.strategy_id,
        "created_utc": created_utc,
        "as_of_session": as_of.strftime("%Y-%m-%d"),
        "effective_session": effective.strftime("%Y-%m-%d"),
        "config_sha256": _sha256_json(asdict(config)),
        "input_data_sha256": input_data_sha256,
        "signal_price": signal_price,
        "moving_average": moving_average,
        "realized_volatility": realized_volatility,
        "trend_on": trend_on,
        "target_tqqq_exposure": float(target_exposure),
        "target_bil_exposure": float(1.0 - target_exposure),
        "independent_source": "Nasdaq.com completed-session close",
        "independent_closes": normalized_closes,
        "source_difference_bps": differences,
        "previous_record_hash": previous_record_hash,
    }
    record["record_hash"] = _record_hash(record)
    return record


def verify_paper_ledger(path: str | Path) -> list[dict[str, Any]]:
    ledger_path = Path(path)
    if not ledger_path.exists():
        return []
    records = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    previous = "GENESIS"
    previous_as_of: pd.Timestamp | None = None
    for record in records:
        claimed_hash = record.get("record_hash")
        if claimed_hash != _record_hash({key: value for key, value in record.items() if key != "record_hash"}):
            raise ValueError("paper ledger record hash mismatch")
        if record.get("previous_record_hash") != previous:
            raise ValueError("paper ledger chain is broken")
        as_of = pd.Timestamp(record["as_of_session"])
        if previous_as_of is not None and as_of <= previous_as_of:
            raise ValueError("paper ledger sessions must increase strictly")
        previous = str(claimed_hash)
        previous_as_of = as_of
    return records


def append_paper_decision(path: str | Path, record: dict[str, Any]) -> None:
    ledger_path = Path(path)
    records = verify_paper_ledger(ledger_path)
    expected_previous = records[-1]["record_hash"] if records else "GENESIS"
    if records and pd.Timestamp(record["as_of_session"]) <= pd.Timestamp(records[-1]["as_of_session"]):
        raise ValueError("cannot append a duplicate or older paper session")
    if record.get("previous_record_hash") != expected_previous:
        raise ValueError("new decision does not extend the current ledger head")
    if record.get("record_hash") != _record_hash({key: value for key, value in record.items() if key != "record_hash"}):
        raise ValueError("new decision hash is invalid")
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def compute_paper_outcome(
    decision: dict[str, Any],
    completed_bar: dict[str, float | str],
    scored_utc: str,
    previous_outcome: dict[str, Any] | None = None,
    total_cost_bps: float = 10.0,
) -> dict[str, Any]:
    if completed_bar.get("session") != decision.get("effective_session"):
        raise ValueError("completed bar must match the decision effective session")
    if total_cost_bps < 0:
        raise ValueError("total_cost_bps must be non-negative")
    required = ("tqqq_open", "tqqq_close", "bil_open", "bil_close")
    prices = {field: float(completed_bar[field]) for field in required}
    if min(prices.values()) <= 0:
        raise ValueError("completed OHLC prices must be positive")

    new_weight = float(decision["target_tqqq_exposure"])
    if not 0.0 <= new_weight <= 1.0:
        raise ValueError("decision exposure must be in [0, 1]")
    if previous_outcome is None:
        old_weight = 0.0
        overnight_return = 0.0
        previous_hash = "GENESIS"
    else:
        old_weight = float(previous_outcome["target_tqqq_exposure"])
        previous_tqqq_close = float(previous_outcome["tqqq_close"])
        previous_bil_close = float(previous_outcome["bil_close"])
        overnight_return = old_weight * (prices["tqqq_open"] / previous_tqqq_close - 1.0) + (1.0 - old_weight) * (
            prices["bil_open"] / previous_bil_close - 1.0
        )
        previous_hash = str(previous_outcome["outcome_hash"])
    intraday_return = new_weight * (prices["tqqq_close"] / prices["tqqq_open"] - 1.0) + (1.0 - new_weight) * (
        prices["bil_close"] / prices["bil_open"] - 1.0
    )
    turnover = abs(new_weight - old_weight)
    trading_cost = turnover * total_cost_bps / 10_000.0
    gross_return = (1.0 + overnight_return) * (1.0 + intraday_return) - 1.0

    outcome: dict[str, Any] = {
        "schema_version": 1,
        "strategy_id": decision["strategy_id"],
        "decision_hash": decision["record_hash"],
        "scored_utc": scored_utc,
        "effective_session": decision["effective_session"],
        "target_tqqq_exposure": new_weight,
        "target_bil_exposure": 1.0 - new_weight,
        **prices,
        "overnight_return": float(overnight_return),
        "intraday_return": float(intraday_return),
        "gross_return": float(gross_return),
        "turnover": float(turnover),
        "total_cost_bps": float(total_cost_bps),
        "trading_cost": float(trading_cost),
        "net_return": float(gross_return - trading_cost),
        "source": str(completed_bar.get("source", "Nasdaq.com historical OHLC")),
        "previous_outcome_hash": previous_hash,
    }
    outcome["outcome_hash"] = _record_hash(outcome)
    return outcome


def verify_outcome_ledger(
    path: str | Path,
    decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    outcome_path = Path(path)
    if not outcome_path.exists():
        return []
    decision_by_hash = {record["record_hash"]: record for record in decisions}
    outcomes = [json.loads(line) for line in outcome_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    previous = "GENESIS"
    previous_outcome: dict[str, Any] | None = None
    previous_session: pd.Timestamp | None = None
    for outcome in outcomes:
        claimed = outcome.get("outcome_hash")
        payload = {key: value for key, value in outcome.items() if key != "outcome_hash"}
        if claimed != _record_hash(payload):
            raise ValueError("paper outcome hash mismatch")
        if outcome.get("previous_outcome_hash") != previous:
            raise ValueError("paper outcome chain is broken")
        decision = decision_by_hash.get(outcome.get("decision_hash"))
        if decision is None or decision["effective_session"] != outcome.get("effective_session"):
            raise ValueError("paper outcome is not linked to a matching decision")
        _validate_outcome_calculation(outcome, decision, previous_outcome)
        session = pd.Timestamp(outcome["effective_session"])
        if previous_session is not None and session <= previous_session:
            raise ValueError("paper outcome sessions must increase strictly")
        previous = str(claimed)
        previous_outcome = outcome
        previous_session = session
    return outcomes


def append_paper_outcome(
    path: str | Path,
    outcome: dict[str, Any],
    decisions: list[dict[str, Any]],
) -> None:
    outcome_path = Path(path)
    outcomes = verify_outcome_ledger(outcome_path, decisions)
    if outcomes and pd.Timestamp(outcome["effective_session"]) <= pd.Timestamp(outcomes[-1]["effective_session"]):
        raise ValueError("cannot append a duplicate or older paper outcome")
    expected_previous = outcomes[-1]["outcome_hash"] if outcomes else "GENESIS"
    if outcome.get("previous_outcome_hash") != expected_previous:
        raise ValueError("new outcome does not extend the current outcome head")
    payload = {key: value for key, value in outcome.items() if key != "outcome_hash"}
    if outcome.get("outcome_hash") != _record_hash(payload):
        raise ValueError("new outcome hash is invalid")
    decision_hashes = {decision["record_hash"] for decision in decisions}
    if outcome.get("decision_hash") not in decision_hashes:
        raise ValueError("new outcome has no matching decision")
    decision = next(decision for decision in decisions if decision["record_hash"] == outcome["decision_hash"])
    _validate_outcome_calculation(outcome, decision, outcomes[-1] if outcomes else None)
    outcome_path.parent.mkdir(parents=True, exist_ok=True)
    with outcome_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(outcome, sort_keys=True, separators=(",", ":")) + "\n")


def canonical_file_sha256(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_immutable_text(path: str | Path, content: str) -> None:
    output = Path(path)
    canonical_content = content.replace("\r\n", "\n")
    if output.exists():
        existing = output.read_text(encoding="utf-8").replace("\r\n", "\n")
        if existing != canonical_content:
            raise ValueError(f"refusing to overwrite immutable evidence: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_content, encoding="utf-8", newline="")


def _record_hash(record: dict[str, Any]) -> str:
    return _sha256_json(record)


def _validate_outcome_calculation(
    outcome: dict[str, Any],
    decision: dict[str, Any],
    previous_outcome: dict[str, Any] | None,
) -> None:
    new_weight = float(decision["target_tqqq_exposure"])
    if outcome.get("strategy_id") != decision.get("strategy_id"):
        raise ValueError("paper outcome strategy does not match its decision")
    if not np.isclose(float(outcome["target_tqqq_exposure"]), new_weight, atol=1e-12):
        raise ValueError("paper outcome exposure does not match its decision")
    if not np.isclose(float(outcome["target_bil_exposure"]), 1.0 - new_weight, atol=1e-12):
        raise ValueError("paper outcome residual exposure is invalid")

    prices = {field: float(outcome[field]) for field in ("tqqq_open", "tqqq_close", "bil_open", "bil_close")}
    if min(prices.values()) <= 0:
        raise ValueError("paper outcome prices must be positive")
    if previous_outcome is None:
        old_weight = 0.0
        expected_overnight = 0.0
    else:
        old_weight = float(previous_outcome["target_tqqq_exposure"])
        expected_overnight = old_weight * (prices["tqqq_open"] / float(previous_outcome["tqqq_close"]) - 1.0) + (
            1.0 - old_weight
        ) * (prices["bil_open"] / float(previous_outcome["bil_close"]) - 1.0)
    expected_intraday = new_weight * (prices["tqqq_close"] / prices["tqqq_open"] - 1.0) + (1.0 - new_weight) * (
        prices["bil_close"] / prices["bil_open"] - 1.0
    )
    expected_turnover = abs(new_weight - old_weight)
    cost_bps = float(outcome["total_cost_bps"])
    if cost_bps < 0:
        raise ValueError("paper outcome cost must be non-negative")
    expected_cost = expected_turnover * cost_bps / 10_000.0
    expected_gross = (1.0 + expected_overnight) * (1.0 + expected_intraday) - 1.0
    expected = {
        "overnight_return": expected_overnight,
        "intraday_return": expected_intraday,
        "turnover": expected_turnover,
        "trading_cost": expected_cost,
        "gross_return": expected_gross,
        "net_return": expected_gross - expected_cost,
    }
    for field, value in expected.items():
        if not np.isclose(float(outcome[field]), value, atol=1e-12):
            raise ValueError(f"paper outcome {field} calculation is invalid")


def _sha256_json(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
