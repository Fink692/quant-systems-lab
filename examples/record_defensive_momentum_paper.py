from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from quantlab.research.defensive_momentum import (
    FrozenDefensiveMomentumConfig,
    compute_defensive_momentum_decision,
    load_defensive_momentum_csv,
)
from quantlab.research.paper_trading import (
    append_paper_decision,
    canonical_file_sha256,
    verify_paper_ledger,
    write_immutable_text,
)


def record_decision(
    source_path: str | Path,
    source_metadata_path: str | Path,
    snapshot_path: str | Path,
    metadata_path: str | Path,
    ledger_path: str | Path,
    effective_session: str,
    config_path: str | Path,
) -> dict[str, object]:
    source = Path(source_path)
    source_metadata = json.loads(Path(source_metadata_path).read_text(encoding="utf-8"))
    if source_metadata["sha256"] != canonical_file_sha256(source):
        raise ValueError("source snapshot checksum does not match its metadata")
    inputs = load_defensive_momentum_csv(source)
    as_of = str(source_metadata["as_of_session"])
    if inputs.index[-1].strftime("%Y-%m-%d") != as_of:
        raise ValueError("source metadata does not match the final completed input session")
    paper_inputs = inputs.tail(320)
    snapshot = Path(snapshot_path)
    write_immutable_text(
        snapshot,
        paper_inputs.to_csv(date_format="%Y-%m-%d", float_format="%.10f", lineterminator="\n"),
    )
    paper_metadata = {
        "as_of_session": as_of,
        "rows": len(paper_inputs),
        "source_snapshot_sha256": source_metadata["sha256"],
        "snapshot_sha256": canonical_file_sha256(snapshot),
        "independent_source": source_metadata["independent_source"],
        "independent_closes": source_metadata["independent_closes"],
        "source_difference_bps": source_metadata["source_difference_bps"],
    }
    write_immutable_text(metadata_path, json.dumps(paper_metadata, indent=2, sort_keys=True) + "\n")
    raw_config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    config = FrozenDefensiveMomentumConfig(**raw_config)
    ledger = verify_paper_ledger(ledger_path)
    previous = ledger[-1]["record_hash"] if ledger else "GENESIS"
    decision = compute_defensive_momentum_decision(
        paper_inputs,
        as_of,
        effective_session,
        datetime.now(timezone.utc).isoformat(),
        config,
        str(paper_metadata["snapshot_sha256"]),
        dict(paper_metadata["independent_closes"]),
        previous,
    )
    append_paper_decision(ledger_path, decision)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description="Append a frozen defensive-momentum paper decision.")
    parser.add_argument("--source", default="data/real/defensive_momentum_ohlc.csv")
    parser.add_argument("--source-metadata", default="data/real/defensive_momentum_ohlc.metadata.json")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--ledger", default="paper/defensive_momentum_decisions.jsonl")
    parser.add_argument("--effective-session", required=True)
    parser.add_argument("--config", default="config/defensive_momentum_paper.json")
    args = parser.parse_args()
    decision = record_decision(
        args.source,
        args.source_metadata,
        args.snapshot,
        args.metadata,
        args.ledger,
        args.effective_session,
        args.config,
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
