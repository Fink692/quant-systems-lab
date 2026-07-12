from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from quantlab.market_data import build_quality_report, load_lobster_sample, reconstruct_and_reconcile
from quantlab.market_making.replay import ReplayConfig
from quantlab.research import (
    chronological_split_from_config,
    load_market_making_experiment_config,
    register_experiment,
    run_market_making_study,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the real-data market-making sample study.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/lobster_sample_experiment.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/market_making_sample"))
    parser.add_argument("--canonical-dir", type=Path, default=Path("data/processed/lobster_sample"))
    parser.add_argument("--registry-dir", type=Path, default=Path("research_runs"))
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--skip-registry", action="store_true")
    args = parser.parse_args()

    config = load_market_making_experiment_config(args.config)
    manifest_path = args.data_dir / "manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_fingerprint = str(manifest_payload["fingerprint"])
    if dataset_fingerprint != config.dataset.sha256:
        raise ValueError(
            f"dataset fingerprint mismatch: config={config.dataset.sha256}, manifest={dataset_fingerprint}"
        )
    message_path = _exactly_one(args.data_dir.rglob("*_message_5.csv"), "message file")
    orderbook_path = _exactly_one(args.data_dir.rglob("*_orderbook_5.csv"), "order-book file")
    dataset = load_lobster_sample(
        message_path,
        orderbook_path,
        symbol=config.dataset.symbol,
        session_date=date(2012, 6, 21),
        levels=5,
        tick_size=config.dataset.tick_size,
        max_rows=args.max_rows,
    )

    args.canonical_dir.mkdir(parents=True, exist_ok=True)
    dataset.events.to_parquet(args.canonical_dir / "events.parquet", index=False)
    dataset.snapshots.to_parquet(args.canonical_dir / "snapshots.parquet", index=False)

    reconstruction = reconstruct_and_reconcile(dataset.events, dataset.snapshots, levels=dataset.levels)
    quality = build_quality_report(dataset, reconstruction)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    quality_json = quality.write_json(args.output_dir / "data_quality.json")
    quality_markdown = quality.write_markdown(args.output_dir / "data_quality.md")

    execution = config.execution
    base_replay = ReplayConfig(
        tick_size=config.dataset.tick_size,
        order_size=execution.order_size,
        inventory_limit=execution.inventory_limit,
        maker_fee_bps=config.cost_model.maker_fee_bps,
        taker_fee_bps=config.cost_model.taker_fee_bps,
        fixed_fee_per_order=config.cost_model.fixed_fee_per_order,
        latency_ns=config.latency_ns[0],
        quote_interval_ns=execution.quote_interval_ns,
        queue_ahead_fraction=config.queue_ahead_fraction[0],
        toxicity_window=execution.toxicity_window,
        volatility_window=execution.volatility_window,
        adverse_horizon_ns=execution.adverse_horizon_ns,
        record_every=execution.record_every,
    )
    split = chronological_split_from_config(dataset, config)
    result = run_market_making_study(
        dataset,
        base_replay,
        queue_grid=config.queue_ahead_fraction,
        latency_grid_ns=config.latency_ns,
        fee_multiplier_grid=config.fee_multiplier,
        bootstrap_replicates=config.evaluation.bootstrap_replicates,
        bootstrap_block_size=config.evaluation.bootstrap_block_size,
        random_seed=config.random_seed,
        split=split,
        strategy_settings=config.strategy_parameters,
    )
    comparison = args.output_dir / "strategy_comparison.csv"
    sensitivity = args.output_dir / "sensitivity.csv"
    validation = args.output_dir / "validation_selection.csv"
    result.comparison.to_csv(comparison, index=False)
    result.sensitivity.to_csv(sensitivity, index=False)
    result.validation_scores.to_csv(validation, index=False)
    study = result.write_markdown(
        args.output_dir / "study.md",
        dataset_fingerprint=dataset_fingerprint,
        config_fingerprint=config.fingerprint,
    )
    summary = {
        "config_fingerprint": config.fingerprint,
        "dataset_fingerprint": dataset_fingerprint,
        "data_quality_status": quality.status,
        "reconstruction_match_rate": quality.reconstruction_match_rate,
        "selected_queue_ahead_fraction": result.selected_queue_ahead_fraction,
        "strategies": result.comparison.to_dict(orient="records"),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not args.skip_registry:
        run_dir = register_experiment(
            args.registry_dir,
            experiment_id=config.experiment_id,
            config_path=args.config,
            config_fingerprint=config.fingerprint,
            dataset_manifest_path=manifest_path,
            dataset_fingerprint=dataset_fingerprint,
            random_seed=config.random_seed,
            command="make reproduce-market-making-sample",
            artifacts={
                "data_quality.json": quality_json,
                "data_quality.md": quality_markdown,
                "strategy_comparison.csv": comparison,
                "sensitivity.csv": sensitivity,
                "validation_selection.csv": validation,
                "study.md": study,
                "summary.json": summary_path,
            },
        )
        summary["registry"] = str(run_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _exactly_one(paths, label: str) -> Path:
    matches = list(paths)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {label}, found {len(matches)}")
    return matches[0]


if __name__ == "__main__":
    main()
