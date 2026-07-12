from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.market_data.lobster import LobsterDataset
from quantlab.market_making.replay import ReplayConfig, ReplayResult, replay_market_maker
from quantlab.market_making.strategies import (
    AvellanedaStoikovPolicy,
    FixedSpreadPolicy,
    LatencyAwarePolicy,
    QueueAwarePolicy,
    QuotingPolicy,
    ToxicityAwarePolicy,
)
from quantlab.research.experiment_config import MarketMakingExperimentConfig, StrategySettings


@dataclass(frozen=True)
class ChronologicalDataSplit:
    train_events: pd.DataFrame
    train_snapshots: pd.DataFrame
    validation_events: pd.DataFrame
    validation_snapshots: pd.DataFrame
    test_events: pd.DataFrame
    test_snapshots: pd.DataFrame
    train_end_ns: int
    validation_start_ns: int
    validation_end_ns: int
    test_start_ns: int


@dataclass(frozen=True)
class MarketCalibration:
    duration_seconds: float
    add_intensity: float
    cancel_intensity: float
    visible_trade_intensity: float
    hidden_trade_intensity: float
    median_spread: float
    spread_transition_probability: float
    mid_change_volatility: float
    average_event_latency_ns: float
    average_signed_move_1s: float


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float


@dataclass(frozen=True)
class MarketMakingStudyResult:
    calibration: MarketCalibration
    selected_queue_ahead_fraction: float
    validation_scores: pd.DataFrame
    test_results: tuple[ReplayResult, ...]
    comparison: pd.DataFrame
    bootstrap_net_pnl_vs_fixed: dict[str, BootstrapInterval]
    sensitivity: pd.DataFrame
    split: ChronologicalDataSplit

    def write_markdown(self, path: str | Path, *, dataset_fingerprint: str, config_fingerprint: str) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        comparison = self.comparison.to_markdown(index=False, floatfmt=".6f")
        sensitivity = self.sensitivity.to_markdown(index=False, floatfmt=".6f")
        intervals = "\n".join(
            f"| {name} | {item.estimate:.6f} | {item.lower:.6f} | {item.upper:.6f} |"
            for name, item in self.bootstrap_net_pnl_vs_fixed.items()
        )
        calibration = "\n".join(f"| {name} | {value:.6f} |" for name, value in asdict(self.calibration).items())
        text = f"""# Queue-Aware Market-Making Sample Study

## Status

This is a **pipeline-validation study**, not flagship strategy evidence. It uses one public LOBSTER demonstration session. The chronological split, replay, accounting, sensitivity, and reporting paths are real; the sample is too short for claims about persistent alpha.

- Dataset fingerprint: `{dataset_fingerprint}`
- Configuration fingerprint: `{config_fingerprint}`
- Selected validation queue-ahead fraction: `{self.selected_queue_ahead_fraction:.2f}`

## Training-period calibration

| Metric | Value |
| --- | ---: |
{calibration}

## Untouched chronological test

{comparison}

## Block-bootstrap net-PnL difference versus fixed spread

| Strategy | Estimate | 95% lower | 95% upper |
| --- | ---: | ---: | ---: |
{intervals}

Intervals resample contiguous PnL blocks and describe this session only. They are not substitutes for session-level inference across many trading days.

## Predeclared sensitivity

{sensitivity}

## Failure interpretation

Negative net PnL, wide uncertainty, sensitivity to queue position, or degradation under latency are retained as research results. No strategy is promoted based on this demonstration session. A publishable paper requires licensed multi-session data and a later untouched test interval.
"""
        output.write_text(text, encoding="utf-8")
        return output


def chronological_split(
    dataset: LobsterDataset,
    *,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    embargo_ns: int = 1_000_000_000,
) -> ChronologicalDataSplit:
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("split fractions must be positive and sum to less than one")
    timestamps = dataset.snapshots["timestamp_ns"].to_numpy(np.int64)
    start, end = int(timestamps[0]), int(timestamps[-1])
    train_end = start + int((end - start) * train_fraction)
    validation_start = train_end + embargo_ns
    validation_end = start + int((end - start) * (train_fraction + validation_fraction))
    test_start = validation_end + embargo_ns
    if validation_start >= validation_end or test_start >= end:
        raise ValueError("embargo leaves an empty split")

    def segment(lower: int, upper: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        snapshots = dataset.snapshots.loc[
            dataset.snapshots["timestamp_ns"].between(lower, upper, inclusive="left")
        ].copy()
        rows = set(snapshots["source_row"].astype(int))
        events = dataset.events.loc[dataset.events["source_row"].isin(rows)].copy()
        return events, snapshots

    train_events, train_snapshots = segment(start, train_end)
    validation_events, validation_snapshots = segment(validation_start, validation_end)
    test_events, test_snapshots = segment(test_start, end + 1)
    return ChronologicalDataSplit(
        train_events,
        train_snapshots,
        validation_events,
        validation_snapshots,
        test_events,
        test_snapshots,
        train_end,
        validation_start,
        validation_end,
        test_start,
    )


def chronological_split_from_config(
    dataset: LobsterDataset, config: MarketMakingExperimentConfig
) -> ChronologicalDataSplit:
    boundaries = {
        "train_start": int(pd.Timestamp(config.train.start).timestamp() * 1e9),
        "train_end": int(pd.Timestamp(config.train.end).timestamp() * 1e9),
        "validation_start": int(pd.Timestamp(config.validation.start).timestamp() * 1e9),
        "validation_end": int(pd.Timestamp(config.validation.end).timestamp() * 1e9),
        "test_start": int(pd.Timestamp(config.test.start).timestamp() * 1e9),
        "test_end": int(pd.Timestamp(config.test.end).timestamp() * 1e9),
    }

    def segment(lower: int, upper: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        snapshots = dataset.snapshots.loc[
            dataset.snapshots["timestamp_ns"].between(lower, upper, inclusive="left")
        ].copy()
        rows = set(snapshots["source_row"].astype(int))
        return dataset.events.loc[dataset.events["source_row"].isin(rows)].copy(), snapshots

    train_events, train_snapshots = segment(boundaries["train_start"], boundaries["train_end"])
    validation_events, validation_snapshots = segment(boundaries["validation_start"], boundaries["validation_end"])
    test_events, test_snapshots = segment(boundaries["test_start"], boundaries["test_end"])
    if min(len(train_snapshots), len(validation_snapshots), len(test_snapshots)) < 2:
        raise ValueError("configured periods do not cover enough dataset observations")
    return ChronologicalDataSplit(
        train_events,
        train_snapshots,
        validation_events,
        validation_snapshots,
        test_events,
        test_snapshots,
        boundaries["train_end"],
        boundaries["validation_start"],
        boundaries["validation_end"],
        boundaries["test_start"],
    )


def calibrate_market_data(events: pd.DataFrame, snapshots: pd.DataFrame) -> MarketCalibration:
    duration = max((snapshots["timestamp_ns"].iloc[-1] - snapshots["timestamp_ns"].iloc[0]) / 1e9, 1e-9)
    counts = events["source_event_name"].value_counts()
    spreads = snapshots["ask_price_1"].to_numpy(float) - snapshots["bid_price_1"].to_numpy(float)
    mids = 0.5 * (snapshots["ask_price_1"].to_numpy(float) + snapshots["bid_price_1"].to_numpy(float))
    transitions = float(np.mean(np.diff(spreads) != 0)) if len(spreads) > 1 else 0.0
    signed_moves: list[float] = []
    snapshot_times = snapshots["timestamp_ns"].to_numpy(np.int64)
    row_to_position = {int(row): index for index, row in enumerate(snapshots["source_row"])}
    for event in events.loc[events["event_type"] == "trade"].itertuples(index=False):
        position = row_to_position.get(int(event.source_row))
        if position is None:
            continue
        future = int(np.searchsorted(snapshot_times, int(event.timestamp_ns) + 1_000_000_000, side="left"))
        if future < len(mids):
            aggressor_sign = 1.0 if event.side == "ask" else -1.0
            signed_moves.append(aggressor_sign * (mids[future] - mids[position]))
    return MarketCalibration(
        duration_seconds=float(duration),
        add_intensity=float(counts.get("submission", 0) / duration),
        cancel_intensity=float((counts.get("partial_cancel", 0) + counts.get("delete", 0)) / duration),
        visible_trade_intensity=float(counts.get("visible_execution", 0) / duration),
        hidden_trade_intensity=float(counts.get("hidden_execution", 0) / duration),
        median_spread=float(np.median(spreads)),
        spread_transition_probability=transitions,
        mid_change_volatility=float(np.std(np.diff(mids), ddof=1)) if len(mids) > 2 else 0.0,
        average_event_latency_ns=float((events["receive_timestamp_ns"] - events["timestamp_ns"]).mean()),
        average_signed_move_1s=float(np.mean(signed_moves)) if signed_moves else 0.0,
    )


def run_market_making_study(
    dataset: LobsterDataset,
    base_config: ReplayConfig,
    *,
    queue_grid: tuple[float, ...] = (0.0, 0.5, 1.0),
    latency_grid_ns: tuple[int, ...] = (0, 1_000_000, 5_000_000),
    fee_multiplier_grid: tuple[float, ...] = (1.0,),
    bootstrap_replicates: int = 500,
    bootstrap_block_size: int = 20,
    random_seed: int = 7,
    split: ChronologicalDataSplit | None = None,
    strategy_settings: StrategySettings | None = None,
) -> MarketMakingStudyResult:
    split = chronological_split(dataset) if split is None else split
    calibration = calibrate_market_data(split.train_events, split.train_snapshots)
    validation_rows: list[dict[str, float]] = []
    for queue in queue_grid:
        config = _replace_config(base_config, queue_ahead_fraction=queue)
        result = replay_market_maker(split.validation_events, split.validation_snapshots, QueueAwarePolicy(), config)
        risk_score = result.net_pnl / max(1.0 + result.max_abs_inventory, 1.0)
        validation_rows.append({"queue_ahead_fraction": queue, "net_pnl": result.net_pnl, "risk_score": risk_score})
    validation_scores = pd.DataFrame(validation_rows)
    selected_queue = float(
        validation_scores.sort_values(["risk_score", "queue_ahead_fraction"], ascending=[False, True]).iloc[0][
            "queue_ahead_fraction"
        ]
    )
    selected_config = _replace_config(base_config, queue_ahead_fraction=selected_queue)
    results = tuple(
        replay_market_maker(split.test_events, split.test_snapshots, policy, selected_config)
        for policy in _policy_set(selected_config.latency_ns, strategy_settings)
    )
    comparison = pd.DataFrame([result.summary() for result in results])
    baseline = next(result for result in results if result.strategy == "fixed_spread")
    intervals: dict[str, BootstrapInterval] = {}
    for result in results:
        if result.strategy == baseline.strategy:
            continue
        intervals[result.strategy] = _block_bootstrap_pnl_difference(
            result.history["pnl"].to_numpy(float),
            baseline.history["pnl"].to_numpy(float),
            bootstrap_replicates,
            random_seed,
            bootstrap_block_size,
        )
    sensitivity_rows: list[dict[str, float | str]] = []
    for latency in latency_grid_ns:
        for queue in queue_grid:
            for fee_multiplier in fee_multiplier_grid:
                config = _replace_config(
                    base_config,
                    latency_ns=latency,
                    queue_ahead_fraction=queue,
                    maker_fee_bps=base_config.maker_fee_bps * fee_multiplier,
                    taker_fee_bps=base_config.taker_fee_bps * fee_multiplier,
                    fixed_fee_per_order=base_config.fixed_fee_per_order * fee_multiplier,
                )
                for policy in _policy_set(latency, strategy_settings):
                    result = replay_market_maker(split.test_events, split.test_snapshots, policy, config)
                    sensitivity_rows.append(
                        {
                            "strategy": result.strategy,
                            "latency_ns": latency,
                            "queue_ahead_fraction": queue,
                            "fee_multiplier": fee_multiplier,
                            "net_pnl": result.net_pnl,
                            "max_abs_inventory": result.max_abs_inventory,
                            "fill_rate": result.fill_rate,
                        }
                    )
    return MarketMakingStudyResult(
        calibration,
        selected_queue,
        validation_scores,
        results,
        comparison,
        intervals,
        pd.DataFrame(sensitivity_rows),
        split,
    )


def _replace_config(config: ReplayConfig, **updates) -> ReplayConfig:
    values = asdict(config)
    values.update(updates)
    return ReplayConfig(**values)


def _policy_set(latency_ns: int, settings: StrategySettings | None) -> tuple[QuotingPolicy, ...]:
    if settings is None:
        return (
            FixedSpreadPolicy(),
            AvellanedaStoikovPolicy(),
            QueueAwarePolicy(),
            ToxicityAwarePolicy(),
            LatencyAwarePolicy(latency_ns=latency_ns),
        )
    return (
        FixedSpreadPolicy(half_spread_ticks=settings.fixed_half_spread_ticks),
        AvellanedaStoikovPolicy(
            risk_aversion=settings.avellaneda_risk_aversion,
            liquidity=settings.avellaneda_liquidity,
            horizon=settings.avellaneda_horizon,
        ),
        QueueAwarePolicy(inventory_skew_ticks=settings.inventory_skew_ticks),
        ToxicityAwarePolicy(
            toxicity_threshold=settings.toxicity_threshold,
            max_widen_ticks=settings.max_toxicity_widen_ticks,
        ),
        LatencyAwarePolicy(latency_ns=latency_ns, toxicity_threshold=settings.toxicity_threshold),
    )


def _block_bootstrap_pnl_difference(
    strategy_pnl: np.ndarray,
    baseline_pnl: np.ndarray,
    replicates: int,
    seed: int,
    block_size: int = 20,
) -> BootstrapInterval:
    strategy_changes = np.diff(strategy_pnl)
    baseline_changes = np.diff(baseline_pnl)
    length = min(len(strategy_changes), len(baseline_changes))
    difference = strategy_changes[:length] - baseline_changes[:length]
    estimate = float(difference.sum()) if length else 0.0
    if length == 0 or replicates < 2:
        return BootstrapInterval(estimate, estimate, estimate, 0.95)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates)
    for replicate in range(replicates):
        sampled: list[float] = []
        while len(sampled) < length:
            start = int(rng.integers(0, max(length - block_size + 1, 1)))
            sampled.extend(difference[start : start + block_size])
        draws[replicate] = np.sum(sampled[:length])
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return BootstrapInterval(estimate, float(lower), float(upper), 0.95)
