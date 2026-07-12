from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

BookLevel = Literal["L2", "L3"]


@dataclass(frozen=True)
class ResearchPeriod:
    start: str
    end: str

    def validate(self, name: str) -> tuple[datetime, datetime]:
        start = _utc_datetime(self.start, f"{name}.start")
        end = _utc_datetime(self.end, f"{name}.end")
        if start >= end:
            raise ValueError(f"{name}.start must precede {name}.end")
        return start, end


@dataclass(frozen=True)
class DatasetReference:
    provider: str
    dataset_id: str
    sha256: str
    schema_version: str
    book_level: BookLevel
    venue: str
    symbol: str
    tick_size: float
    timezone: str = "UTC"

    def validate(self) -> None:
        if not all((self.provider, self.dataset_id, self.schema_version, self.venue, self.symbol)):
            raise ValueError("dataset text fields must be non-empty")
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256.lower()):
            raise ValueError("dataset.sha256 must be a 64-character hexadecimal digest")
        if self.book_level not in {"L2", "L3"}:
            raise ValueError("dataset.book_level must be L2 or L3")
        if self.tick_size <= 0:
            raise ValueError("dataset.tick_size must be positive")
        if self.timezone != "UTC":
            raise ValueError("canonical datasets must use UTC")


@dataclass(frozen=True)
class CostModel:
    maker_fee_bps: float
    taker_fee_bps: float
    fixed_fee_per_order: float = 0.0

    def validate(self) -> None:
        if self.taker_fee_bps < self.maker_fee_bps:
            raise ValueError("taker_fee_bps must be at least maker_fee_bps")
        if self.fixed_fee_per_order < 0:
            raise ValueError("fixed_fee_per_order must be non-negative")


@dataclass(frozen=True)
class ExecutionSettings:
    order_size: float
    inventory_limit: float
    quote_interval_ns: int
    toxicity_window: int
    volatility_window: int
    adverse_horizon_ns: int
    record_every: int

    def validate(self) -> None:
        if self.order_size <= 0 or self.inventory_limit <= 0:
            raise ValueError("execution order_size and inventory_limit must be positive")
        if (
            min(
                self.quote_interval_ns,
                self.toxicity_window,
                self.volatility_window,
                self.adverse_horizon_ns,
                self.record_every,
            )
            < 1
        ):
            raise ValueError("execution intervals and windows must be positive")


@dataclass(frozen=True)
class StrategySettings:
    fixed_half_spread_ticks: float
    avellaneda_risk_aversion: float
    avellaneda_liquidity: float
    avellaneda_horizon: float
    inventory_skew_ticks: float
    toxicity_threshold: float
    max_toxicity_widen_ticks: float

    def validate(self) -> None:
        if (
            min(
                self.fixed_half_spread_ticks,
                self.avellaneda_risk_aversion,
                self.avellaneda_liquidity,
                self.avellaneda_horizon,
            )
            <= 0
        ):
            raise ValueError("strategy scale parameters must be positive")
        if min(self.inventory_skew_ticks, self.toxicity_threshold, self.max_toxicity_widen_ticks) < 0:
            raise ValueError("strategy skew and toxicity parameters must be non-negative")


@dataclass(frozen=True)
class EvaluationSettings:
    embargo_ns: int
    bootstrap_replicates: int
    bootstrap_block_size: int

    def validate(self) -> None:
        if self.embargo_ns < 0 or self.bootstrap_replicates < 2 or self.bootstrap_block_size < 1:
            raise ValueError("evaluation settings are invalid")


@dataclass(frozen=True)
class MarketMakingExperimentConfig:
    schema_version: str
    experiment_id: str
    random_seed: int
    dataset: DatasetReference
    train: ResearchPeriod
    validation: ResearchPeriod
    test: ResearchPeriod
    strategies: tuple[str, ...]
    latency_ns: tuple[int, ...]
    queue_ahead_fraction: tuple[float, ...]
    fee_multiplier: tuple[float, ...]
    cost_model: CostModel
    execution: ExecutionSettings
    strategy_parameters: StrategySettings
    evaluation: EvaluationSettings

    def validate(self) -> None:
        if self.schema_version != "1.0.0":
            raise ValueError("unsupported experiment schema_version")
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must be non-empty")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
        self.dataset.validate()
        train = self.train.validate("train")
        validation = self.validation.validate("validation")
        test = self.test.validate("test")
        if not (train[1] <= validation[0] and validation[1] <= test[0]):
            raise ValueError("train, validation, and test periods must be chronological and non-overlapping")
        required_strategies = {
            "fixed_spread",
            "avellaneda_stoikov",
            "queue_aware",
            "toxicity_aware",
            "latency_aware",
        }
        if set(self.strategies) != required_strategies or len(self.strategies) != len(required_strategies):
            raise ValueError(f"strategies must contain exactly {sorted(required_strategies)}")
        if not self.latency_ns or any(value < 0 for value in self.latency_ns):
            raise ValueError("latency_ns must contain non-negative scenarios")
        if tuple(sorted(set(self.latency_ns))) != self.latency_ns:
            raise ValueError("latency_ns must be unique and sorted")
        if not self.queue_ahead_fraction or any(not 0.0 <= value <= 1.0 for value in self.queue_ahead_fraction):
            raise ValueError("queue_ahead_fraction values must lie in [0, 1]")
        if tuple(sorted(set(self.queue_ahead_fraction))) != self.queue_ahead_fraction:
            raise ValueError("queue_ahead_fraction must be unique and sorted")
        if not self.fee_multiplier or any(value <= 0 for value in self.fee_multiplier):
            raise ValueError("fee_multiplier must contain positive scenarios")
        self.cost_model.validate()
        self.execution.validate()
        self.strategy_parameters.validate()
        self.evaluation.validate()

    @property
    def fingerprint(self) -> str:
        """SHA-256 of the canonical config, suitable for result attribution."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_market_making_experiment_config(path: str | Path) -> MarketMakingExperimentConfig:
    """Load a strict configuration; unknown fields are rejected by constructors."""
    with Path(path).open(encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "experiment_id",
            "random_seed",
            "dataset",
            "periods",
            "strategies",
            "sensitivity",
            "cost_model",
            "execution",
            "strategy_parameters",
            "evaluation",
        },
        "config",
    )
    _require_exact_keys(
        payload["dataset"],
        {
            "provider",
            "dataset_id",
            "sha256",
            "schema_version",
            "book_level",
            "venue",
            "symbol",
            "tick_size",
            "timezone",
        },
        "dataset",
    )
    _require_exact_keys(payload["periods"], {"train", "validation", "test"}, "periods")
    for period_name in ("train", "validation", "test"):
        _require_exact_keys(payload["periods"][period_name], {"start", "end"}, f"periods.{period_name}")
    _require_exact_keys(payload["sensitivity"], {"latency_ns", "queue_ahead_fraction", "fee_multiplier"}, "sensitivity")
    _require_exact_keys(payload["cost_model"], {"maker_fee_bps", "taker_fee_bps", "fixed_fee_per_order"}, "cost_model")
    _require_exact_keys(
        payload["execution"],
        {
            "order_size",
            "inventory_limit",
            "quote_interval_ns",
            "toxicity_window",
            "volatility_window",
            "adverse_horizon_ns",
            "record_every",
        },
        "execution",
    )
    _require_exact_keys(
        payload["strategy_parameters"],
        {
            "fixed_half_spread_ticks",
            "avellaneda_risk_aversion",
            "avellaneda_liquidity",
            "avellaneda_horizon",
            "inventory_skew_ticks",
            "toxicity_threshold",
            "max_toxicity_widen_ticks",
        },
        "strategy_parameters",
    )
    _require_exact_keys(
        payload["evaluation"], {"embargo_ns", "bootstrap_replicates", "bootstrap_block_size"}, "evaluation"
    )
    config = MarketMakingExperimentConfig(
        schema_version=payload["schema_version"],
        experiment_id=payload["experiment_id"],
        random_seed=int(payload["random_seed"]),
        dataset=DatasetReference(**payload["dataset"]),
        train=ResearchPeriod(**payload["periods"]["train"]),
        validation=ResearchPeriod(**payload["periods"]["validation"]),
        test=ResearchPeriod(**payload["periods"]["test"]),
        strategies=tuple(payload["strategies"]),
        latency_ns=tuple(int(value) for value in payload["sensitivity"]["latency_ns"]),
        queue_ahead_fraction=tuple(float(value) for value in payload["sensitivity"]["queue_ahead_fraction"]),
        fee_multiplier=tuple(float(value) for value in payload["sensitivity"]["fee_multiplier"]),
        cost_model=CostModel(**payload["cost_model"]),
        execution=ExecutionSettings(**payload["execution"]),
        strategy_parameters=StrategySettings(**payload["strategy_parameters"]),
        evaluation=EvaluationSettings(**payload["evaluation"]),
    )
    config.validate()
    return config


def _require_exact_keys(payload: Any, expected: set[str], name: str) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be an object")
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"{name} fields differ from schema; missing={missing}, unknown={unknown}")


def _utc_datetime(value: str, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{name} must be UTC")
    return parsed
