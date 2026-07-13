from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FixedEnsembleConfig:
    strategy_id: str = "btc-defensive-equal-weight-monthly-v1"
    bitcoin_weight: float = 0.5
    defensive_weight: float = 0.5
    rebalance_frequency: str = "monthly"
    rebalance_cost_bps: float = 10.0
    full_start: str = "2016-01-01"
    evaluation_start: str = "2021-01-01"
    target_cagr: float = 0.2
    annualization_days: int = 365

    def validate(self) -> None:
        if not self.strategy_id:
            raise ValueError("strategy_id is required")
        if not np.isclose(self.bitcoin_weight + self.defensive_weight, 1.0):
            raise ValueError("ensemble weights must sum to one")
        if min(self.bitcoin_weight, self.defensive_weight) < 0:
            raise ValueError("ensemble weights must be non-negative")
        if self.rebalance_frequency != "monthly":
            raise ValueError("frozen ensemble must rebalance monthly")
        if self.rebalance_cost_bps < 0:
            raise ValueError("rebalance cost must be non-negative")
        if pd.Timestamp(self.evaluation_start) <= pd.Timestamp(self.full_start):
            raise ValueError("evaluation must start after full history")
        if self.annualization_days <= 1 or not 0 < self.target_cagr < 1:
            raise ValueError("annualization days and target CAGR are invalid")


@dataclass(frozen=True)
class FixedEnsembleResult:
    history: pd.DataFrame
    period_metrics: pd.DataFrame
    component_correlation: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    allocation_diagnostics: pd.DataFrame
    calendar_returns: pd.Series
    rolling_summary: pd.DataFrame
    bootstrap: pd.Series
    config: FixedEnsembleConfig


def run_fixed_strategy_ensemble(
    bitcoin_returns: pd.Series,
    defensive_returns: pd.Series,
    config: FixedEnsembleConfig | None = None,
    bootstrap_samples: int = 5_000,
    block_size: int = 30,
    seed: int = 20_260_713,
) -> FixedEnsembleResult:
    cfg = FixedEnsembleConfig() if config is None else config
    cfg.validate()
    if bootstrap_samples <= 0 or block_size <= 1:
        raise ValueError("bootstrap samples and block size must be positive")
    components = _align_components(bitcoin_returns, defensive_returns, cfg.full_start)
    history = _run_ensemble(components, cfg)
    period_rows = []
    for name, start in (("evaluation", cfg.evaluation_start), ("full", cfg.full_start)):
        period_rows.append({"period": name, **_metrics(history.loc[start:, "strategy_return"], cfg.annualization_days)})

    cost_rows = []
    for cost_bps in (0.0, 10.0, 25.0, 50.0, 100.0, 200.0):
        scenario_cfg = FixedEnsembleConfig(**{**cfg.__dict__, "rebalance_cost_bps": cost_bps})
        scenario = _run_ensemble(components, scenario_cfg)
        cost_rows.append(
            {
                "cost_bps": cost_bps,
                "evaluation_cagr": _metrics(
                    scenario.loc[cfg.evaluation_start :, "strategy_return"], cfg.annualization_days
                )["cagr"],
                "full_cagr": _metrics(scenario.loc[cfg.full_start :, "strategy_return"], cfg.annualization_days)[
                    "cagr"
                ],
            }
        )

    allocation_rows = []
    for bitcoin_weight in (0.25, 0.5, 0.75):
        scenario_cfg = FixedEnsembleConfig(
            **{
                **cfg.__dict__,
                "bitcoin_weight": bitcoin_weight,
                "defensive_weight": 1.0 - bitcoin_weight,
            }
        )
        scenario = _run_ensemble(components, scenario_cfg)
        allocation_rows.append(
            {
                "bitcoin_weight": bitcoin_weight,
                "defensive_weight": 1.0 - bitcoin_weight,
                **_metrics(scenario.loc[cfg.evaluation_start :, "strategy_return"], cfg.annualization_days),
            }
        )

    evaluation = history.loc[cfg.evaluation_start :, "strategy_return"]
    calendar = evaluation.groupby(evaluation.index.year).apply(lambda values: float((1.0 + values).prod() - 1.0))
    calendar.index.name = "year"
    full_returns = history.loc[cfg.full_start :, "strategy_return"]
    rolling_rows = []
    for years in (3, 5):
        window = years * cfg.annualization_days
        rolling = (1.0 + full_returns).rolling(window).apply(np.prod, raw=True) ** (1.0 / years) - 1.0
        rolling = rolling.dropna()
        rolling_rows.append(
            {
                "years": years,
                "windows": len(rolling),
                "minimum_cagr": float(rolling.min()),
                "median_cagr": float(rolling.median()),
                "fraction_at_or_above_target": float((rolling >= cfg.target_cagr).mean()),
            }
        )

    bootstrap_values = _moving_block_bootstrap(
        evaluation.to_numpy(), bootstrap_samples, block_size, cfg.annualization_days, seed
    )
    bootstrap = pd.Series(
        {
            "samples": float(bootstrap_samples),
            "block_size": float(block_size),
            "cagr_5th_percentile": float(np.quantile(bootstrap_values, 0.05)),
            "cagr_median": float(np.quantile(bootstrap_values, 0.5)),
            "cagr_95th_percentile": float(np.quantile(bootstrap_values, 0.95)),
            "probability_cagr_at_or_above_target": float((bootstrap_values >= cfg.target_cagr).mean()),
        }
    )
    correlation = components.loc[cfg.evaluation_start :, ["bitcoin_return", "defensive_return"]].corr()
    return FixedEnsembleResult(
        history,
        pd.DataFrame(period_rows).set_index("period"),
        correlation,
        pd.DataFrame(cost_rows),
        pd.DataFrame(allocation_rows),
        calendar,
        pd.DataFrame(rolling_rows).set_index("years"),
        bootstrap,
        cfg,
    )


def _align_components(
    bitcoin_returns: pd.Series,
    defensive_returns: pd.Series,
    start: str,
) -> pd.DataFrame:
    bitcoin = bitcoin_returns.dropna().sort_index().rename("bitcoin_return")
    defensive = defensive_returns.dropna().sort_index().rename("defensive_return")
    end = min(bitcoin.index.max(), defensive.index.max())
    index = pd.date_range(max(pd.Timestamp(start), bitcoin.index.min(), defensive.index.min()), end, freq="D")
    components = pd.concat([bitcoin.reindex(index), defensive.reindex(index)], axis=1)
    components["defensive_available"] = components["defensive_return"].notna()
    components[["bitcoin_return", "defensive_return"]] = components[["bitcoin_return", "defensive_return"]].fillna(0.0)
    if components.empty or (components <= -1.0).any().any():
        raise ValueError("component returns must contain valid overlapping history")
    return components


def _run_ensemble(components: pd.DataFrame, cfg: FixedEnsembleConfig) -> pd.DataFrame:
    target = np.array([cfg.bitcoin_weight, cfg.defensive_weight])
    sleeve_values = target.copy()
    rows: list[dict[str, float]] = []
    rebalanced_month = components.index[0].to_period("M")
    for date, component_return in components.iterrows():
        sleeve_return = component_return[["bitcoin_return", "defensive_return"]].astype(float)
        start_value = float(sleeve_values.sum())
        weights = sleeve_values / start_value
        turnover = 0.0
        current_month = date.to_period("M")
        if current_month != rebalanced_month and bool(component_return["defensive_available"]):
            turnover = float(np.abs(target - weights).sum())
            start_value *= 1.0 - turnover * cfg.rebalance_cost_bps / 10_000.0
            sleeve_values = start_value * target
            weights = target.copy()
            rebalanced_month = current_month
        sleeve_values *= 1.0 + sleeve_return.to_numpy()
        strategy_return = float(
            sleeve_values.sum()
            / (start_value / (1.0 - turnover * cfg.rebalance_cost_bps / 10_000.0) if turnover else start_value)
            - 1.0
        )
        rows.append(
            {
                "bitcoin_return": float(sleeve_return.iloc[0]),
                "defensive_return": float(sleeve_return.iloc[1]),
                "bitcoin_weight": float(weights[0]),
                "defensive_weight": float(weights[1]),
                "turnover": turnover,
                "rebalance_cost": turnover * cfg.rebalance_cost_bps / 10_000.0,
                "strategy_return": strategy_return,
            }
        )
    return pd.DataFrame(rows, index=components.index)


def _metrics(returns: pd.Series, annualization_days: int) -> dict[str, float]:
    equity = (1.0 + returns).cumprod()
    if equity.empty:
        raise ValueError("metric window contains no observations")
    volatility = float(returns.std(ddof=1) * np.sqrt(annualization_days))
    peaks = np.maximum.accumulate(np.concatenate(([1.0], equity.to_numpy())))[1:]
    return {
        "cagr": float(equity.iloc[-1] ** (annualization_days / len(equity)) - 1.0),
        "sharpe": 0.0 if volatility == 0 else float(returns.mean() * annualization_days / volatility),
        "max_drawdown": float(np.max(1.0 - equity.to_numpy() / peaks)),
        "annualized_volatility": volatility,
        "observations": float(len(returns)),
    }


def _moving_block_bootstrap(
    returns: np.ndarray,
    samples: int,
    block_size: int,
    annualization_days: int,
    seed: int,
) -> np.ndarray:
    starts = np.arange(len(returns) - block_size + 1)
    if len(starts) == 0:
        raise ValueError("block size exceeds evaluation history")
    rng = np.random.default_rng(seed)
    result = np.empty(samples)
    blocks_needed = int(np.ceil(len(returns) / block_size))
    for sample_index in range(samples):
        sampled_starts = rng.choice(starts, size=blocks_needed, replace=True)
        sampled = np.concatenate([returns[start : start + block_size] for start in sampled_starts])[: len(returns)]
        result[sample_index] = float(np.prod(1.0 + sampled) ** (annualization_days / len(sampled)) - 1.0)
    return result
