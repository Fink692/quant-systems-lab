from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BitcoinTrendConfig:
    train_start: str = "2016-01-01"
    train_end: str = "2017-12-31"
    validation_start: str = "2018-01-01"
    validation_end: str = "2020-12-31"
    evaluation_start: str = "2021-01-01"
    moving_average_days: tuple[int, ...] = (100, 150, 200, 250)
    volatility_days: tuple[int, ...] = (21, 63)
    target_volatilities: tuple[float, ...] = (0.3, 0.5, 0.75, 1.0)
    hysteresis_bands: tuple[float, ...] = (0.0, 0.02)
    max_exposure: float = 1.0
    transaction_cost_bps: float = 25.0
    target_cagr: float = 0.2
    annualization_days: int = 365

    def validate(self) -> None:
        dates = list(map(pd.Timestamp, (self.train_start, self.train_end, self.validation_start, self.validation_end)))
        if not (dates[0] <= dates[1] < dates[2] <= dates[3] < pd.Timestamp(self.evaluation_start)):
            raise ValueError("train, validation, and evaluation periods must be chronological and disjoint")
        if not self.moving_average_days or not self.volatility_days or not self.target_volatilities:
            raise ValueError("parameter grids must not be empty")
        if min(self.moving_average_days) <= 1 or min(self.volatility_days) <= 1:
            raise ValueError("lookbacks must exceed one day")
        if min(self.target_volatilities) <= 0 or self.max_exposure <= 0:
            raise ValueError("volatility targets and maximum exposure must be positive")
        if min(self.hysteresis_bands) < 0 or self.transaction_cost_bps < 0:
            raise ValueError("hysteresis and costs must be non-negative")
        if self.annualization_days <= 1 or not 0 < self.target_cagr < 1:
            raise ValueError("annualization days and target CAGR are invalid")


@dataclass(frozen=True)
class BitcoinTrendResult:
    selected: pd.Series
    history: pd.DataFrame
    grid_metrics: pd.DataFrame
    period_metrics: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    parameter_breadth: pd.Series
    calendar_returns: pd.Series
    rolling_summary: pd.DataFrame
    bootstrap: pd.Series
    config: BitcoinTrendConfig


def load_bitcoin_coinbase_csv(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    required = {"open", "high", "low", "close", "volume", "dff"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if data.index.has_duplicates or not data.index.is_monotonic_increasing:
        raise ValueError("dates must be unique and increasing")
    if data[list(required)].isna().any().any():
        raise ValueError("Bitcoin inputs must not contain missing values")
    if (data[["open", "high", "low", "close", "volume"]] <= 0).any().any():
        raise ValueError("Bitcoin OHLCV inputs must be positive")
    return data


def run_frozen_bitcoin_candidate(
    inputs: pd.DataFrame,
    config: BitcoinTrendConfig | None = None,
) -> pd.DataFrame:
    cfg = BitcoinTrendConfig() if config is None else config
    cfg.validate()
    return _run_candidate(inputs, cfg, 100, 63, 0.3, 0.02)


def run_bitcoin_trend_study(
    inputs: pd.DataFrame,
    config: BitcoinTrendConfig | None = None,
    bootstrap_samples: int = 2_000,
    block_size: int = 30,
    seed: int = 20_260_713,
) -> BitcoinTrendResult:
    cfg = BitcoinTrendConfig() if config is None else config
    cfg.validate()
    if bootstrap_samples <= 0 or block_size <= 1:
        raise ValueError("bootstrap samples and block size must be positive")
    combinations = list(
        product(cfg.moving_average_days, cfg.volatility_days, cfg.target_volatilities, cfg.hysteresis_bands)
    )
    histories: dict[tuple[int, int, float, float], pd.DataFrame] = {}
    rows: list[dict[str, float]] = []
    for parameters in combinations:
        history = _run_candidate(inputs, cfg, *parameters)
        histories[parameters] = history
        row: dict[str, float] = {
            "moving_average_days": float(parameters[0]),
            "volatility_days": float(parameters[1]),
            "target_volatility": parameters[2],
            "hysteresis_band": parameters[3],
        }
        for name, start, end in _periods(cfg):
            window = _window(history, start, end)
            row.update({f"{name}_{key}": value for key, value in _metrics(window, cfg.annualization_days).items()})
        rows.append(row)
    grid = pd.DataFrame(rows).sort_values(["validation_sharpe", "validation_cagr"], ascending=False, ignore_index=True)
    selected = grid.iloc[0]
    key = (
        int(selected["moving_average_days"]),
        int(selected["volatility_days"]),
        float(selected["target_volatility"]),
        float(selected["hysteresis_band"]),
    )
    history = histories[key]
    period_rows = []
    for name, start, end in _periods(cfg):
        period_rows.append({"period": name, **_metrics(_window(history, start, end), cfg.annualization_days)})

    costs = []
    for cost_bps in (10.0, 25.0, 50.0, 100.0, 200.0):
        scenario_cfg = BitcoinTrendConfig(**{**cfg.__dict__, "transaction_cost_bps": cost_bps})
        scenario = _run_candidate(inputs, scenario_cfg, *key)
        costs.append(
            {
                "cost_bps": cost_bps,
                "evaluation_cagr": _metrics(_window(scenario, cfg.evaluation_start, None), cfg.annualization_days)[
                    "cagr"
                ],
                "full_cagr": _metrics(_window(scenario, cfg.train_start, None), cfg.annualization_days)["cagr"],
            }
        )

    evaluation = _window(history, cfg.evaluation_start, None)["strategy_return"].dropna()
    calendar = evaluation.groupby(evaluation.index.year).apply(lambda values: float((1.0 + values).prod() - 1.0))
    calendar.index.name = "year"
    rolling_rows = []
    full_returns = _window(history, cfg.train_start, None)["strategy_return"].dropna()
    for years in (3, 5):
        window_days = years * cfg.annualization_days
        rolling = (1.0 + full_returns).rolling(window_days).apply(np.prod, raw=True) ** (1.0 / years) - 1.0
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

    bootstrap_values = _moving_block_bootstrap_cagr(
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
    clears_evaluation = grid["evaluation_cagr"] >= cfg.target_cagr
    clears_full = grid["full_cagr"] >= cfg.target_cagr
    breadth = pd.Series(
        {
            "configurations": float(len(grid)),
            "evaluation_at_or_above_target": float(clears_evaluation.sum()),
            "full_at_or_above_target": float(clears_full.sum()),
            "both_at_or_above_target": float((clears_evaluation & clears_full).sum()),
        }
    )
    return BitcoinTrendResult(
        selected,
        history,
        grid,
        pd.DataFrame(period_rows).set_index("period"),
        pd.DataFrame(costs),
        breadth,
        calendar,
        pd.DataFrame(rolling_rows).set_index("years"),
        bootstrap,
        cfg,
    )


def _run_candidate(
    data: pd.DataFrame,
    cfg: BitcoinTrendConfig,
    moving_average_days: int,
    volatility_days: int,
    target_volatility: float,
    hysteresis_band: float,
) -> pd.DataFrame:
    close = data["close"]
    moving_average = close.rolling(moving_average_days).mean()
    state = np.zeros(len(data), dtype=float)
    active = False
    for index in range(len(data)):
        price = close.iloc[index]
        average = moving_average.iloc[index]
        if np.isfinite(average):
            if active and price < average * (1.0 - hysteresis_band):
                active = False
            elif not active and price > average * (1.0 + hysteresis_band):
                active = True
        state[index] = float(active)
    realized_volatility = close.pct_change().rolling(volatility_days).std(ddof=1) * np.sqrt(cfg.annualization_days)
    desired_exposure = pd.Series(state, index=data.index) * (
        target_volatility / realized_volatility.clip(lower=0.05)
    ).clip(upper=cfg.max_exposure)
    new_exposure = desired_exposure.shift(1).fillna(0.0)
    old_exposure = desired_exposure.shift(2).fillna(0.0)
    overnight_asset = data["open"] / close.shift(1) - 1.0
    intraday_asset = close / data["open"] - 1.0
    cash_return = (data["dff"].shift(1).bfill() / 100.0 / 360.0).clip(lower=0.0)
    overnight = old_exposure * overnight_asset + (1.0 - old_exposure) * cash_return * (2.0 / 3.0)
    intraday = new_exposure * intraday_asset + (1.0 - new_exposure) * cash_return * (1.0 / 3.0)
    turnover = (new_exposure - old_exposure).abs()
    trading_cost = turnover * cfg.transaction_cost_bps / 10_000.0
    strategy_return = (1.0 + overnight.fillna(0.0)) * (1.0 + intraday.fillna(0.0)) - 1.0 - trading_cost
    return pd.DataFrame(
        {
            "moving_average": moving_average,
            "signal_state": state,
            "realized_volatility": realized_volatility,
            "desired_exposure": desired_exposure,
            "old_exposure": old_exposure,
            "new_exposure": new_exposure,
            "turnover": turnover,
            "trading_cost": trading_cost,
            "overnight_return": overnight,
            "intraday_return": intraday,
            "strategy_return": strategy_return,
        },
        index=data.index,
    )


def _periods(cfg: BitcoinTrendConfig) -> list[tuple[str, str, str | None]]:
    return [
        ("train", cfg.train_start, cfg.train_end),
        ("validation", cfg.validation_start, cfg.validation_end),
        ("evaluation", cfg.evaluation_start, None),
        ("full", cfg.train_start, None),
    ]


def _window(history: pd.DataFrame, start: str, end: str | None) -> pd.DataFrame:
    window = history.loc[history.index >= pd.Timestamp(start)]
    return window if end is None else window.loc[window.index <= pd.Timestamp(end)]


def _metrics(history: pd.DataFrame, annualization_days: int) -> dict[str, float]:
    returns = history["strategy_return"].dropna()
    equity = (1.0 + returns).cumprod()
    if equity.empty:
        raise ValueError("metric window contains no observations")
    cagr = float(equity.iloc[-1] ** (annualization_days / len(equity)) - 1.0)
    peaks = np.maximum.accumulate(np.concatenate(([1.0], equity.to_numpy())))[1:]
    volatility = float(returns.std(ddof=1) * np.sqrt(annualization_days))
    return {
        "cagr": cagr,
        "sharpe": 0.0 if volatility == 0 else float(returns.mean() * annualization_days / volatility),
        "max_drawdown": float(np.max(1.0 - equity.to_numpy() / peaks)),
        "annualized_volatility": volatility,
        "observations": float(len(returns)),
    }


def _moving_block_bootstrap_cagr(
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
        sample = np.concatenate([returns[start : start + block_size] for start in sampled_starts])[: len(returns)]
        result[sample_index] = float(np.prod(1.0 + sample) ** (annualization_days / len(sample)) - 1.0)
    return result
