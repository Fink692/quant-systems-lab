from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.risk.var import historical_cvar, historical_var


@dataclass(frozen=True)
class LeveragedTrendConfig:
    train_end: str = "2016-12-31"
    validation_start: str = "2017-01-01"
    validation_end: str = "2020-12-31"
    holdout_start: str = "2021-01-01"
    moving_average_candidates: tuple[int, ...] = (100, 150, 200, 250)
    band_candidates: tuple[float, ...] = (0.0, 0.01, 0.02)
    target_volatility_candidates: tuple[float, ...] = (0.30, 0.40, 0.50, 1.00)
    volatility_lookback: int = 21
    max_exposure: float = 1.0
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 5.0
    bootstrap_samples: int = 2_000
    bootstrap_block_days: int = 21
    bootstrap_seed: int = 7
    target_cagr: float = 0.20

    def validate(self) -> None:
        if not self.moving_average_candidates or min(self.moving_average_candidates) <= 1:
            raise ValueError("moving-average candidates must be greater than one day")
        if not self.target_volatility_candidates or min(self.target_volatility_candidates) <= 0:
            raise ValueError("target-volatility candidates must be positive")
        if any(value < 0 for value in self.band_candidates):
            raise ValueError("signal bands must be non-negative")
        if min(self.volatility_lookback, self.bootstrap_samples, self.bootstrap_block_days) <= 0:
            raise ValueError("lookback and bootstrap settings must be positive")
        if not 0 < self.max_exposure <= 1:
            raise ValueError("max_exposure must be in (0, 1]")
        if min(self.transaction_cost_bps, self.slippage_bps) < 0:
            raise ValueError("cost assumptions must be non-negative")
        if not 0 < self.target_cagr < 1:
            raise ValueError("target_cagr must be in (0, 1)")
        train_end = pd.Timestamp(self.train_end)
        validation_start = pd.Timestamp(self.validation_start)
        validation_end = pd.Timestamp(self.validation_end)
        holdout_start = pd.Timestamp(self.holdout_start)
        if not train_end < validation_start <= validation_end < holdout_start:
            raise ValueError("train, validation, and holdout dates must be strictly ordered")


@dataclass(frozen=True)
class LeveragedTrendResult:
    history: pd.DataFrame
    metrics: pd.Series
    selected_parameters: pd.Series
    selection_grid: pd.DataFrame
    parameter_sensitivity: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    annual_returns: pd.DataFrame
    bootstrap: pd.Series
    config: LeveragedTrendConfig

    @property
    def holdout_target_met(self) -> bool:
        return bool(self.metrics["cagr"] >= self.config.target_cagr)


def load_leveraged_etf_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "tqqq", "qqq", "bil"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    frame = frame.sort_values("date").drop_duplicates("date").set_index("date")
    frame = frame[["tqqq", "qqq", "bil"]].apply(pd.to_numeric, errors="coerce")
    if frame.isna().any().any():
        raise ValueError("adjusted prices must be complete")
    if (frame <= 0).any().any():
        raise ValueError("adjusted prices must be positive")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("dates must be increasing")
    return frame


def run_leveraged_trend_study(
    prices: pd.DataFrame,
    config: LeveragedTrendConfig | None = None,
) -> LeveragedTrendResult:
    cfg = LeveragedTrendConfig() if config is None else config
    cfg.validate()
    data = _validate_prices(prices)
    returns = data.pct_change().fillna(0.0)
    selection_rows: list[dict[str, float]] = []
    histories: dict[tuple[int, float, float], pd.DataFrame] = {}
    for moving_average, band, target_volatility in product(
        cfg.moving_average_candidates,
        cfg.band_candidates,
        cfg.target_volatility_candidates,
    ):
        key = (moving_average, band, target_volatility)
        history = _run_strategy(data, returns, cfg, moving_average, band, target_volatility)
        histories[key] = history
        validation = _date_slice(history, cfg.validation_start, cfg.validation_end)
        validation_metrics = _performance_metrics(validation, periods_per_year=252.0)
        selection_rows.append(
            {
                "moving_average_days": moving_average,
                "band": band,
                "target_volatility": target_volatility,
                "validation_cagr": validation_metrics["cagr"],
                "validation_sharpe": validation_metrics["sharpe"],
                "validation_max_drawdown": validation_metrics["max_drawdown"],
            }
        )
    selection_grid = (
        pd.DataFrame(selection_rows)
        .sort_values(["validation_sharpe", "validation_cagr"], ascending=False)
        .reset_index(drop=True)
    )
    selected = selection_grid.iloc[0]
    selected_key = (
        int(selected["moving_average_days"]),
        float(selected["band"]),
        float(selected["target_volatility"]),
    )
    full_history = histories[selected_key]
    holdout = _date_slice(full_history, cfg.holdout_start, None).copy()
    if holdout.empty:
        raise ValueError("holdout period has no observations")
    metrics = _performance_metrics(holdout, periods_per_year=252.0)
    train = _date_slice(full_history, None, cfg.train_end)
    validation = _date_slice(full_history, cfg.validation_start, cfg.validation_end)
    metrics["train_cagr"] = _performance_metrics(train, 252.0)["cagr"]
    metrics["validation_cagr"] = _performance_metrics(validation, 252.0)["cagr"]
    metrics["holdout_observations"] = float(len(holdout))
    metrics["holdout_years"] = float(len(holdout) / 252.0)

    parameter_sensitivity = _parameter_sensitivity(histories, cfg)
    cost_sensitivity = _cost_sensitivity(data, returns, cfg, selected_key)
    annual_returns = _annual_returns(holdout)
    bootstrap = _block_bootstrap(holdout["strategy_return"], cfg)
    return LeveragedTrendResult(
        holdout,
        metrics,
        selected[["moving_average_days", "band", "target_volatility", "validation_cagr", "validation_sharpe"]],
        selection_grid,
        parameter_sensitivity,
        cost_sensitivity,
        annual_returns,
        bootstrap,
        cfg,
    )


def _validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    required = ["tqqq", "qqq", "bil"]
    missing = set(required) - set(prices.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    data = prices[required].copy().sort_index()
    if data.index.has_duplicates or data.isna().any().any() or (data <= 0).any().any():
        raise ValueError("prices require unique dates and complete positive values")
    return data


def _run_strategy(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    cfg: LeveragedTrendConfig,
    moving_average: int,
    band: float,
    target_volatility: float,
    total_cost_bps: float | None = None,
) -> pd.DataFrame:
    average = prices["tqqq"].rolling(moving_average).mean()
    state = pd.Series(np.nan, index=prices.index, dtype=float)
    state.loc[prices["tqqq"] > average * (1.0 + band)] = 1.0
    state.loc[prices["tqqq"] < average * (1.0 - band)] = 0.0
    state = state.ffill().fillna(0.0)
    realized_volatility = returns["tqqq"].rolling(cfg.volatility_lookback).std(ddof=1) * np.sqrt(252.0)
    volatility_scalar = (target_volatility / realized_volatility.clip(lower=1e-8)).clip(upper=cfg.max_exposure)
    desired_exposure = (state * volatility_scalar).fillna(0.0)
    exposure = desired_exposure.shift(1).fillna(0.0)
    turnover = exposure.diff().abs().fillna(exposure.abs())
    cost_bps = cfg.transaction_cost_bps + cfg.slippage_bps if total_cost_bps is None else total_cost_bps
    trading_cost = turnover * cost_bps / 10_000.0
    strategy_return = exposure * returns["tqqq"] + (1.0 - exposure) * returns["bil"] - trading_cost
    history = pd.DataFrame(
        {
            "tqqq_price": prices["tqqq"],
            "moving_average": average,
            "signal": state,
            "realized_volatility": realized_volatility,
            "exposure": exposure,
            "turnover": turnover,
            "trading_cost": trading_cost,
            "strategy_return": strategy_return,
            "tqqq_return": returns["tqqq"],
            "qqq_return": returns["qqq"],
            "cash_return": returns["bil"],
        },
        index=prices.index,
    )
    history["strategy_equity"] = (1.0 + history["strategy_return"]).cumprod()
    history["tqqq_equity"] = (1.0 + history["tqqq_return"]).cumprod()
    history["qqq_equity"] = (1.0 + history["qqq_return"]).cumprod()
    return history


def _performance_metrics(history: pd.DataFrame, periods_per_year: float) -> pd.Series:
    if history.empty:
        raise ValueError("cannot calculate metrics for an empty period")
    returns = history["strategy_return"]
    equity = (1.0 + returns).cumprod()
    tqqq_equity = (1.0 + history["tqqq_return"]).cumprod()
    qqq_equity = (1.0 + history["qqq_return"]).cumprod()
    drawdown = _drawdown(equity)
    annual_volatility = float(returns.std(ddof=1) * np.sqrt(periods_per_year))
    downside = returns[returns < 0.0]
    benchmark = history["qqq_return"]
    beta = _beta(returns.to_numpy(), benchmark.to_numpy())
    cagr = _cagr(equity, periods_per_year)
    return pd.Series(
        {
            "cagr": cagr,
            "tqqq_buy_hold_cagr": _cagr(tqqq_equity, periods_per_year),
            "qqq_buy_hold_cagr": _cagr(qqq_equity, periods_per_year),
            "volatility": annual_volatility,
            "sharpe": _safe_divide(float(returns.mean() * periods_per_year), annual_volatility),
            "sortino": _safe_divide(
                float(returns.mean() * periods_per_year), float(downside.std(ddof=1) * np.sqrt(periods_per_year))
            ),
            "max_drawdown": float(drawdown.max()),
            "calmar": _safe_divide(cagr, float(drawdown.max())),
            "hit_rate": float((returns > 0.0).mean()),
            "profit_factor": _profit_factor(returns.to_numpy()),
            "average_daily_turnover": float(history["turnover"].mean()),
            "annual_turnover": float(history["turnover"].mean() * periods_per_year),
            "average_exposure": float(history["exposure"].mean()),
            "total_cost": float(history["trading_cost"].sum()),
            "beta_to_qqq": beta,
            "alpha_to_qqq_annual": float(periods_per_year * (returns.mean() - beta * benchmark.mean())),
            "value_at_risk_95": historical_var(returns.to_numpy(), confidence=0.95),
            "conditional_var_95": historical_cvar(returns.to_numpy(), confidence=0.95),
            "max_drawdown_duration_days": float(_max_drawdown_duration(drawdown)),
        }
    )


def _parameter_sensitivity(
    histories: dict[tuple[int, float, float], pd.DataFrame], cfg: LeveragedTrendConfig
) -> pd.DataFrame:
    rows = []
    for (moving_average, band, target_volatility), history in histories.items():
        holdout = _date_slice(history, cfg.holdout_start, None)
        metrics = _performance_metrics(holdout, 252.0)
        rows.append(
            {
                "moving_average_days": moving_average,
                "band": band,
                "target_volatility": target_volatility,
                "holdout_cagr": metrics["cagr"],
                "holdout_sharpe": metrics["sharpe"],
                "holdout_max_drawdown": metrics["max_drawdown"],
                "target_met": bool(metrics["cagr"] >= cfg.target_cagr),
            }
        )
    return pd.DataFrame(rows).sort_values("holdout_cagr", ascending=False).reset_index(drop=True)


def _cost_sensitivity(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    cfg: LeveragedTrendConfig,
    selected_key: tuple[int, float, float],
) -> pd.DataFrame:
    rows = []
    for total_cost_bps in (0.0, 10.0, 25.0, 50.0, 100.0):
        history = _run_strategy(prices, returns, cfg, *selected_key, total_cost_bps=total_cost_bps)
        metrics = _performance_metrics(_date_slice(history, cfg.holdout_start, None), 252.0)
        rows.append(
            {
                "total_cost_bps": total_cost_bps,
                "holdout_cagr": metrics["cagr"],
                "holdout_sharpe": metrics["sharpe"],
                "holdout_max_drawdown": metrics["max_drawdown"],
                "target_met": bool(metrics["cagr"] >= cfg.target_cagr),
            }
        )
    return pd.DataFrame(rows)


def _annual_returns(history: pd.DataFrame) -> pd.DataFrame:
    grouped = history.groupby(history.index.year)
    return pd.DataFrame(
        {
            "strategy_return": grouped["strategy_return"].apply(lambda values: float((1.0 + values).prod() - 1.0)),
            "tqqq_return": grouped["tqqq_return"].apply(lambda values: float((1.0 + values).prod() - 1.0)),
            "qqq_return": grouped["qqq_return"].apply(lambda values: float((1.0 + values).prod() - 1.0)),
        }
    ).rename_axis("year")


def _block_bootstrap(returns: pd.Series, cfg: LeveragedTrendConfig) -> pd.Series:
    values = returns.to_numpy(dtype=float)
    rng = np.random.default_rng(cfg.bootstrap_seed)
    block = min(cfg.bootstrap_block_days, len(values))
    blocks_needed = int(np.ceil(len(values) / block))
    cagr_samples = np.empty(cfg.bootstrap_samples)
    for sample in range(cfg.bootstrap_samples):
        starts = rng.integers(0, len(values) - block + 1, size=blocks_needed)
        simulated = np.concatenate([values[start : start + block] for start in starts])[: len(values)]
        terminal = float(np.prod(1.0 + simulated))
        cagr_samples[sample] = terminal ** (252.0 / len(values)) - 1.0
    return pd.Series(
        {
            "samples": float(cfg.bootstrap_samples),
            "block_days": float(block),
            "cagr_p05": float(np.quantile(cagr_samples, 0.05)),
            "cagr_median": float(np.quantile(cagr_samples, 0.50)),
            "cagr_p95": float(np.quantile(cagr_samples, 0.95)),
            "probability_cagr_at_least_target": float((cagr_samples >= cfg.target_cagr).mean()),
        }
    )


def _date_slice(history: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    mask = pd.Series(True, index=history.index)
    if start is not None:
        mask &= history.index >= pd.Timestamp(start)
    if end is not None:
        mask &= history.index <= pd.Timestamp(end)
    return history.loc[mask]


def _cagr(equity: pd.Series, periods_per_year: float) -> float:
    if equity.empty or equity.iloc[-1] <= 0:
        return -1.0
    return float(equity.iloc[-1] ** (periods_per_year / len(equity)) - 1.0)


def _drawdown(equity: pd.Series) -> pd.Series:
    peaks = np.maximum.accumulate(np.concatenate(([1.0], equity.to_numpy(dtype=float))))[1:]
    return pd.Series(1.0 - equity.to_numpy(dtype=float) / peaks, index=equity.index)


def _max_drawdown_duration(drawdown: pd.Series) -> int:
    longest = current = 0
    for value in drawdown:
        current = current + 1 if value > 1e-12 else 0
        longest = max(longest, current)
    return longest


def _profit_factor(returns: np.ndarray) -> float:
    gains = float(returns[returns > 0.0].sum())
    losses = float(-returns[returns < 0.0].sum())
    return float("inf") if losses == 0 else gains / losses


def _beta(returns: np.ndarray, benchmark: np.ndarray) -> float:
    covariance = np.cov(returns, benchmark, ddof=1)
    return 0.0 if covariance[1, 1] == 0 else float(covariance[0, 1] / covariance[1, 1])


def _safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 or not np.isfinite(denominator) else float(numerator / denominator)
