from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExecutionTimingConfig:
    moving_average_days: int = 200
    volatility_lookback: int = 21
    target_volatility: float = 0.30
    max_exposure: float = 1.0
    total_cost_bps: float = 10.0


@dataclass(frozen=True)
class ExecutionTimingResult:
    history: pd.DataFrame
    period_metrics: pd.DataFrame
    config: ExecutionTimingConfig


def load_adjusted_ohlc_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "tqqq_open", "tqqq_close", "bil_open", "bil_close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    frame = frame.sort_values("date").drop_duplicates("date").set_index("date")
    frame = frame[["tqqq_open", "tqqq_close", "bil_open", "bil_close"]].apply(pd.to_numeric, errors="coerce")
    if frame.isna().any().any() or (frame <= 0).any().any():
        raise ValueError("adjusted OHLC values must be complete and positive")
    return frame


def run_execution_timing_audit(
    ohlc: pd.DataFrame,
    config: ExecutionTimingConfig | None = None,
) -> ExecutionTimingResult:
    cfg = ExecutionTimingConfig() if config is None else config
    if min(cfg.moving_average_days, cfg.volatility_lookback) <= 1:
        raise ValueError("lookbacks must exceed one session")
    if min(cfg.target_volatility, cfg.max_exposure) <= 0 or cfg.total_cost_bps < 0:
        raise ValueError("risk settings must be positive and costs non-negative")
    data = ohlc.copy().sort_index()
    close_return = data["tqqq_close"].pct_change()
    moving_average = data["tqqq_close"].rolling(cfg.moving_average_days).mean()
    realized_volatility = close_return.rolling(cfg.volatility_lookback).std(ddof=1) * np.sqrt(252.0)
    desired = (data["tqqq_close"] > moving_average).astype(float) * (
        cfg.target_volatility / realized_volatility.clip(lower=1e-8)
    ).clip(upper=cfg.max_exposure)
    new_exposure = desired.shift(1).fillna(0.0)
    old_exposure = desired.shift(2).fillna(0.0)
    turnover = (new_exposure - old_exposure).abs()
    cost = turnover * cfg.total_cost_bps / 10_000.0

    risky_overnight = data["tqqq_open"] / data["tqqq_close"].shift(1) - 1.0
    cash_overnight = data["bil_open"] / data["bil_close"].shift(1) - 1.0
    risky_intraday = data["tqqq_close"] / data["tqqq_open"] - 1.0
    cash_intraday = data["bil_close"] / data["bil_open"] - 1.0
    overnight = old_exposure * risky_overnight + (1.0 - old_exposure) * cash_overnight
    intraday = new_exposure * risky_intraday + (1.0 - new_exposure) * cash_intraday
    next_open_return = (1.0 + overnight) * (1.0 + intraday) - 1.0 - cost

    cash_close_return = data["bil_close"].pct_change()
    close_to_close_return = new_exposure * close_return + (1.0 - new_exposure) * cash_close_return - cost
    history = pd.DataFrame(
        {
            "desired_exposure": desired,
            "old_exposure": old_exposure,
            "new_exposure": new_exposure,
            "turnover": turnover,
            "trading_cost": cost,
            "overnight_return": overnight,
            "intraday_return": intraday,
            "next_open_return": next_open_return,
            "close_to_close_return": close_to_close_return,
        },
        index=data.index,
    )
    periods = {
        "train": ("2011-01-01", "2016-12-31"),
        "validation": ("2017-01-01", "2020-12-31"),
        "holdout": ("2021-01-01", None),
        "full": ("2011-01-01", None),
    }
    rows = []
    for period, (start, end) in periods.items():
        window = history.loc[history.index >= pd.Timestamp(start)]
        if end is not None:
            window = window.loc[window.index <= pd.Timestamp(end)]
        for convention in ("next_open", "close_to_close"):
            metrics = _metrics(window[f"{convention}_return"].dropna())
            rows.append({"period": period, "convention": convention, **metrics})
    return ExecutionTimingResult(history, pd.DataFrame(rows).set_index(["period", "convention"]), cfg)


def _metrics(returns: pd.Series) -> dict[str, float]:
    equity = (1.0 + returns).cumprod()
    cagr = float(equity.iloc[-1] ** (252.0 / len(equity)) - 1.0)
    peaks = np.maximum.accumulate(np.concatenate(([1.0], equity.to_numpy())))[1:]
    max_drawdown = float(np.max(1.0 - equity.to_numpy() / peaks))
    volatility = float(returns.std(ddof=1) * np.sqrt(252.0))
    return {
        "cagr": cagr,
        "sharpe": 0.0 if volatility == 0 else float(returns.mean() * 252.0 / volatility),
        "max_drawdown": max_drawdown,
        "observations": float(len(returns)),
    }
