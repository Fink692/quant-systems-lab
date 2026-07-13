from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LeveragedTrendStressConfig:
    evaluation_start: str = "2000-01-01"
    moving_average_days: int = 200
    volatility_lookback: int = 21
    target_volatility: float = 0.30
    max_exposure: float = 1.0
    leverage_multiple: float = 3.0
    financing_units: float = 2.0
    annual_fund_drag: float = 0.0095
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 5.0
    target_cagr: float = 0.20

    def validate(self) -> None:
        if min(self.moving_average_days, self.volatility_lookback) <= 1:
            raise ValueError("lookbacks must exceed one session")
        if min(self.target_volatility, self.max_exposure, self.leverage_multiple) <= 0:
            raise ValueError("risk settings must be positive")
        if min(self.financing_units, self.annual_fund_drag, self.transaction_cost_bps, self.slippage_bps) < 0:
            raise ValueError("drag and cost settings must be non-negative")
        if not 0 < self.target_cagr < 1:
            raise ValueError("target_cagr must be in (0, 1)")


@dataclass(frozen=True)
class LeveragedTrendStressResult:
    history: pd.DataFrame
    period_metrics: pd.DataFrame
    drag_sensitivity: pd.DataFrame
    reconciliation: pd.Series
    config: LeveragedTrendStressConfig

    @property
    def long_history_target_met(self) -> bool:
        return bool(self.period_metrics.loc["full_history", "cagr"] >= self.config.target_cagr)


def load_qqq_fred_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    missing = {"date", "qqq", "dff"} - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    frame = frame.sort_values("date").drop_duplicates("date").set_index("date")[["qqq", "dff"]]
    frame = frame.apply(pd.to_numeric, errors="coerce")
    if frame.isna().any().any() or (frame <= 0).any().any():
        raise ValueError("QQQ and DFF inputs must be complete and positive")
    return frame


def run_leveraged_trend_stress(
    inputs: pd.DataFrame,
    actual_tqqq: pd.Series | None = None,
    config: LeveragedTrendStressConfig | None = None,
) -> LeveragedTrendStressResult:
    cfg = LeveragedTrendStressConfig() if config is None else config
    cfg.validate()
    data = inputs[["qqq", "dff"]].copy().sort_index()
    if data.index.has_duplicates or data.isna().any().any() or (data <= 0).any().any():
        raise ValueError("stress inputs require unique dates and complete positive values")
    history = _run_scenario(data, cfg, cfg.annual_fund_drag)
    history = history.loc[history.index >= pd.Timestamp(cfg.evaluation_start)].copy()
    periods = {
        "dotcom_and_gfc": ("2000-01-01", "2009-12-31"),
        "pre_tqqq": ("2000-01-01", "2010-02-10"),
        "tqqq_pre_holdout": ("2010-02-11", "2020-12-31"),
        "published_holdout": ("2021-01-01", None),
        "full_history": (cfg.evaluation_start, None),
    }
    rows = []
    for name, (start, end) in periods.items():
        window = history.loc[history.index >= pd.Timestamp(start)]
        if end is not None:
            window = window.loc[window.index <= pd.Timestamp(end)]
        rows.append({"period": name, **_metrics(window)})
    period_metrics = pd.DataFrame(rows).set_index("period")

    drag_rows = []
    for annual_drag in (0.0095, 0.015, 0.025, 0.04):
        scenario = _run_scenario(data, cfg, annual_drag)
        scenario = scenario.loc[scenario.index >= pd.Timestamp(cfg.evaluation_start)]
        drag_rows.append({"annual_drag": annual_drag, **_metrics(scenario)})
    reconciliation = _reconcile(history, actual_tqqq) if actual_tqqq is not None else pd.Series(dtype=float)
    return LeveragedTrendStressResult(history, period_metrics, pd.DataFrame(drag_rows), reconciliation, cfg)


def _run_scenario(data: pd.DataFrame, cfg: LeveragedTrendStressConfig, annual_drag: float) -> pd.DataFrame:
    qqq_return = data["qqq"].pct_change().fillna(0.0)
    cash_return = (data["dff"].shift(1).bfill() / 100.0 / 360.0).clip(lower=0.0)
    levered_return = cfg.leverage_multiple * qqq_return - cfg.financing_units * cash_return - annual_drag / 252.0
    synthetic_price = (1.0 + levered_return).cumprod()
    moving_average = synthetic_price.rolling(cfg.moving_average_days).mean()
    signal = (synthetic_price > moving_average).astype(float)
    realized_volatility = levered_return.rolling(cfg.volatility_lookback).std(ddof=1) * np.sqrt(252.0)
    desired = signal * (cfg.target_volatility / realized_volatility.clip(lower=1e-8)).clip(upper=cfg.max_exposure)
    exposure = desired.shift(1).fillna(0.0)
    turnover = exposure.diff().abs().fillna(exposure.abs())
    trading_cost = turnover * (cfg.transaction_cost_bps + cfg.slippage_bps) / 10_000.0
    strategy_return = exposure * levered_return + (1.0 - exposure) * cash_return - trading_cost
    return pd.DataFrame(
        {
            "qqq_return": qqq_return,
            "cash_return": cash_return,
            "levered_return": levered_return,
            "synthetic_price": synthetic_price,
            "moving_average": moving_average,
            "signal": signal,
            "realized_volatility": realized_volatility,
            "exposure": exposure,
            "turnover": turnover,
            "trading_cost": trading_cost,
            "strategy_return": strategy_return,
        },
        index=data.index,
    )


def _metrics(history: pd.DataFrame) -> dict[str, float]:
    returns = history["strategy_return"]
    equity = (1.0 + returns).cumprod()
    cagr = float(equity.iloc[-1] ** (252.0 / len(equity)) - 1.0)
    peaks = np.maximum.accumulate(np.concatenate(([1.0], equity.to_numpy())))[1:]
    max_drawdown = float(np.max(1.0 - equity.to_numpy() / peaks))
    volatility = float(returns.std(ddof=1) * np.sqrt(252.0))
    return {
        "cagr": cagr,
        "sharpe": 0.0 if volatility == 0 else float(returns.mean() * 252.0 / volatility),
        "max_drawdown": max_drawdown,
        "observations": float(len(history)),
    }


def _reconcile(history: pd.DataFrame, actual_tqqq: pd.Series) -> pd.Series:
    actual_return = actual_tqqq.sort_index().pct_change().rename("actual")
    joined = pd.concat([history["levered_return"].rename("synthetic"), actual_return], axis=1).dropna()
    residual = joined["actual"] - joined["synthetic"]
    return pd.Series(
        {
            "overlap_observations": float(len(joined)),
            "daily_return_correlation": float(joined.corr().iloc[0, 1]),
            "actual_minus_synthetic_annual_mean": float(residual.mean() * 252.0),
            "annualized_tracking_error": float(residual.std(ddof=1) * np.sqrt(252.0)),
        }
    )
