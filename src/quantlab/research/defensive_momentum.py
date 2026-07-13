from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ASSETS = ("qqq", "gld", "tlt")


@dataclass(frozen=True)
class DefensiveMomentumConfig:
    development_start: str = "2005-01-01"
    evaluation_start: str = "2017-01-01"
    momentum_days: tuple[int, ...] = (63, 126, 252)
    trend_days: tuple[int, ...] = (100, 200)
    target_volatilities: tuple[float, ...] = (0.25, 0.30, 0.40, 0.50)
    max_leverages: tuple[float, ...] = (1.5, 2.0, 3.0)
    volatility_days: int = 63
    rebalance_frequencies: tuple[str, ...] = ("weekly", "monthly")
    transaction_cost_bps: float = 10.0
    annual_excess_leverage_drag: float = 0.01
    target_cagr: float = 0.20

    def validate(self) -> None:
        if not self.momentum_days or not self.trend_days or not self.target_volatilities or not self.max_leverages:
            raise ValueError("parameter grid cannot be empty")
        if min((*self.momentum_days, *self.trend_days, self.volatility_days)) <= 1:
            raise ValueError("lookbacks must exceed one session")
        if min((*self.target_volatilities, *self.max_leverages)) <= 0:
            raise ValueError("risk settings must be positive")
        if set(self.rebalance_frequencies) - {"weekly", "monthly"}:
            raise ValueError("rebalance frequencies must be weekly or monthly")
        if min(self.transaction_cost_bps, self.annual_excess_leverage_drag) < 0:
            raise ValueError("cost and drag settings must be non-negative")
        if pd.Timestamp(self.evaluation_start) <= pd.Timestamp(self.development_start):
            raise ValueError("evaluation must begin after development")


@dataclass(frozen=True)
class DefensiveMomentumResult:
    selected_parameters: pd.Series
    selected_history: pd.DataFrame
    grid_metrics: pd.DataFrame
    period_metrics: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    config: DefensiveMomentumConfig

    @property
    def evaluation_target_met(self) -> bool:
        return bool(self.period_metrics.loc["evaluation", "cagr"] >= self.config.target_cagr)


def load_defensive_momentum_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "dff"} | {f"{asset}_{field}" for asset in ASSETS for field in ("open", "close")}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    frame = frame.sort_values("date").drop_duplicates("date").set_index("date")
    frame = frame[sorted(required - {"date"})].apply(pd.to_numeric, errors="coerce")
    price_columns = [column for column in frame if column != "dff"]
    if frame.isna().any().any() or (frame[price_columns] <= 0).any().any() or (frame["dff"] < 0).any():
        raise ValueError("prices must be positive and DFF must be non-negative")
    return frame


def run_defensive_momentum_study(
    inputs: pd.DataFrame,
    config: DefensiveMomentumConfig | None = None,
) -> DefensiveMomentumResult:
    cfg = DefensiveMomentumConfig() if config is None else config
    cfg.validate()
    data = inputs.copy().sort_index()
    combinations = list(
        product(
            cfg.momentum_days,
            cfg.trend_days,
            cfg.target_volatilities,
            cfg.max_leverages,
            cfg.rebalance_frequencies,
        )
    )
    histories: dict[tuple[int, int, float, float, str], pd.DataFrame] = {}
    rows = []
    development_end = pd.Timestamp(cfg.evaluation_start) - pd.Timedelta(days=1)
    for parameters in combinations:
        history = _run_candidate(data, cfg, *parameters)
        histories[parameters] = history
        development = history.loc[
            (history.index >= pd.Timestamp(cfg.development_start)) & (history.index <= development_end)
        ]
        evaluation = history.loc[history.index >= pd.Timestamp(cfg.evaluation_start)]
        full = history.loc[history.index >= pd.Timestamp(cfg.development_start)]
        rows.append(
            {
                "momentum_days": parameters[0],
                "trend_days": parameters[1],
                "target_volatility": parameters[2],
                "max_leverage": parameters[3],
                "rebalance_frequency": parameters[4],
                **{f"development_{key}": value for key, value in _metrics(development).items()},
                **{f"evaluation_{key}": value for key, value in _metrics(evaluation).items()},
                **{f"full_{key}": value for key, value in _metrics(full).items()},
            }
        )
    grid = pd.DataFrame(rows).sort_values(
        ["development_sharpe", "development_cagr"], ascending=False, ignore_index=True
    )
    selected = grid.iloc[0]
    key = (
        int(selected["momentum_days"]),
        int(selected["trend_days"]),
        float(selected["target_volatility"]),
        float(selected["max_leverage"]),
        str(selected["rebalance_frequency"]),
    )
    selected_history = histories[key]
    periods = {
        "development": (cfg.development_start, development_end),
        "evaluation": (cfg.evaluation_start, None),
        "full": (cfg.development_start, None),
    }
    period_rows = []
    for name, (start, end) in periods.items():
        window = selected_history.loc[selected_history.index >= pd.Timestamp(start)]
        if end is not None:
            window = window.loc[window.index <= end]
        period_rows.append({"period": name, **_metrics(window)})
    cost_rows = []
    for cost_bps in (5.0, 10.0, 25.0, 50.0):
        scenario_cfg = DefensiveMomentumConfig(**{**cfg.__dict__, "transaction_cost_bps": cost_bps})
        scenario = _run_candidate(data, scenario_cfg, *key)
        window = scenario.loc[scenario.index >= pd.Timestamp(cfg.development_start)]
        cost_rows.append({"cost_bps": cost_bps, **_metrics(window)})
    return DefensiveMomentumResult(
        selected,
        selected_history,
        grid,
        pd.DataFrame(period_rows).set_index("period"),
        pd.DataFrame(cost_rows),
        cfg,
    )


def _run_candidate(
    data: pd.DataFrame,
    cfg: DefensiveMomentumConfig,
    momentum_days: int,
    trend_days: int,
    target_volatility: float,
    max_leverage: float,
    rebalance_frequency: str,
) -> pd.DataFrame:
    close = pd.DataFrame({asset: data[f"{asset}_close"] for asset in ASSETS})
    open_price = pd.DataFrame({asset: data[f"{asset}_open"] for asset in ASSETS})
    close_return = close.pct_change()
    score = (close / close.shift(momentum_days) - 1.0).where(close > close.rolling(trend_days).mean())
    has_candidate = score.notna().any(axis=1)
    selected_asset = score.fillna(-np.inf).idxmax(axis=1).where(has_candidate)
    volatility = close_return.rolling(cfg.volatility_days).std(ddof=1) * np.sqrt(252.0)
    desired = pd.DataFrame(0.0, index=data.index, columns=ASSETS)
    for asset in ASSETS:
        leverage = (target_volatility / volatility[asset].clip(lower=0.05)).clip(upper=max_leverage)
        desired[asset] = (selected_asset == asset).astype(float) * leverage

    period = data.index.to_period("W-FRI" if rebalance_frequency == "weekly" else "M")
    period_series = pd.Series(period, index=data.index)
    is_period_end = period_series.ne(period_series.shift(-1))
    new_weight = desired.where(is_period_end).ffill().shift(1).fillna(0.0)
    old_weight = new_weight.shift(1).fillna(0.0)
    old_leverage = old_weight.sum(axis=1)
    new_leverage = new_weight.sum(axis=1)
    overnight_asset_return = open_price / close.shift(1) - 1.0
    intraday_asset_return = close / open_price - 1.0
    cash_return = (data["dff"].shift(1).bfill() / 100.0 / 360.0).clip(lower=0.0)
    overnight = (old_weight * overnight_asset_return).sum(axis=1) + (1.0 - old_leverage) * cash_return * (2 / 3)
    intraday = (new_weight * intraday_asset_return).sum(axis=1) + (1.0 - new_leverage) * cash_return * (1 / 3)
    turnover = (new_weight - old_weight).abs().sum(axis=1)
    trading_cost = turnover * cfg.transaction_cost_bps / 10_000.0
    excess_leverage = np.maximum(old_leverage - 1.0, 0.0) * (2 / 3) + np.maximum(new_leverage - 1.0, 0.0) * (1 / 3)
    leverage_drag = excess_leverage * cfg.annual_excess_leverage_drag / 252.0
    strategy_return = (1.0 + overnight) * (1.0 + intraday) - 1.0 - trading_cost - leverage_drag
    return pd.DataFrame(
        {
            "selected_asset": selected_asset,
            "old_leverage": old_leverage,
            "new_leverage": new_leverage,
            "turnover": turnover,
            "trading_cost": trading_cost,
            "leverage_drag": leverage_drag,
            "overnight_return": overnight,
            "intraday_return": intraday,
            "strategy_return": strategy_return,
        },
        index=data.index,
    )


def _metrics(history: pd.DataFrame) -> dict[str, float]:
    returns = history["strategy_return"].dropna()
    equity = (1.0 + returns).cumprod()
    cagr = float(equity.iloc[-1] ** (252.0 / len(equity)) - 1.0)
    peaks = np.maximum.accumulate(np.concatenate(([1.0], equity.to_numpy())))[1:]
    volatility = float(returns.std(ddof=1) * np.sqrt(252.0))
    return {
        "cagr": cagr,
        "sharpe": 0.0 if volatility == 0 else float(returns.mean() * 252.0 / volatility),
        "max_drawdown": float(np.max(1.0 - equity.to_numpy() / peaks)),
        "observations": float(len(returns)),
    }
