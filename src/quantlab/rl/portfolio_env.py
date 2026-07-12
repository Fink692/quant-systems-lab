from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioTradingState:
    time_index: int
    prices: pd.Series
    weights: pd.Series
    return_history: pd.DataFrame
    equity: float
    peak_equity: float
    drawdown: float
    trailing_volatility: float


PortfolioPolicy = Callable[[PortfolioTradingState], np.ndarray | pd.Series]


@dataclass(frozen=True)
class PortfolioPolicyResult:
    history: pd.DataFrame
    weights: pd.DataFrame

    @property
    def total_return(self) -> float:
        equity = self.history["equity"]
        return float(equity.iloc[-1] / equity.iloc[0] - 1.0)

    @property
    def max_drawdown(self) -> float:
        return float(self.history["drawdown"].max())

    @property
    def average_turnover(self) -> float:
        return float(self.history["turnover"].mean())

    @property
    def sharpe(self) -> float:
        returns = self.history["portfolio_return"].iloc[1:]
        if len(returns) < 2 or returns.std(ddof=1) == 0.0:
            return 0.0
        return float(np.sqrt(252.0) * returns.mean() / returns.std(ddof=1))


class PortfolioTradingEnv:
    """Multi-asset portfolio trading environment for risk-constrained RL baselines."""

    def __init__(
        self,
        prices: pd.DataFrame | np.ndarray,
        initial_equity: float = 1_000_000.0,
        transaction_cost_bps: float = 1.0,
        drawdown_penalty: float = 0.0,
        volatility_penalty: float = 0.0,
        volatility_window: int = 20,
        long_only: bool = True,
        max_leverage: float = 1.0,
    ) -> None:
        if initial_equity <= 0 or transaction_cost_bps < 0 or drawdown_penalty < 0 or volatility_penalty < 0:
            raise ValueError("equity, costs, and penalties must be non-negative with positive equity")
        if volatility_window < 1 or max_leverage <= 0:
            raise ValueError("volatility_window and max_leverage must be positive")
        self.prices = _as_price_frame(prices)
        self.asset_names = list(self.prices.columns)
        self.returns = self.prices.pct_change().dropna()
        self.initial_equity = float(initial_equity)
        self.transaction_cost = transaction_cost_bps / 10_000.0
        self.drawdown_penalty = float(drawdown_penalty)
        self.volatility_penalty = float(volatility_penalty)
        self.volatility_window = int(volatility_window)
        self.long_only = bool(long_only)
        self.max_leverage = float(max_leverage)
        self.reset()

    def reset(self) -> PortfolioTradingState:
        self.time_index = 0
        self.equity = self.initial_equity
        self.peak_equity = self.initial_equity
        self.weights = pd.Series(0.0, index=self.asset_names, name="weight")
        self.realized_returns: list[float] = []
        return self.state

    @property
    def state(self) -> PortfolioTradingState:
        drawdown = 1.0 - self.equity / self.peak_equity if self.peak_equity > 0 else 0.0
        trailing = np.asarray(self.realized_returns[-self.volatility_window :], dtype=float)
        trailing_volatility = float(trailing.std(ddof=1) * np.sqrt(252.0)) if len(trailing) > 1 else 0.0
        return PortfolioTradingState(
            time_index=int(self.time_index),
            prices=self.prices.iloc[self.time_index].rename("price"),
            weights=self.weights.copy(),
            return_history=self.returns.iloc[: self.time_index].copy(),
            equity=float(self.equity),
            peak_equity=float(self.peak_equity),
            drawdown=float(drawdown),
            trailing_volatility=trailing_volatility,
        )

    def step(
        self, target_weights: np.ndarray | pd.Series
    ) -> tuple[PortfolioTradingState, float, bool, dict[str, float]]:
        if self.time_index >= len(self.returns):
            raise RuntimeError("environment is already done")
        old_equity = self.equity
        target = _normalize_weights(target_weights, self.asset_names, self.long_only, self.max_leverage)
        turnover = float((target - self.weights).abs().sum())
        transaction_cost = old_equity * turnover * self.transaction_cost
        equity_after_cost = old_equity - transaction_cost
        period_returns = self.returns.iloc[self.time_index]
        portfolio_return = float(target @ period_returns)
        self.equity = equity_after_cost * (1.0 + portfolio_return)
        self.weights = target.rename("weight")
        self.time_index += 1
        self.peak_equity = max(self.peak_equity, self.equity)
        self.realized_returns.append(float(self.equity / old_equity - 1.0))
        state = self.state
        reward = (
            self.equity / old_equity
            - 1.0
            - self.drawdown_penalty * state.drawdown
            - self.volatility_penalty * state.trailing_volatility
        )
        done = self.time_index == len(self.returns)
        info = {
            "portfolio_return": portfolio_return,
            "turnover": turnover,
            "transaction_cost": float(transaction_cost),
            "drawdown": state.drawdown,
            "trailing_volatility": state.trailing_volatility,
        }
        return state, float(reward), done, info


def constant_mix_policy(weights: np.ndarray | pd.Series) -> PortfolioPolicy:
    def policy(state: PortfolioTradingState) -> pd.Series:
        return _normalize_weights(weights, list(state.weights.index), long_only=True, max_leverage=1.0)

    return policy


def momentum_rotation_policy(lookback: int = 20, top_n: int = 1) -> PortfolioPolicy:
    """Allocate equally to the assets with the strongest trailing price momentum."""
    if lookback < 1 or top_n < 1:
        raise ValueError("lookback and top_n must be positive")

    def policy(state: PortfolioTradingState) -> pd.Series:
        if len(state.return_history) < lookback:
            return pd.Series(1.0 / len(state.prices), index=state.prices.index)
        momentum = state.return_history.tail(lookback).sum(axis=0)
        ranked = momentum.rank(method="first", ascending=False)
        chosen = ranked <= min(top_n, len(momentum))
        weights = pd.Series(0.0, index=momentum.index)
        weights.loc[chosen] = 1.0 / chosen.sum()
        return weights

    return policy


def run_portfolio_policy(env: PortfolioTradingEnv, policy: PortfolioPolicy) -> PortfolioPolicyResult:
    rows: list[dict[str, float]] = []
    weight_rows: list[pd.Series] = []
    state = env.reset()
    rows.append(_row_from_state(state, reward=0.0, portfolio_return=0.0, turnover=0.0, transaction_cost=0.0))
    weight_rows.append(state.weights.rename(env.prices.index[0]))
    done = False
    while not done:
        state, reward, done, info = env.step(policy(state))
        rows.append(
            _row_from_state(
                state,
                reward=reward,
                portfolio_return=info["portfolio_return"],
                turnover=info["turnover"],
                transaction_cost=info["transaction_cost"],
            )
        )
        weight_rows.append(state.weights.rename(env.prices.index[state.time_index]))
    return PortfolioPolicyResult(
        history=pd.DataFrame(rows, index=env.prices.index[: len(rows)]),
        weights=pd.DataFrame(weight_rows),
    )


def _row_from_state(
    state: PortfolioTradingState,
    reward: float,
    portfolio_return: float,
    turnover: float,
    transaction_cost: float,
) -> dict[str, float]:
    return {
        "time_index": float(state.time_index),
        "equity": float(state.equity),
        "reward": float(reward),
        "portfolio_return": float(portfolio_return),
        "turnover": float(turnover),
        "transaction_cost": float(transaction_cost),
        "drawdown": float(state.drawdown),
        "trailing_volatility": float(state.trailing_volatility),
    }


def _as_price_frame(prices: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    if isinstance(prices, pd.DataFrame):
        frame = prices.astype(float)
    else:
        values = np.asarray(prices, dtype=float)
        if values.ndim != 2:
            raise ValueError("prices must be a two-dimensional time-by-asset matrix")
        frame = pd.DataFrame(values, columns=[f"asset_{idx}" for idx in range(values.shape[1])])
    if frame.shape[0] < 2 or frame.shape[1] < 1 or frame.isna().any().any() or (frame <= 0).any().any():
        raise ValueError("prices must be positive, complete, and contain at least two rows")
    return frame


def _normalize_weights(
    weights: np.ndarray | pd.Series,
    asset_names: list[str],
    long_only: bool,
    max_leverage: float,
) -> pd.Series:
    series = (
        pd.Series(weights, index=asset_names, dtype=float)
        if not isinstance(weights, pd.Series)
        else weights.reindex(asset_names).astype(float)
    )
    if series.isna().any() or not np.isfinite(series.to_numpy()).all():
        raise ValueError("target weights must be finite and cover every asset")
    if long_only:
        series = series.clip(lower=0.0)
        total = float(series.sum())
        if total <= 0:
            return pd.Series(0.0, index=asset_names, name="weight")
        series = series / total
    gross = float(series.abs().sum())
    if gross > max_leverage:
        series = series * (max_leverage / gross)
    return series.rename("weight")
