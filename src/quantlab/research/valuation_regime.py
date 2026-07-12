from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from quantlab.risk.var import historical_cvar, historical_var


@dataclass(frozen=True)
class ValuationRegimeConfig:
    train_months: int = 120
    validation_months: int = 36
    test_months: int = 36
    minimum_test_months: int = 12
    step_months: int = 36
    cheap_percentiles: tuple[float, ...] = (0.25, 0.35, 0.45)
    expensive_percentiles: tuple[float, ...] = (0.65, 0.75, 0.85)
    cheap_exposure: float = 1.0
    neutral_exposure: float = 0.6
    expensive_exposure: float = 0.25
    target_volatility: float = 0.12
    max_leverage: float = 1.25
    drawdown_limit: float = 0.20
    derisk_exposure: float = 0.25
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 2.0
    volatility_lookback: int = 12
    bootstrap_replicates: int = 500
    bootstrap_block_months: int = 12
    random_seed: int = 12

    def validate(self) -> None:
        if (
            min(
                self.train_months,
                self.validation_months,
                self.test_months,
                self.minimum_test_months,
                self.step_months,
                self.volatility_lookback,
            )
            <= 0
        ):
            raise ValueError("window lengths must be positive")
        if not all(0 < p < 1 for p in self.cheap_percentiles + self.expensive_percentiles):
            raise ValueError("percentiles must be in (0, 1)")
        if min(self.cheap_exposure, self.neutral_exposure, self.expensive_exposure, self.derisk_exposure) < 0:
            raise ValueError("exposures must be non-negative")
        if min(self.target_volatility, self.max_leverage) <= 0:
            raise ValueError("target_volatility and max_leverage must be positive")
        if not 0 < self.drawdown_limit < 1:
            raise ValueError("drawdown_limit must be in (0, 1)")
        if min(self.transaction_cost_bps, self.slippage_bps) < 0:
            raise ValueError("cost assumptions must be non-negative")
        if min(self.bootstrap_replicates, self.bootstrap_block_months) <= 0:
            raise ValueError("bootstrap settings must be positive")


@dataclass(frozen=True)
class TearSheet:
    metrics: pd.Series
    monthly_returns: pd.DataFrame
    drawdowns: pd.Series
    regime_breakdown: pd.DataFrame
    stress_tests: pd.DataFrame


@dataclass(frozen=True)
class RobustnessResult:
    grid: pd.DataFrame

    @property
    def positive_cost_scenarios(self) -> int:
        return int((self.grid["strategy_cagr"] > 0.0).sum())

    @property
    def worst_cost_cagr(self) -> float:
        return float(self.grid["strategy_cagr"].min())


@dataclass(frozen=True)
class ResearchDiagnostics:
    baseline_comparison: pd.DataFrame
    bootstrap_confidence_intervals: pd.DataFrame
    parameter_stability: pd.DataFrame
    overfitting_metrics: pd.Series


@dataclass(frozen=True)
class ValuationRegimeResult:
    history: pd.DataFrame
    folds: pd.DataFrame
    tear_sheet: TearSheet
    robustness: RobustnessResult
    diagnostics: ResearchDiagnostics
    config: ValuationRegimeConfig

    @property
    def final_equity(self) -> float:
        return float(self.history["strategy_equity"].iloc[-1])


def load_shiller_sp500_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "sp500", "dividend", "long_rate", "pe10"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    frame = frame.sort_values("date").drop_duplicates("date").set_index("date")
    for column in required - {"date"}:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["sp500", "pe10", "long_rate"]].isna().any().any():
        raise ValueError("sp500, pe10, and long_rate must be complete")
    if (frame[["sp500", "pe10"]] <= 0).any().any():
        raise ValueError("sp500 and pe10 must be positive")
    frame["dividend"] = frame["dividend"].fillna(0.0).clip(lower=0.0)
    return frame


def run_valuation_regime_walk_forward(
    data: pd.DataFrame,
    config: ValuationRegimeConfig | None = None,
) -> ValuationRegimeResult:
    cfg = ValuationRegimeConfig() if config is None else config
    cfg.validate()
    prepared = _prepare_monthly_market_data(data)
    if len(prepared) < cfg.train_months + cfg.validation_months + cfg.test_months + 2:
        raise ValueError("not enough observations for configured walk-forward windows")
    fold_rows: list[dict[str, float | pd.Timestamp]] = []
    test_histories: list[pd.DataFrame] = []
    start = 0
    while start + cfg.train_months + cfg.validation_months + cfg.minimum_test_months <= len(prepared):
        train = prepared.iloc[start : start + cfg.train_months]
        validation = prepared.iloc[start + cfg.train_months : start + cfg.train_months + cfg.validation_months]
        test_start = start + cfg.train_months + cfg.validation_months
        test = prepared.iloc[test_start : min(test_start + cfg.test_months, len(prepared))]
        params = _select_parameters(train, validation, cfg)
        selection = _selection_diagnostics(train, validation, test, cfg, params)
        test_history = _run_strategy_window(
            test, cfg, params["cheap_threshold"], params["expensive_threshold"], initial_equity=1.0
        )
        test_histories.append(test_history)
        fold_rows.append(
            {
                "train_start": train.index[0],
                "train_end": train.index[-1],
                "validation_start": validation.index[0],
                "validation_end": validation.index[-1],
                "test_start": test.index[0],
                "test_end": test.index[-1],
                "cheap_threshold": params["cheap_threshold"],
                "expensive_threshold": params["expensive_threshold"],
                "cheap_percentile": params["cheap_percentile"],
                "expensive_percentile": params["expensive_percentile"],
                "validation_sharpe": params["validation_sharpe"],
                "selected_test_rank_percentile": selection["selected_test_rank_percentile"],
                "test_return": float(
                    test_history["strategy_equity"].iloc[-1] / test_history["strategy_equity"].iloc[0] - 1.0
                ),
            }
        )
        start += cfg.step_months

    stitched = _stitch_test_histories(test_histories)
    benchmark = prepared.reindex(stitched.index)
    stitched["benchmark_return"] = benchmark["market_return"]
    stitched["benchmark_realized_volatility"] = benchmark["realized_volatility"]
    stitched["bond_return"] = benchmark["bond_return"]
    stitched["benchmark_equity"] = (1.0 + stitched["benchmark_return"]).cumprod()
    tear_sheet = _build_tear_sheet(stitched)
    robustness = _run_robustness(prepared, cfg)
    folds = pd.DataFrame(fold_rows)
    diagnostics = _build_research_diagnostics(stitched, folds, cfg)
    return ValuationRegimeResult(stitched, folds, tear_sheet, robustness, diagnostics, cfg)


def _prepare_monthly_market_data(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    if "date" in frame.columns:
        frame = frame.set_index(pd.to_datetime(frame["date"])).drop(columns=["date"])
    frame = frame.sort_index()
    price_return = frame["sp500"].pct_change()
    dividend_yield = (frame["dividend"].shift(1).fillna(0.0) / frame["sp500"].shift(1)).clip(lower=0.0) / 12.0
    market_return = (price_return + dividend_yield).dropna()
    out = frame.loc[market_return.index, ["sp500", "pe10", "long_rate"]].copy()
    out["market_return"] = market_return
    out["risk_free_return"] = (out["long_rate"].shift(1).fillna(out["long_rate"]) / 100.0) / 12.0
    yield_decimal = out["long_rate"] / 100.0
    out["bond_return"] = out["risk_free_return"] - 7.0 * yield_decimal.diff().fillna(0.0)
    out["signal_pe10"] = out["pe10"].shift(1)
    out["realized_volatility"] = out["market_return"].shift(1).rolling(12).std(ddof=1) * np.sqrt(12.0)
    out = out.dropna(subset=["signal_pe10", "realized_volatility"])
    return out


def _select_parameters(train: pd.DataFrame, validation: pd.DataFrame, cfg: ValuationRegimeConfig) -> dict[str, float]:
    best: dict[str, float] | None = None
    train_signal = train["signal_pe10"]
    for cheap_pct in cfg.cheap_percentiles:
        for expensive_pct in cfg.expensive_percentiles:
            if cheap_pct >= expensive_pct:
                continue
            cheap_threshold = float(train_signal.quantile(cheap_pct))
            expensive_threshold = float(train_signal.quantile(expensive_pct))
            validation_history = _run_strategy_window(
                validation, cfg, cheap_threshold, expensive_threshold, initial_equity=1.0
            )
            sharpe = _sharpe(validation_history["strategy_return"].to_numpy())
            if best is None or sharpe > best["validation_sharpe"]:
                best = {
                    "cheap_threshold": cheap_threshold,
                    "expensive_threshold": expensive_threshold,
                    "cheap_percentile": cheap_pct,
                    "expensive_percentile": expensive_pct,
                    "validation_sharpe": sharpe,
                }
    if best is None:
        raise ValueError("no valid threshold candidates")
    return best


def _selection_diagnostics(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    cfg: ValuationRegimeConfig,
    selected: dict[str, float],
) -> dict[str, float]:
    candidates = []
    train_signal = train["signal_pe10"]
    for cheap_pct in cfg.cheap_percentiles:
        for expensive_pct in cfg.expensive_percentiles:
            if cheap_pct >= expensive_pct:
                continue
            cheap_threshold = float(train_signal.quantile(cheap_pct))
            expensive_threshold = float(train_signal.quantile(expensive_pct))
            validation_history = _run_strategy_window(
                validation, cfg, cheap_threshold, expensive_threshold, initial_equity=1.0
            )
            test_history = _run_strategy_window(test, cfg, cheap_threshold, expensive_threshold, initial_equity=1.0)
            candidates.append(
                {
                    "cheap_percentile": cheap_pct,
                    "expensive_percentile": expensive_pct,
                    "validation_sharpe": _sharpe(validation_history["strategy_return"].to_numpy()),
                    "test_sharpe": _sharpe(test_history["strategy_return"].to_numpy()),
                }
            )
    grid = pd.DataFrame(candidates)
    selected_mask = np.isclose(grid["cheap_percentile"], selected["cheap_percentile"]) & np.isclose(
        grid["expensive_percentile"], selected["expensive_percentile"]
    )
    test_rank = grid["test_sharpe"].rank(method="average", pct=True)
    return {"selected_test_rank_percentile": float(test_rank.loc[selected_mask].iloc[0])}


def _run_strategy_window(
    data: pd.DataFrame,
    cfg: ValuationRegimeConfig,
    cheap_threshold: float,
    expensive_threshold: float,
    initial_equity: float,
) -> pd.DataFrame:
    rows = []
    equity = initial_equity
    peak = initial_equity
    previous_exposure = 0.0
    cost_rate = (cfg.transaction_cost_bps + cfg.slippage_bps) / 10_000.0
    for date, row in data.iterrows():
        raw_exposure = _valuation_exposure(row["signal_pe10"], cheap_threshold, expensive_threshold, cfg)
        realized_vol = max(float(row["realized_volatility"]), 1e-8)
        vol_scaled = min(cfg.max_leverage, raw_exposure * cfg.target_volatility / realized_vol)
        drawdown = 0.0 if peak <= 0 else max(1.0 - equity / peak, 0.0)
        exposure = min(vol_scaled, cfg.derisk_exposure) if drawdown >= cfg.drawdown_limit else vol_scaled
        turnover = abs(exposure - previous_exposure)
        trading_cost = turnover * cost_rate
        strategy_return = (
            exposure * float(row["market_return"]) + (1.0 - exposure) * float(row["risk_free_return"]) - trading_cost
        )
        equity *= 1.0 + strategy_return
        peak = max(peak, equity)
        rows.append(
            {
                "date": date,
                "signal_pe10": float(row["signal_pe10"]),
                "market_return": float(row["market_return"]),
                "risk_free_return": float(row["risk_free_return"]),
                "raw_exposure": float(raw_exposure),
                "exposure": float(exposure),
                "turnover": float(turnover),
                "trading_cost": float(trading_cost),
                "strategy_return": float(strategy_return),
                "strategy_equity": float(equity),
                "drawdown_control_active": bool(drawdown >= cfg.drawdown_limit),
            }
        )
        previous_exposure = exposure
    return pd.DataFrame(rows).set_index("date")


def _valuation_exposure(
    pe10: float, cheap_threshold: float, expensive_threshold: float, cfg: ValuationRegimeConfig
) -> float:
    if pe10 <= cheap_threshold:
        return cfg.cheap_exposure
    if pe10 >= expensive_threshold:
        return cfg.expensive_exposure
    return cfg.neutral_exposure


def _stitch_test_histories(histories: list[pd.DataFrame]) -> pd.DataFrame:
    out = pd.concat(histories).sort_index()
    out = out[~out.index.duplicated(keep="first")].copy()
    out["strategy_equity"] = (1.0 + out["strategy_return"]).cumprod()
    return out


def _build_tear_sheet(history: pd.DataFrame) -> TearSheet:
    returns = history["strategy_return"]
    benchmark = history["benchmark_return"]
    excess = returns - benchmark
    equity = history["strategy_equity"]
    drawdowns = 1.0 - equity / equity.cummax()
    metrics = pd.Series(
        {
            "cagr": _cagr(equity),
            "benchmark_cagr": _cagr(history["benchmark_equity"]),
            "volatility": float(returns.std(ddof=1) * np.sqrt(12.0)),
            "sharpe": _sharpe(returns.to_numpy()),
            "sortino": _sortino(returns.to_numpy()),
            "max_drawdown": float(drawdowns.max()),
            "calmar": _safe_divide(_cagr(equity), float(drawdowns.max())),
            "hit_rate": float((returns > 0.0).mean()),
            "profit_factor": _profit_factor(returns.to_numpy()),
            "average_turnover": float(history["turnover"].mean()),
            "average_exposure": float(history["exposure"].mean()),
            "beta": _beta(returns.to_numpy(), benchmark.to_numpy()),
            "alpha_annual": _alpha_annual(returns.to_numpy(), benchmark.to_numpy()),
            "tracking_error": float(excess.std(ddof=1) * np.sqrt(12.0)),
            "value_at_risk_95": historical_var(returns.to_numpy(), confidence=0.95),
            "conditional_var_95": historical_cvar(returns.to_numpy(), confidence=0.95),
            "max_drawdown_duration_months": _max_drawdown_duration(drawdowns),
            "total_cost": float(history["trading_cost"].sum()),
        }
    )
    monthly = returns.to_frame("strategy_return")
    monthly["benchmark_return"] = benchmark
    monthly["year"] = monthly.index.year
    monthly["month"] = monthly.index.month
    monthly_returns = monthly.pivot_table(index="year", columns="month", values="strategy_return", aggfunc="sum")
    regime_breakdown = (
        history.assign(regime=_regime_labels(history["signal_pe10"]))
        .groupby("regime")
        .agg(
            observations=("strategy_return", "count"),
            average_return=("strategy_return", "mean"),
            volatility=("strategy_return", "std"),
            average_exposure=("exposure", "mean"),
            hit_rate=("strategy_return", lambda values: float((values > 0).mean())),
        )
    )
    stress_tests = _stress_tests(history)
    return TearSheet(metrics, monthly_returns, drawdowns.rename("drawdown"), regime_breakdown, stress_tests)


def _build_research_diagnostics(
    history: pd.DataFrame, folds: pd.DataFrame, cfg: ValuationRegimeConfig
) -> ResearchDiagnostics:
    valid_trials = sum(cheap < expensive for cheap in cfg.cheap_percentiles for expensive in cfg.expensive_percentiles)
    parameter_stability = folds[
        [
            "test_start",
            "test_end",
            "cheap_percentile",
            "expensive_percentile",
            "cheap_threshold",
            "expensive_threshold",
            "validation_sharpe",
            "test_return",
            "selected_test_rank_percentile",
        ]
    ].copy()
    overfitting_metrics = pd.Series(
        {
            "probability_backtest_overfitting": float((folds["selected_test_rank_percentile"] <= 0.5).mean()),
            "deflated_sharpe_probability": _deflated_sharpe_probability(
                history["strategy_return"].to_numpy(), valid_trials
            ),
            "selection_trials_per_fold": float(valid_trials),
            "walk_forward_folds": float(len(folds)),
        }
    )
    return ResearchDiagnostics(
        baseline_comparison=_baseline_comparison(history, cfg),
        bootstrap_confidence_intervals=_block_bootstrap_confidence_intervals(history, cfg),
        parameter_stability=parameter_stability,
        overfitting_metrics=overfitting_metrics,
    )


def _baseline_comparison(history: pd.DataFrame, cfg: ValuationRegimeConfig) -> pd.DataFrame:
    market = history["benchmark_return"]
    risk_free = history["risk_free_return"]
    strategy = history["strategy_return"]
    lagged_vol = history["benchmark_realized_volatility"].clip(lower=1e-8)
    volatility_target_exposure = (cfg.target_volatility / lagged_vol).clip(upper=cfg.max_leverage)
    strategy_vol = float(strategy.std(ddof=1) * np.sqrt(12.0))
    market_vol = float(market.std(ddof=1) * np.sqrt(12.0))
    volatility_match = 0.0 if market_vol == 0 else min(cfg.max_leverage, strategy_vol / market_vol)
    beta_match = float(np.clip(_beta(strategy.to_numpy(), market.to_numpy()), 0.0, cfg.max_leverage))
    baselines = {
        "valuation_regime": strategy,
        "valuation_regime_bond_sleeve": history["exposure"] * market
        + (1.0 - history["exposure"]) * history["bond_return"]
        - history["trading_cost"],
        "buy_and_hold": market,
        "volatility_targeted_equity": volatility_target_exposure * market
        + (1.0 - volatility_target_exposure) * risk_free,
        "sixty_forty_proxy": 0.6 * market + 0.4 * history["bond_return"],
        "volatility_matched_equity": volatility_match * market + (1.0 - volatility_match) * risk_free,
        "beta_matched_equity": beta_match * market + (1.0 - beta_match) * risk_free,
    }
    rows = []
    for name, returns in baselines.items():
        equity = (1.0 + returns).cumprod()
        drawdown = 1.0 - equity / equity.cummax()
        rows.append(
            {
                "baseline": name,
                "cagr": _cagr(equity),
                "volatility": float(returns.std(ddof=1) * np.sqrt(12.0)),
                "sharpe": _sharpe(returns.to_numpy()),
                "max_drawdown": float(drawdown.max()),
                "beta": _beta(returns.to_numpy(), market.to_numpy()),
            }
        )
    return pd.DataFrame(rows).set_index("baseline")


def _block_bootstrap_confidence_intervals(history: pd.DataFrame, cfg: ValuationRegimeConfig) -> pd.DataFrame:
    strategy = history["strategy_return"].to_numpy(dtype=float)
    benchmark = history["benchmark_return"].to_numpy(dtype=float)
    rng = np.random.default_rng(cfg.random_seed)
    n = len(strategy)
    block = min(cfg.bootstrap_block_months, n)
    samples: dict[str, list[float]] = {"annualized_return": [], "sharpe": [], "alpha_annual": []}
    for _ in range(cfg.bootstrap_replicates):
        indices: list[int] = []
        while len(indices) < n:
            start = int(rng.integers(0, n))
            indices.extend(((start + np.arange(block)) % n).tolist())
        chosen = np.asarray(indices[:n], dtype=int)
        strategy_sample = strategy[chosen]
        benchmark_sample = benchmark[chosen]
        samples["annualized_return"].append(float(12.0 * strategy_sample.mean()))
        samples["sharpe"].append(_sharpe(strategy_sample))
        samples["alpha_annual"].append(_alpha_annual(strategy_sample, benchmark_sample))
    estimates = {
        "annualized_return": float(12.0 * strategy.mean()),
        "sharpe": _sharpe(strategy),
        "alpha_annual": _alpha_annual(strategy, benchmark),
    }
    rows = []
    for metric, values in samples.items():
        lower, upper = np.quantile(values, [0.025, 0.975])
        rows.append({"metric": metric, "estimate": estimates[metric], "lower_95": lower, "upper_95": upper})
    return pd.DataFrame(rows).set_index("metric")


def _deflated_sharpe_probability(returns: np.ndarray, selection_trials: int) -> float:
    values = np.asarray(returns, dtype=float)
    if len(values) < 3 or values.std(ddof=1) == 0:
        return 0.0
    monthly_sharpe = float(values.mean() / values.std(ddof=1))
    skew = float(pd.Series(values).skew())
    kurtosis = float(pd.Series(values).kurt() + 3.0)
    variance = max(
        (1.0 - skew * monthly_sharpe + ((kurtosis - 1.0) / 4.0) * monthly_sharpe**2) / (len(values) - 1),
        1e-12,
    )
    sharpe_std = math.sqrt(variance)
    trials = max(int(selection_trials), 2)
    normal = NormalDist()
    euler_gamma = 0.5772156649015329
    expected_max = sharpe_std * (
        (1.0 - euler_gamma) * normal.inv_cdf(1.0 - 1.0 / trials)
        + euler_gamma * normal.inv_cdf(1.0 - 1.0 / (trials * math.e))
    )
    return float(normal.cdf((monthly_sharpe - expected_max) / sharpe_std))


def _run_robustness(data: pd.DataFrame, cfg: ValuationRegimeConfig) -> RobustnessResult:
    rows = []
    for cost_bps in (0.0, 5.0, 10.0, 25.0, 50.0):
        scenario_cfg = ValuationRegimeConfig(
            train_months=cfg.train_months,
            validation_months=cfg.validation_months,
            test_months=cfg.test_months,
            minimum_test_months=cfg.minimum_test_months,
            step_months=cfg.step_months,
            cheap_percentiles=cfg.cheap_percentiles,
            expensive_percentiles=cfg.expensive_percentiles,
            cheap_exposure=cfg.cheap_exposure,
            neutral_exposure=cfg.neutral_exposure,
            expensive_exposure=cfg.expensive_exposure,
            target_volatility=cfg.target_volatility,
            max_leverage=cfg.max_leverage,
            drawdown_limit=cfg.drawdown_limit,
            derisk_exposure=cfg.derisk_exposure,
            transaction_cost_bps=cost_bps,
            slippage_bps=cfg.slippage_bps,
            volatility_lookback=cfg.volatility_lookback,
            bootstrap_replicates=cfg.bootstrap_replicates,
            bootstrap_block_months=cfg.bootstrap_block_months,
            random_seed=cfg.random_seed,
        )
        base = _run_without_nested_robustness(data, scenario_cfg)
        rows.append(
            {
                "transaction_cost_bps": cost_bps,
                "strategy_cagr": base["cagr"],
                "sharpe": base["sharpe"],
                "max_drawdown": base["max_drawdown"],
            }
        )
    return RobustnessResult(pd.DataFrame(rows))


def _run_without_nested_robustness(data: pd.DataFrame, cfg: ValuationRegimeConfig) -> pd.Series:
    fold_histories = []
    start = 0
    while start + cfg.train_months + cfg.validation_months + cfg.minimum_test_months <= len(data):
        train = data.iloc[start : start + cfg.train_months]
        validation = data.iloc[start + cfg.train_months : start + cfg.train_months + cfg.validation_months]
        test_start = start + cfg.train_months + cfg.validation_months
        test = data.iloc[test_start : min(test_start + cfg.test_months, len(data))]
        params = _select_parameters(train, validation, cfg)
        fold_histories.append(
            _run_strategy_window(
                test, cfg, params["cheap_threshold"], params["expensive_threshold"], initial_equity=1.0
            )
        )
        start += cfg.step_months
    history = _stitch_test_histories(fold_histories)
    benchmark = data.reindex(history.index)
    history["benchmark_return"] = benchmark["market_return"]
    history["benchmark_equity"] = (1.0 + history["benchmark_return"]).cumprod()
    return _build_tear_sheet(history).metrics


def _regime_labels(signal_pe10: pd.Series) -> pd.Series:
    low = signal_pe10.quantile(0.33)
    high = signal_pe10.quantile(0.67)
    return pd.Series(
        np.where(signal_pe10 <= low, "cheap", np.where(signal_pe10 >= high, "expensive", "neutral")),
        index=signal_pe10.index,
    )


def _stress_tests(history: pd.DataFrame) -> pd.DataFrame:
    volatility = history["benchmark_return"].rolling(12).std(ddof=1)
    high_vol_cutoff = volatility.quantile(0.8)
    scenarios = {
        "worst_12_months": history["strategy_return"].rolling(12).sum().min(),
        "high_vol_months": history.loc[volatility >= high_vol_cutoff, "strategy_return"].mean(),
        "expensive_months": history.loc[
            _regime_labels(history["signal_pe10"]) == "expensive", "strategy_return"
        ].mean(),
        "large_down_market_months": history.loc[
            history["benchmark_return"] <= history["benchmark_return"].quantile(0.1), "strategy_return"
        ].mean(),
    }
    return pd.DataFrame({"scenario_return": pd.Series(scenarios)})


def _cagr(equity: pd.Series) -> float:
    years = len(equity) / 12.0
    if years <= 0 or equity.iloc[0] <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)


def _sharpe(returns: np.ndarray) -> float:
    values = np.asarray(returns, dtype=float)
    std = values.std(ddof=1)
    return 0.0 if std == 0 else float(np.sqrt(12.0) * values.mean() / std)


def _sortino(returns: np.ndarray) -> float:
    values = np.asarray(returns, dtype=float)
    downside = values[values < 0.0]
    if len(downside) == 0 or downside.std(ddof=1) == 0:
        return 0.0
    return float(np.sqrt(12.0) * values.mean() / downside.std(ddof=1))


def _profit_factor(returns: np.ndarray) -> float:
    values = np.asarray(returns, dtype=float)
    gains = values[values > 0.0].sum()
    losses = -values[values < 0.0].sum()
    return float("inf") if losses == 0 else float(gains / losses)


def _beta(returns: np.ndarray, benchmark: np.ndarray) -> float:
    cov = np.cov(returns, benchmark, ddof=1)
    return 0.0 if cov[1, 1] == 0 else float(cov[0, 1] / cov[1, 1])


def _alpha_annual(returns: np.ndarray, benchmark: np.ndarray) -> float:
    beta = _beta(returns, benchmark)
    return float(12.0 * (np.mean(returns) - beta * np.mean(benchmark)))


def _max_drawdown_duration(drawdowns: pd.Series) -> float:
    max_duration = 0
    current = 0
    for value in drawdowns:
        if value > 1e-12:
            current += 1
            max_duration = max(max_duration, current)
        else:
            current = 0
    return float(max_duration)


def _safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)
