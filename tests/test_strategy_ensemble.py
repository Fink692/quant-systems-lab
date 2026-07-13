from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantlab.research.bitcoin_trend import load_bitcoin_coinbase_csv, run_frozen_bitcoin_candidate
from quantlab.research.defensive_momentum import (
    load_defensive_momentum_csv,
    run_frozen_defensive_monthly_candidate,
)
from quantlab.research.strategy_ensemble import run_fixed_strategy_ensemble

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def published_result():
    bitcoin = run_frozen_bitcoin_candidate(load_bitcoin_coinbase_csv(ROOT / "data/real/coinbase_btc_dff.csv"))[
        "strategy_return"
    ]
    defensive = run_frozen_defensive_monthly_candidate(
        load_defensive_momentum_csv(ROOT / "data/real/defensive_momentum_ohlc.csv")
    )["strategy_return"]
    return run_fixed_strategy_ensemble(bitcoin, defensive)


def test_equal_weight_ensemble_accounting() -> None:
    index = pd.date_range("2024-01-30", periods=4, freq="D")
    bitcoin = pd.Series([0.10, 0.0, 0.0, 0.0], index=index)
    defensive = pd.Series([0.0, 0.0, 0.0, 0.0], index=index)
    result = run_fixed_strategy_ensemble(bitcoin, defensive, bootstrap_samples=5, block_size=2)
    history = result.history
    assert np.isclose(history.iloc[0]["strategy_return"], 0.05)
    expected_turnover = 2 * abs((0.5 * 1.1 / 1.05) - 0.5)
    assert np.isclose(history.loc[pd.Timestamp("2024-02-01"), "turnover"], expected_turnover)
    assert np.isclose(
        history.loc[pd.Timestamp("2024-02-01"), "strategy_return"],
        -expected_turnover * 10.0 / 10_000.0,
    )


def test_weekend_month_boundary_waits_for_defensive_session() -> None:
    index = pd.date_range("2024-05-30", periods=5, freq="D")
    bitcoin = pd.Series([0.10, 0.0, 0.0, 0.0, 0.0], index=index)
    defensive = pd.Series([0.0, 0.0, 0.0], index=[index[0], index[1], index[4]])
    result = run_fixed_strategy_ensemble(bitcoin, defensive, bootstrap_samples=5, block_size=2)
    assert result.history.loc[pd.Timestamp("2024-06-01"), "turnover"] == 0.0
    assert result.history.loc[pd.Timestamp("2024-06-02"), "turnover"] == 0.0
    assert result.history.loc[pd.Timestamp("2024-06-03"), "turnover"] > 0.0


def test_published_ensemble_regression(published_result) -> None:
    evaluation = published_result.period_metrics.loc["evaluation"]
    assert np.isclose(evaluation["cagr"], 0.2939339399, atol=1e-9)
    assert evaluation["sharpe"] > 1.5
    assert evaluation["max_drawdown"] < 0.20
    assert published_result.component_correlation.iloc[0, 1] < 0.15
    assert published_result.bootstrap["probability_cagr_at_or_above_target"] > 0.80
    cost_200 = published_result.cost_sensitivity.loc[
        published_result.cost_sensitivity["cost_bps"] == 200.0, "evaluation_cagr"
    ].iloc[0]
    assert cost_200 > 0.28


def test_ensemble_is_not_twenty_percent_every_year(published_result) -> None:
    assert published_result.calendar_returns.loc[2022] < 0.0
    assert published_result.rolling_summary.loc[5, "minimum_cagr"] < 0.20
