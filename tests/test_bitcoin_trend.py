from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.research.bitcoin_trend import (
    BitcoinTrendConfig,
    _run_candidate,
    load_bitcoin_coinbase_csv,
    run_bitcoin_trend_study,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/real/coinbase_btc_dff.csv"
METADATA = ROOT / "data/real/coinbase_btc_dff.metadata.json"


def test_snapshot_integrity_and_completed_session() -> None:
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == metadata["sha256"]
    data = load_bitcoin_coinbase_csv(DATA)
    assert len(data) == 4_011
    assert data.index[-1] == pd.Timestamp("2026-07-12")
    assert metadata["daily_return_correlation"] > 0.98


def test_next_bar_accounting_has_no_same_close_execution() -> None:
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    data = pd.DataFrame(
        {
            "open": [100, 100, 100, 100, 100, 110, 121, 121],
            "high": [101, 101, 101, 101, 111, 122, 122, 122],
            "low": [99, 99, 99, 99, 99, 109, 120, 120],
            "close": [100, 100, 100, 100, 110, 121, 121, 121],
            "volume": 1.0,
            "dff": 0.0,
        },
        index=index,
    )
    cfg = BitcoinTrendConfig(
        moving_average_days=(3,),
        volatility_days=(2,),
        target_volatilities=(1.0,),
        hysteresis_bands=(0.0,),
        transaction_cost_bps=0.0,
    )
    history = _run_candidate(data, cfg, 3, 2, 1.0, 0.0)
    assert history.loc[index[4], "new_exposure"] == 0.0
    assert history.loc[index[5], "new_exposure"] > 0.0
    expected = history.loc[index[5], "new_exposure"] * (121 / 110 - 1)
    assert np.isclose(history.loc[index[5], "strategy_return"], expected)


def test_published_result_and_falsification_regression() -> None:
    data = load_bitcoin_coinbase_csv(DATA)
    result = run_bitcoin_trend_study(data)
    assert int(result.selected["moving_average_days"]) == 100
    assert int(result.selected["volatility_days"]) == 63
    assert result.selected["target_volatility"] == 0.3
    assert result.selected["hysteresis_band"] == 0.02
    assert result.period_metrics.loc["evaluation", "cagr"] > 0.20
    assert result.period_metrics.loc["evaluation", "max_drawdown"] < 0.30
    assert result.parameter_breadth["both_at_or_above_target"] == 30
    cost_50 = result.cost_sensitivity.loc[result.cost_sensitivity["cost_bps"] == 50.0, "evaluation_cagr"].iloc[0]
    assert cost_50 < 0.20
    assert result.bootstrap["probability_cagr_at_or_above_target"] < 0.50


def test_evaluation_mutation_cannot_change_validation_selection() -> None:
    data = load_bitcoin_coinbase_csv(DATA)
    baseline = run_bitcoin_trend_study(data, bootstrap_samples=10)
    changed = data.copy()
    mask = changed.index >= pd.Timestamp("2021-01-01")
    changed.loc[mask, ["open", "high", "low", "close"]] *= np.linspace(1.0, 3.0, mask.sum())[:, None]
    mutated = run_bitcoin_trend_study(changed, bootstrap_samples=10)
    fields = ["moving_average_days", "volatility_days", "target_volatility", "hysteresis_band"]
    assert baseline.selected[fields].to_dict() == mutated.selected[fields].to_dict()
