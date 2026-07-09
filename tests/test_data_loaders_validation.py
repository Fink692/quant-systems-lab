import json

import numpy as np
import pandas as pd
import pytest

from quantlab.cli import main
from quantlab.data.loaders import (
    load_credit_spread_curve_csv,
    load_option_chain_csv,
    load_price_panel_csv,
    returns_from_prices,
    validate_credit_spread_curve,
    validate_option_chain,
    validate_price_panel,
)
from quantlab.data.synthetic import synthetic_credit_spreads, synthetic_factor_panel, synthetic_option_chain
from quantlab.workflows.demo_suite import run_full_demo


def test_option_chain_csv_loader_and_validation(tmp_path):
    chain = synthetic_option_chain(maturities=np.array([0.5]), moneyness=np.array([0.95, 1.0, 1.05]))
    path = tmp_path / "options.csv"
    chain.to_csv(path, index=False)
    loaded = load_option_chain_csv(path)
    validation = validate_option_chain(loaded)
    assert validation.is_valid
    assert validation.rows == len(chain)

    broken = chain.drop(columns=["price"])
    assert not validate_option_chain(broken).is_valid


def test_price_panel_loader_and_return_construction(tmp_path):
    panel = synthetic_factor_panel(periods=6, assets=3, factors=2, seed=33)
    frame = panel.prices.reset_index(names="date")
    path = tmp_path / "prices.csv"
    frame.to_csv(path, index=False)
    loaded = load_price_panel_csv(path, date_column="date")
    simple = returns_from_prices(loaded, method="simple")
    log_returns = returns_from_prices(loaded, method="log")
    assert validate_price_panel(loaded).is_valid
    assert simple.shape == (5, 3)
    assert log_returns.shape == (5, 3)
    assert np.isfinite(log_returns.to_numpy()).all()

    bad = loaded.copy()
    bad.iloc[0, 0] = -1.0
    with pytest.raises(ValueError):
        returns_from_prices(bad)


def test_credit_spread_curve_loader_and_validation(tmp_path):
    curve = synthetic_credit_spreads()
    path = tmp_path / "spreads.csv"
    curve.to_csv(path, index=False)
    loaded = load_credit_spread_curve_csv(path)
    assert validate_credit_spread_curve(loaded).is_valid

    unsorted = loaded.iloc[::-1].reset_index(drop=True)
    result = validate_credit_spread_curve(unsorted)
    assert not result.is_valid
    assert any("strictly increasing" in issue for issue in result.issues)


def test_data_demo_cli_and_full_demo_metrics(capsys):
    assert main(["data-demo", "--seed", "10"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["option_chain_valid"]
    assert payload["price_panel_valid"]
    assert payload["credit_spread_curve_valid"]
    assert payload["price_return_rows"] == 251

    result = run_full_demo(seed=10).as_dict()
    assert result["options"]["option_chain_valid"]
    assert result["factor_risk"]["price_panel_valid"]
    assert result["factor_risk"]["price_panel_return_rows"] == 251
    assert result["credit"]["spread_curve_valid"]
