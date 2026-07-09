import json

import numpy as np

from quantlab.cli import main
from quantlab.data.synthetic import (
    synthetic_cointegrated_prices,
    synthetic_credit_spreads,
    synthetic_exposure_network,
    synthetic_factor_panel,
    synthetic_option_chain,
)
from quantlab.workflows.demo_suite import run_full_demo


def test_synthetic_datasets_have_expected_shapes():
    chain = synthetic_option_chain(maturities=np.array([0.5, 1.0]), moneyness=np.array([0.9, 1.0, 1.1]))
    assert len(chain) == 12
    assert {"strike", "maturity", "option_type", "implied_volatility", "price"}.issubset(chain.columns)
    assert (chain["price"] >= 0.0).all()

    panel = synthetic_factor_panel(periods=40, assets=5, factors=2, seed=1)
    assert panel.asset_returns.shape == (40, 5)
    assert panel.factor_returns.shape == (40, 2)
    assert panel.true_exposures.shape == (5, 2)
    assert panel.prices.shape == (40, 5)

    stat_prices = synthetic_cointegrated_prices(periods=60, seed=2)
    assert list(stat_prices.columns) == ["PairA", "PairB", "Diversifier"]

    spreads = synthetic_credit_spreads()
    assert spreads["credit_spread"].is_monotonic_increasing

    exposures, capital = synthetic_exposure_network(institutions=4, seed=3)
    assert exposures.shape == (4, 4)
    assert capital.shape == (4,)


def test_full_demo_runs_all_project_families():
    result = run_full_demo(seed=5).as_dict()
    assert set(result) == {
        "options",
        "market_making",
        "rl_trading",
        "factor_risk",
        "portfolio",
        "rough_vol",
        "stat_arb",
        "credit",
        "surface_arbitrage",
        "systemic",
    }
    assert result["options"]["quotes"] > 0
    assert result["options"]["local_vol_mean"] > 0.0
    assert result["surface_arbitrage"]["maturities"] >= 2
    assert result["factor_risk"]["assets"] == 8
    assert result["factor_risk"]["total_variance"] > 0.0
    assert result["market_making"]["rows"] == 100
    assert result["portfolio"]["robust_first_weight"] >= 0.0
    assert result["rough_vol"]["rough_option_price"] > 0.0
    assert 0.0 <= result["credit"]["merton_default_probability"] <= 1.0
    assert result["systemic"]["clearing_defaults"] >= 0


def test_cli_price_option_and_demo_suite(capsys):
    exit_code = main(
        [
            "price-option",
            "--spot",
            "100",
            "--strike",
            "100",
            "--maturity",
            "1",
            "--rate",
            "0.03",
            "--volatility",
            "0.2",
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["price"] > 0.0

    exit_code = main(["demo-suite", "--seed", "5"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert "options" in payload
    assert "systemic" in payload


def test_cli_implied_vol_round_trip(capsys):
    main(
        [
            "implied-vol",
            "--price",
            "9.413403383853016",
            "--spot",
            "100",
            "--strike",
            "100",
            "--maturity",
            "1",
            "--rate",
            "0.03",
            "--option-type",
            "call",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert abs(payload["implied_volatility"] - 0.2) < 1e-8
