import json

import numpy as np

from quantlab.cli import main
from quantlab.risk.backtesting import basel_traffic_light, christoffersen_var_backtest, kupiec_var_backtest
from quantlab.workflows.demo_suite import run_full_demo


def test_christoffersen_backtest_counts_exception_transitions():
    returns = np.array([0.01, 0.01, -0.05, -0.04, 0.01, -0.06, 0.02, 0.01])
    var_estimates = np.full_like(returns, 0.03)
    result = christoffersen_var_backtest(returns, var_estimates, confidence=0.95)

    assert result.exceptions == 3
    assert result.transition_counts == {"n00": 2, "n01": 2, "n10": 2, "n11": 1}
    assert result.independence_statistic >= 0.0
    assert 0.0 <= result.independence_p_value <= 1.0
    assert 0.0 <= result.conditional_coverage_p_value <= 1.0


def test_kupiec_and_traffic_light_diagnostics_are_bounded():
    returns = np.r_[np.full(245, 0.001), np.full(5, -0.05)]
    var_estimates = np.full_like(returns, 0.02)
    kupiec = kupiec_var_backtest(returns, var_estimates, confidence=0.99)
    green = basel_traffic_light(kupiec.exceptions, observations=kupiec.observations, confidence=0.99)
    red = basel_traffic_light(25, observations=250, confidence=0.99)

    assert kupiec.exceptions == 5
    assert 0.0 <= kupiec.kupiec_p_value <= 1.0
    assert green.zone in {"green", "yellow"}
    assert red.zone == "red"


def test_risk_demo_cli_exposes_model_validation_metrics(capsys):
    assert main(["risk-demo", "--seed", "8"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "kupiec_p_value" in payload
    assert "christoffersen_p_value" in payload
    assert payload["traffic_light_zone"] in {"green", "yellow", "red"}


def test_full_demo_exposes_christoffersen_and_traffic_light_metrics():
    result = run_full_demo(seed=24).as_dict()
    assert 0.0 <= result["portfolio"]["christoffersen_p_value"] <= 1.0
    assert isinstance(result["portfolio"]["var_traffic_light_green"], bool)
