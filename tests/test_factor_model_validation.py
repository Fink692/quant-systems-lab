import numpy as np
import pandas as pd

from quantlab.risk.model_validation import rolling_factor_model_validation
from quantlab.workflows.demo_suite import run_full_demo


def test_rolling_factor_model_validation_scores_predictive_fit():
    rng = np.random.default_rng(31)
    dates = pd.date_range("2021-01-01", periods=140, freq="B")
    factors = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(140, 3)),
        index=dates,
        columns=["market", "size", "value"],
    )
    betas = pd.DataFrame(
        {
            "market": [1.1, 0.8, 1.3, 0.6, 1.0],
            "size": [0.3, -0.2, 0.1, 0.5, -0.1],
            "value": [-0.1, 0.4, 0.2, -0.3, 0.1],
        },
        index=["A", "B", "C", "D", "E"],
    )
    assets = factors @ betas.T + rng.normal(0.0, 0.001, size=(140, 5))

    result = rolling_factor_model_validation(assets, factors, train_window=70, test_window=20, step_size=20)

    assert len(result.report) == 3
    assert result.mean_oos_r_squared > 0.85
    assert 0.0 <= result.mean_abs_residual_correlation <= 1.0
    assert 0.0 < result.mean_specific_risk_share < 0.10
    assert np.isfinite(result.max_covariance_condition_number)


def test_full_demo_exposes_factor_model_validation_metrics():
    result = run_full_demo(seed=41).as_dict()

    assert result["factor_risk"]["factor_oos_r_squared"] > 0.0
    assert 0.0 <= result["factor_risk"]["factor_residual_correlation"] <= 1.0
    assert 0.0 <= result["factor_risk"]["factor_specific_risk_share"] <= 1.0
    assert result["factor_risk"]["factor_covariance_condition"] >= 1.0
