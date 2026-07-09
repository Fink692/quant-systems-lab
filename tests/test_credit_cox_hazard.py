import numpy as np
import pandas as pd
import pytest

from quantlab.credit.cox import fit_cox_ph
from quantlab.workflows.demo_suite import run_full_demo


def test_cox_ph_orders_low_and_high_credit_risk():
    covariates = pd.DataFrame(
        {
            "leverage": [0.30, 0.40, 0.55, 0.70, 0.85, 1.00],
            "spread": [0.006, 0.008, 0.012, 0.020, 0.032, 0.050],
        }
    )
    durations = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    observed = np.array([False, False, True, True, True, True])

    fit = fit_cox_ph(covariates, durations, observed, l2_penalty=0.2)
    low_high = pd.DataFrame({"leverage": [0.35, 1.05], "spread": [0.007, 0.055]})
    ratios = fit.hazard_ratio(low_high)
    survival = fit.survival_probability(low_high, maturity=3.0)

    assert fit.defaults == 4
    assert fit.baseline_cumulative_hazard.is_monotonic_increasing
    assert ratios.iloc[1] > ratios.iloc[0]
    assert survival.iloc[1] < survival.iloc[0]
    assert np.all((survival >= 0.0) & (survival <= 1.0))


def test_cox_ph_rejects_data_without_defaults():
    covariates = pd.DataFrame({"leverage": [0.3, 0.4], "spread": [0.006, 0.008]})

    with pytest.raises(ValueError, match="at least one default"):
        fit_cox_ph(covariates, np.array([1.0, 2.0]), np.array([False, False]))


def test_full_demo_exposes_cox_credit_metrics():
    result = run_full_demo(seed=32).as_dict()

    assert result["credit"]["cox_hazard_ratio_stressed"] > 0.0
    assert 0.0 <= result["credit"]["cox_three_year_default_probability"] <= 1.0
