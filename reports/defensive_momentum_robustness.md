# Frozen Monthly Defensive-Momentum Robustness

The candidate is fixed at 252-session momentum, 200-session trend, 25% volatility target, 1.5x cap, monthly rebalancing, and next-open execution.

Baseline full-sample CAGR is **20.73%**, but the 20% threshold is **not robust**.

## Consistency

- Calendar years at or above 20%: 13 of 22.
- Negative calendar years: 3.
- Rolling five-year windows at or above 20%: 40.39%.
- Worst rolling five-year CAGR: -2.19%.
- Median rolling five-year CAGR: 17.37%.

## Calendar Returns

| year | return |
| --- | --- |
| 2005 | 0.076543 |
| 2006 | 0.316332 |
| 2007 | 0.290413 |
| 2008 | 0.272886 |
| 2009 | 0.286588 |
| 2010 | 0.123141 |
| 2011 | 0.247618 |
| 2012 | 0.028177 |
| 2013 | 0.322309 |
| 2014 | 0.284454 |
| 2015 | -0.121141 |
| 2016 | -0.264398 |
| 2017 | 0.509205 |
| 2018 | 0.098908 |
| 2019 | 0.016628 |
| 2020 | 0.328926 |
| 2021 | 0.354336 |
| 2022 | -0.162316 |
| 2023 | 0.356826 |
| 2024 | 0.517505 |
| 2025 | 0.941535 |
| 2026 | 0.189655 |

## Cost Sensitivity

| cost_bps | cagr | sharpe | max_drawdown | observations |
| --- | --- | --- | --- | --- |
| 5.000000 | 0.211442 | 0.923039 | 0.452935 | 5414.000000 |
| 10.000000 | 0.207329 | 0.908748 | 0.460715 | 5414.000000 |
| 25.000000 | 0.195044 | 0.865511 | 0.483456 | 5414.000000 |
| 50.000000 | 0.174746 | 0.792423 | 0.519433 | 5414.000000 |
| 100.000000 | 0.134809 | 0.643906 | 0.584661 | 5414.000000 |

At 25 bps per unit of turnover, full-sample CAGR falls below 20%.

## Excess-Leverage Drag Sensitivity

| annual_excess_leverage_drag | cagr | sharpe | max_drawdown | observations |
| --- | --- | --- | --- | --- |
| 0.005000 | 0.209378 | 0.915845 | 0.458778 | 5414.000000 |
| 0.010000 | 0.207329 | 0.908748 | 0.460715 | 5414.000000 |
| 0.020000 | 0.203242 | 0.894553 | 0.464568 | 5414.000000 |
| 0.040000 | 0.195109 | 0.866162 | 0.472192 | 5414.000000 |

## Moving-Block Bootstrap

- Samples: 2000; block size: 21 sessions.
- 5th / median / 95th percentile CAGR: 11.72% / 20.91% / 31.28%.
- Probability CAGR is at least 20%: 55.70%.

## Monthly Grid Breadth

Within the original monthly subfamily, 19 of 72 configurations clear 20% in both evaluation and full periods.

## Interpretation

The observed 20.73% CAGR is reproducible, but it is marginal to costs, absent from most rolling five-year windows, and only slightly more likely than not under the block bootstrap. The candidate remains suitable for frozen paper observation, not for a verified 20% profitability claim.
