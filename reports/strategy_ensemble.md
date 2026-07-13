# Fixed Cross-Asset Strategy Ensemble

Status: **exploratory historical evidence; frozen for prospective evaluation**.

Bitcoin snapshot: `5b8cc36ecc5ab4364d67586fd3b208403e1b6f2640d38ad824ac55446d83839d` through 2026-07-12.
Defensive snapshot: `75b02a7a51ce735ad39282dd8bfa610ab22a0093df1153c0a2fa1b98809ddd21` through 2026-07-13.

## Frozen Rule

Allocate 50% to `bitcoin-trend-v1` and 50% to `defensive-momentum-monthly-v1`. Rebalance on the first observed defensive-market session of each month and charge 10 bps times two-way ensemble turnover. Component returns already include their published trading costs, financing, and leverage drag.

No ensemble weight was selected on performance. The 50/50 allocation was fixed before this combined evaluation as the neutral diversification rule.

## Historical Results

| Period | CAGR | Sharpe | Max drawdown | Observations |
| --- | ---: | ---: | ---: | ---: |
| Evaluation (2021-present) | 29.39% | 1.55 | 19.41% | 2,019 |
| Full (2016-present) | 33.37% | 1.66 | 19.64% | 3,846 |

Evaluation component-return correlation is 0.118.

## Cost Sensitivity

| Ensemble turnover cost | Evaluation CAGR | Full CAGR |
| ---: | ---: | ---: |
| 0 bps | 29.44% | 33.42% |
| 10 bps | 29.39% | 33.37% |
| 25 bps | 29.32% | 33.28% |
| 50 bps | 29.19% | 33.14% |
| 100 bps | 28.94% | 32.85% |
| 200 bps | 28.45% | 32.28% |

## Allocation Diagnostics

These rows are sensitivity diagnostics, not a selection grid.

| Bitcoin weight | Defensive weight | Evaluation CAGR | Sharpe | Max drawdown |
| ---: | ---: | ---: | ---: | ---: |
| 25% | 75% | 33.04% | 1.55 | 19.05% |
| 50% | 50% | 29.39% | 1.55 | 19.41% |
| 75% | 25% | 25.24% | 1.29 | 20.97% |

## Path Robustness

The 30-day moving-block bootstrap gives a 82.66% probability of CAGR at or above 20%. Its 5th/median/95th percentiles are 14.10%, 28.42%, and 45.00%.

| Evaluation year | Return |
| ---: | ---: |
| 2021 | 31.88% |
| 2022 | -13.06% |
| 2023 | 49.62% |
| 2024 | 59.86% |
| 2025 | 41.35% |
| 2026 | 7.31% |

| Rolling horizon | Minimum CAGR | Median CAGR | Fraction >= 20% |
| ---: | ---: | ---: | ---: |
| 3 years | 16.47% | 33.30% | 95.78% |
| 5 years | 18.19% | 31.49% | 97.87% |

## Claim Boundary

The historical point estimate and most resampled paths clear 20%, but this is not untouched ensemble validation. Both component evaluation histories were visible before the combination was tested, and 2022 lost money. The ensemble is therefore frozen as a prospective candidate, not represented as verified future profitability or a 20% return every year.
