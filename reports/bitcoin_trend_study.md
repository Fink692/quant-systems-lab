# Bitcoin Trend Study

Data: Coinbase BTC-USD completed daily candles through 2026-07-12; official FRED DFF financing.
Snapshot SHA-256: `5b8cc36ecc5ab4364d67586fd3b208403e1b6f2640d38ad824ac55446d83839d`.

## Prespecified Design

The 64-row grid uses 2016-2017 training, 2018-2020 validation-only selection, and a one-time 2021-present evaluation. A completed close changes the state; exposure begins on the next UTC candle. Costs are 25 bps times exposure turnover.

## Selected Candidate

Validation selected MA 100, volatility lookback 63, target volatility 30%, and hysteresis 2%.

| Period | CAGR | Sharpe | Max drawdown | Observations |
| --- | ---: | ---: | ---: | ---: |
| Train | 113.15% | 2.44 | 18.43% | 731 |
| Validation | 40.69% | 1.60 | 25.36% | 1,096 |
| Evaluation | 20.64% | 0.94 | 26.03% | 2,019 |
| Full | 40.44% | 1.48 | 27.86% | 3,846 |

The evaluation point estimate MET the 20% historical CAGR threshold. Full-history CAGR is 40.44%.

## Falsification Checks

| Cost per unit turnover | Evaluation CAGR | Full CAGR |
| ---: | ---: | ---: |
| 10 bps | 21.53% | 41.40% |
| 25 bps | 20.64% | 40.44% |
| 50 bps | 19.18% | 38.87% |
| 100 bps | 16.29% | 35.76% |
| 200 bps | 10.71% | 29.72% |

30 of 64 configurations clear 20% in both evaluation and full history.
The 30-day moving-block bootstrap gives a 44.65% probability of CAGR at or above 20%; its 5th/median/95th percentiles are 1.07%, 18.29%, and 42.25%.

| Evaluation year | Return |
| ---: | ---: |
| 2021 | 26.25% |
| 2022 | -10.31% |
| 2023 | 61.24% |
| 2024 | 64.85% |
| 2025 | 0.06% |
| 2026 | -6.26% |

| Rolling horizon | Minimum CAGR | Median CAGR | Fraction >= 20% |
| ---: | ---: | ---: | ---: |
| 3 years | 16.80% | 38.56% | 97.24% |
| 5 years | 17.54% | 42.33% | 92.98% |

## Conclusion

Robust/prospective 20% claim: **NOT SUPPORTED**. The historical evaluation CAGR narrowly clears the threshold, but 50 bps costs push it below 20%, bootstrap support is below 50%, and individual years include losses. This is a reproducible historical result, not verified future profitability or a guaranteed annual return.

Coinbase daily bars meet at the midnight UTC boundary and do not contain executable bid/ask fills. The turnover charge is a sensitivity assumption; live slippage, fees, taxes, outages, and custody risks can be materially worse.
