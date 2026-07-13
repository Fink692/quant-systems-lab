# Bitcoin Trend Holdout Study

## Research Question

Can a simple, prespecified Bitcoin trend and volatility-targeting family clear a 20% annualized return threshold on untouched recent history after explicit turnover costs?

The answer is narrow: the selected candidate clears 20% historically, but the evidence does not support a robust or prospective 20% claim.

## Data and Provenance

The committed snapshot contains 4,011 completed Coinbase Exchange `BTC-USD` UTC daily candles from July 20, 2015 through July 12, 2026. It excludes the still-forming July 13 candle. Cash financing uses the official FRED effective federal funds rate (`DFF`), lagged before use.

An independent Yahoo Finance comparison covers 4,010 overlapping daily returns. Coinbase and Yahoo returns correlate at 0.98984; Coinbase-minus-Yahoo annual mean return is 0.50%, with 9.53% annualized tracking error. The exact CSV SHA-256 is `5b8cc36ecc5ab4364d67586fd3b208403e1b6f2640d38ad824ac55446d83839d`.

## Experimental Design

- Training: 2016-2017.
- Validation-only model selection: 2018-2020.
- One-time evaluation: January 2021 through July 12, 2026.
- Grid: 100/150/200/250-day moving average, 21/63-day volatility, 30%/50%/75%/100% target volatility, and 0%/2% hysteresis.
- Exposure is capped at 100%; no leverage is used.
- A completed close sets desired exposure, applied beginning with the next UTC candle.
- The old exposure earns `open / prior close`; the new exposure earns `close / open`.
- Cash earns lagged DFF, split two-thirds overnight and one-third intraday.
- Baseline cost is 25 bps times absolute exposure turnover.

Validation Sharpe, then validation CAGR, selects the candidate. Evaluation data never participates in selection. A mutation regression test changes all post-2020 prices and confirms the selected parameters remain unchanged.

## Results

Validation selected the 100-day moving average, 63-day volatility, 30% target volatility, and 2% hysteresis candidate.

| Period | CAGR | Sharpe | Maximum drawdown |
| --- | ---: | ---: | ---: |
| Training | 113.15% | 2.44 | 18.43% |
| Validation | 40.69% | 1.60 | 25.36% |
| Evaluation | 20.64% | 0.94 | 26.03% |
| Full 2016-present | 40.44% | 1.48 | 27.86% |

The evaluation point estimate clears the historical 20% threshold. Thirty of 64 configurations clear it in both evaluation and full history, so the result is not isolated to one exact parameter row.

## Falsification

The stronger claim fails:

- At 50 bps per unit of turnover, evaluation CAGR falls to 19.18%.
- A 2,000-sample, 30-day moving-block bootstrap estimates only a 44.65% probability of CAGR at or above 20%.
- The bootstrap 5th percentile is 1.07%, and its median is 18.29%.
- Evaluation includes -10.31% in 2022, approximately flat 2025, and -6.26% in partial 2026. It does not earn 20% each year.
- The minimum rolling three- and five-year CAGRs are 16.80% and 17.54%.

## Interpretation

This is the strongest distinct historical candidate found so far: it uses primary venue candles, an independent source reconciliation, chronological selection, next-bar exposure, and broad parameter support. It still does not verify future profitability. The 20.64% estimate has only 64 bps of margin over the objective at baseline costs, and plausible cost or path variation removes that margin.

Coinbase daily candle boundaries are midnight UTC marks in a continuous market, not executable quoted fills. The study does not observe bid/ask spread, order size impact, exchange tier fees, taxes, outages, or custody losses. Those omissions matter more for a marginal threshold result.

## Reproduction

```powershell
python examples/fetch_bitcoin_trend_data.py --output data/real/coinbase_btc_dff.csv --metadata data/real/coinbase_btc_dff.metadata.json
python examples/run_bitcoin_trend_study.py --data data/real/coinbase_btc_dff.csv --metadata data/real/coinbase_btc_dff.metadata.json --config config/bitcoin_trend.json --output reports/bitcoin_trend_study.md
```

Use the committed snapshot for the exact published result. Fetching again creates a new experiment with a later completed UTC candle.
