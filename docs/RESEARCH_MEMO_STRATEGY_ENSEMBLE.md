# Fixed Bitcoin and Defensive Strategy Ensemble

## Objective

Test whether a neutral, fixed blend of two already-frozen and economically distinct strategies improves the consistency of the historical 20% CAGR evidence.

This is an ensemble robustness exercise, not an untouched holdout. The component evaluation histories were visible before the combination was tested.

## Frozen Components

### Bitcoin sleeve

- Candidate: 100-day trend, 63-day volatility, 30% volatility target, 2% hysteresis.
- Maximum exposure: 100%.
- Signal execution: next UTC daily candle.
- Internal cost: 25 bps times exposure turnover.
- Primary data: Coinbase Exchange BTC-USD completed candles.
- Snapshot SHA-256: `5b8cc36ecc5ab4364d67586fd3b208403e1b6f2640d38ad824ac55446d83839d`.

### Defensive sleeve

- Candidate: monthly QQQ/GLD/TLT momentum and trend filter.
- Momentum/trend/volatility lookbacks: 252/200/63 sessions.
- Target volatility and maximum leverage: 25% and 1.5x.
- Internal cost: 10 bps times turnover plus 1% annual excess-leverage drag.
- Data: Yahoo adjusted OHLC with official FRED DFF financing and Nasdaq close reconciliation.
- Snapshot SHA-256: `75b02a7a51ce735ad39282dd8bfa610ab22a0093df1153c0a2fa1b98809ddd21`.

## Ensemble Rule

The frozen strategy ID is `btc-defensive-equal-weight-monthly-v1`.

- Allocate 50% of capital to each component.
- Do not optimize the 50/50 weight.
- Let sleeve weights drift between rebalances.
- Rebalance on the first observed defensive-market session of each month.
- Charge 10 bps times two-way ensemble turnover in addition to component costs.
- Treat missing defensive-session returns on weekends and holidays as zero marking returns; the next observed defensive return contains the multi-day close-to-close move.

The 25/75 and 75/25 rows are disclosed only as allocation sensitivity. They do not alter the frozen equal-weight rule.

## Results

| Period | CAGR | Sharpe | Maximum drawdown |
| --- | ---: | ---: | ---: |
| 2021-July 12, 2026 evaluation | 29.39% | 1.55 | 19.41% |
| 2016-July 12, 2026 full | 33.37% | 1.66 | 19.64% |

Evaluation daily component-return correlation is 0.118. The low correlation is the primary reason the blend has lower drawdown and stronger path consistency than either sleeve alone.

## Robustness and Falsification

- At 200 bps per unit of ensemble turnover, evaluation CAGR remains 28.45%.
- In 5,000 moving-block bootstrap samples, 82.66% clear a 20% CAGR.
- The bootstrap 5th percentile is 14.10%, so adverse paths still miss the objective materially.
- The minimum rolling three-year CAGR is 16.47%.
- The minimum rolling five-year CAGR is 18.19%.
- Calendar 2022 loses 13.06%. The strategy does not earn 20% every year.
- The 25%, 50%, and 75% Bitcoin allocation diagnostics all clear 25% evaluation CAGR, but these are observed-history diagnostics rather than independent confirmations.

## Claim Boundary

The ensemble strengthens the historical case but does not verify future profitability. Combining already-observed evaluation histories can capitalize on known diversification and regime behavior even when the weight is not numerically optimized. The only valid next promotion is prospective evidence generated after the rule was frozen.

Production implementation also needs executable quotes, venue and broker fees, position sizing against actual liquidity, tax and custody treatment, operational controls, and independent review. Historical daily endpoints cannot prove those properties.

## Reproduction

```powershell
python examples/run_strategy_ensemble.py --bitcoin-data data/real/coinbase_btc_dff.csv --bitcoin-metadata data/real/coinbase_btc_dff.metadata.json --defensive-data data/real/defensive_momentum_ohlc.csv --defensive-metadata data/real/defensive_momentum_ohlc.metadata.json --config config/strategy_ensemble.json --output reports/strategy_ensemble.md
```
