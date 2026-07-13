# Defensive Multi-Asset Momentum Exploration

## Question

Can rotation among technology equities, gold, and long-duration Treasuries produce a more persistent 20% return than the single-asset leveraged-trend model while retaining explicit next-open execution?

## Data and Design

- Assets: QQQ, GLD, and TLT adjusted open and close from Yahoo Finance.
- Financing: official Federal Funds Effective Rate from the committed FRED DFF snapshot.
- Independent check: July 13, 2026 completed closes from Nasdaq.com, with all differences below 5 bps.
- Common sample: 5,444 sessions from November 18, 2004 through July 13, 2026.
- Development: January 2005 through December 2016.
- Evaluation: January 2017 through July 13, 2026.

The 144-row grid combines 63/126/252-session momentum, 100/200-session trend filters, 25%/30%/40%/50% volatility targets, 1.5x/2x/3x caps, and weekly/monthly rebalancing. The highest-momentum eligible asset receives the volatility-scaled allocation; otherwise the strategy holds the financing proxy. Selection maximizes development Sharpe and then development CAGR.

Signals use completed period-end closes. The old allocation earns the next overnight move and the new allocation begins at the following open. Returns include 10 bps times gross weight turnover, lagged DFF financing, and 1% annual drag on exposure above 1x.

## Result

The broad-grid selector chooses 63-session momentum, a 200-session trend filter, 25% target volatility, a 1.5x cap, and weekly rebalancing. It produces:

- Development CAGR: **14.79%**, Sharpe 0.73, maximum drawdown 40.62%.
- Evaluation CAGR: **8.06%**, Sharpe 0.44, maximum drawdown 41.60%.
- Full-sample CAGR: **11.76%**.

The 20% evaluation threshold is **not met**. At 25 bps per unit of turnover, full-sample CAGR falls to 8.02%.

## Exploratory Monthly Lead

The best monthly-only row uses 252-session momentum, a 200-session trend filter, a 25% target, and a 1.5x cap. It records 30.01% evaluation CAGR and 20.73% full-sample CAGR. This is not the broad-grid winner. Frequency sensitivity was discovered after evaluation results were inspected, so selecting the monthly row now would contaminate the historical evaluation.

The monthly configuration is a research lead, not verified 20% profitability. Its honest evidence gate is a separately versioned prospective decision ledger with no parameter changes after future returns arrive.

## Frozen-Lead Robustness

The frozen monthly row was subsequently stressed without changing its parameters:

- Full 2005–2026 CAGR: 20.73% at the baseline 10 bps turnover cost.
- Full CAGR at 25 bps turnover cost: 19.50%.
- Rolling five-year windows at or above 20% CAGR: 40.39%.
- Worst rolling five-year CAGR: -2.19%; median: 17.37%.
- Calendar years at or above 20%: 13 of 22; negative years: 3.
- Moving-block bootstrap probability of at least 20% CAGR: 55.70%.
- Original monthly subfamily rows clearing 20% in both evaluation and full periods: 19 of 72.

The point estimate is real, but the threshold is marginal rather than persistent. See `reports/defensive_momentum_robustness.md`.

## Conclusion

Defensive assets do not make the broad momentum family robust to selection and execution assumptions. The experiment adds a plausible lower-leverage candidate for future testing, but it does not complete the 20% objective.
