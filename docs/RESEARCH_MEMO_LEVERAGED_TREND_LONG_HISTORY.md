# Leveraged Trend Long-History Falsification

## Question

The observed TQQQ study exceeded 20% CAGR on its January 2021 through July 2026 holdout. Does the same frozen 200-day trend and 30% volatility-targeting rule survive the dot-com crash and global financial crisis?

TQQQ did not exist before 2010, so this question cannot be answered with observed TQQQ prices. This extension uses real QQQ adjusted closes and real Federal Reserve rates, then labels every pre-inception 3x return as a reconstruction rather than market data.

## Inputs and Formula

- QQQ adjusted closes: Yahoo Finance, March 10, 1999 through July 10, 2026.
- Financing rate: Federal Funds Effective Rate (FRED series DFF), lagged one session.
- Source snapshot: `data/real/qqq_fred_stress_daily.csv` with a canonical CSV SHA-256 sidecar.
- Frozen strategy: 200-day trend filter, trailing 21-session volatility, 30% annual target, maximum 100% allocation to the reconstructed 3x sleeve, and one-session execution lag.
- Trading friction: 5 bps transaction cost plus 5 bps slippage on exposure turnover.

The reconstructed daily risky-sleeve return is:

```text
r_3x,t = 3 * r_QQQ,t - 2 * r_cash,t - 0.95% / 252
```

This models three units of QQQ exposure, financing on the two additional units, and annual fund drag. It is a stress proxy, not a claim about a tradeable pre-2010 ETF.

## Reconstruction Check

The reconstructed return is compared with actual TQQQ adjusted returns over 4,126 overlapping sessions from February 2010 onward:

- Daily-return correlation: 0.99894.
- Actual minus reconstructed annual mean return: -2.38%.
- Annualized tracking error: 2.95%.

The high correlation supports the proxy's day-to-day shape. The negative mean residual shows that the baseline reconstruction is optimistic by about 2.38% annually relative to actual TQQQ. Therefore, it should not be used to rescue a marginal performance result.

## Results

After modeled costs:

- Dot-com plus GFC decade, 2000-2009: **3.96% CAGR**, 39.06% maximum drawdown.
- Pre-TQQQ period through February 10, 2010: **2.70% CAGR**.
- February 2010 through 2020: **22.59% CAGR**.
- Published 2021-July 2026 holdout under the proxy: **25.42% CAGR**.
- Full January 2000-July 2026 history: **15.13% CAGR**, 0.67 Sharpe, 39.06% maximum drawdown.

The full-history 20% CAGR threshold is **not met**. Increasing assumed annual drag to 4% lowers full-history CAGR to 13.96% and raises maximum drawdown above 50%.

## Attempted Short Extension

A long/short extension was also tested using the same QQQ and rate inputs. A validation-selected 150-day rule with up to 3x long exposure and 0.5x short exposure earned 34.52% on the 2021-July 2026 holdout, but lost 6.65% annually in 2000-2009, produced an 82.95% drawdown, and compounded at only 13.48% over the full history. It is rejected.

## Interpretation

The recent TQQQ holdout is not fabricated, but it is regime-dependent. The long-history failure contradicts any claim that this model has established a persistent 20% annual return process. The result also illustrates why synthetic pre-inception histories must be reconciled to live instruments and clearly labeled.

## Next Evidence Gate

The model parameters should remain frozen while paper-traded forward. A future claim requires independent prices, timestamped decisions, recorded executable quotes, realized slippage, and enough later observations to separate a durable edge from the post-2010 technology regime.
