# Leveraged Nasdaq Trend: Validation and Holdout Study

## Research Question

Can a slow trend filter and volatility-scaled exposure to a liquid leveraged Nasdaq-100 ETF produce at least 20% compounded annual growth on a genuinely later historical holdout, after explicit turnover costs?

The target is a historical research hurdle, not a promise of future returns. No backtest can prove a strategy will earn 20% in each future year.

## Economic Hypothesis

[ProShares states](https://www.proshares.com/our-etfs/leveraged-and-inverse/tqqq) that TQQQ targets three times the **daily** return of the Nasdaq-100 before fees and expenses. Persistent equity trends can reward leveraged exposure, but leverage also magnifies drawdowns, volatility drag, and path dependence. A slow moving-average filter attempts to participate in established positive trends and move toward Treasury-bill exposure in sustained declines. A 21-day realized-volatility scaler limits exposure when the ETF becomes unusually volatile.

The model uses the actual ETF history rather than synthesizing 3x returns, so the observed prices already reflect the fund's realized path dependence, financing implementation, expense drag, distributions, and corporate actions.

## Data and Integrity Controls

- Instruments: TQQQ as the risky asset, QQQ as the unlevered benchmark, and BIL as the cash proxy.
- Source: Yahoo Finance daily adjusted-close history, cached in `data/real/leveraged_etf_adjusted.csv`.
- Snapshot: 4,127 common sessions from February 11, 2010 through July 10, 2026.
- Integrity: the fetch creates `data/real/leveraged_etf_adjusted.metadata.json` with row count, dates, retrieval time, and SHA-256 checksum.
- Completion rule: the fetch discards the current UTC date because the endpoint can expose a still-open session as a daily bar.
- Adjustments: [Yahoo defines adjusted close](https://help.yahoo.com/kb/SLN28256.html) as incorporating applicable splits and dividend distributions.

The study does not use fabricated prices. It remains limited by a free public data source and should be independently replicated against a licensed institutional feed before capital is committed.

## Experiment Design

The parameter family was evaluated with fixed chronological boundaries:

- Training/context: inception through December 31, 2016.
- Validation/selection: January 1, 2017 through December 31, 2020.
- Untouched holdout: January 1, 2021 through July 10, 2026.

The validation grid contains 48 combinations:

- Moving average: 100, 150, 200, or 250 sessions.
- Hysteresis band: 0%, 1%, or 2% around the moving average.
- Annualized volatility target: 30%, 40%, 50%, or 100%, capped at 100% ETF exposure.

Selection maximizes validation Sharpe, then validation CAGR. The winning configuration is a 200-day moving average, no band, and a 30% volatility target.

At each close, the model computes the trend state and trailing 21-session realized volatility. The resulting desired exposure is shifted by one full session before earning returns. Remaining capital earns BIL returns. Turnover is charged 5 bps transaction cost plus 5 bps slippage. Shorting and exposure above 100% of TQQQ are prohibited.

## Holdout Results

From January 2021 through July 10, 2026, after the baseline 10 bps turnover charge:

- Strategy CAGR: **23.29%**.
- TQQQ buy-and-hold CAGR: 25.97%.
- QQQ buy-and-hold CAGR: 17.17%.
- Annualized volatility: 27.62%.
- Sharpe: 0.90.
- Sortino: 1.03.
- Maximum drawdown: 24.22%.
- Calmar: 0.96.
- Average TQQQ exposure: 46.37%.
- Annual turnover: 6.22 times capital.
- Beta to QQQ: 0.79.

The strategy clears the 20% holdout CAGR hurdle and substantially reduces drawdown relative to TQQQ buy-and-hold. It does not beat TQQQ buy-and-hold CAGR in this sample.

Calendar returns were uneven: 42.25% in 2021, -7.87% in 2022, 44.94% in 2023, 27.16% in 2024, 15.94% in 2025, and 12.88% for partial-year 2026. Therefore, “23.29% CAGR” must not be restated as “at least 20% every year.”

## Robustness and Uncertainty

- 40 of 48 prespecified parameter combinations exceeded 20% holdout CAGR.
- Selected-model CAGR is 24.06% at zero costs, 23.29% at 10 bps, 22.15% at 25 bps, 20.27% at 50 bps, and 16.58% at 100 bps.
- A deterministic 2,000-sample moving-block bootstrap with 21-session blocks estimates a 5th/median/95th percentile CAGR of 2.79% / 22.81% / 46.29%.
- Only 56.65% of bootstrap samples exceed the 20% target.

The parameter surface and cost checks support the existence of a broad historical result. The bootstrap shows that the precision of the 20% estimate is weak. Both facts matter.

## Failed Candidate

Before this daily model, a monthly rotation strategy across TQQQ, UPRO, TMF, UGL, and BIL was selected on 2017-2020 data. It achieved 20.48% training CAGR and 29.41% validation CAGR but only 15.38% on the 2021-2026 holdout. Monthly rebalancing reacted too slowly during the 2022 leveraged-asset drawdown. That failure motivated, but did not alter the holdout results of, the simpler daily trend family.

## Failure Modes

- TQQQ targets 3x **daily**, not long-horizon, Nasdaq-100 returns. Volatility drag can be severe.
- The 200-day filter can exit after a sharp loss and re-enter after a rebound, creating whipsaw.
- The holdout spans only about 5.5 years and is dominated by a strong technology-equity regime.
- TQQQ did not exist through the dot-com crash or 2008 crisis, so the study cannot observe those live-fund paths.
- The strategy lost money in 2022 and remained below a prior equity peak for as long as 379 sessions.
- Turnover is high. At 100 bps per exposure change, the historical CAGR falls below the target.
- Taxes, market impact, account restrictions, tracking error at the investor's execution price, and BIL bid/ask spreads are not modeled separately.
- Parameter-family selection and the decision to research leveraged trend following create selection risk that a bootstrap of realized returns cannot capture.

## Conclusion

The repository now contains reproducible evidence of a real-data strategy that exceeded a 20% historical holdout CAGR after modeled costs. It does not contain evidence that 20% annual returns are guaranteed, probable in every year, or safe. The honest next validation is forward paper trading with frozen parameters and independent price data.
