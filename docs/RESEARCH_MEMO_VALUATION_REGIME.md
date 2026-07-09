# Research Memo: S&P 500 Valuation-Regime Allocation

## Hypothesis

The strategy tests a simple economic hypothesis: long-run equity returns are partly regime-dependent on valuation. When Shiller CAPE/PE10 is low relative to its own historical training distribution, forward equity risk compensation should be more attractive. When PE10 is high, the strategy should reduce market exposure and preserve capital, especially after volatility scaling and drawdown controls.

This is not a short-horizon alpha model. It is a slow allocation model meant to answer whether valuation can improve risk-adjusted exposure to the S&P 500 after realistic frictions.

## Data

The study uses real monthly S&P 500, dividend, rate, and Shiller CAPE/PE10 data from the DataHub `s-and-p-500` package.

- Cached dataset: `data/real/shiller_sp500_monthly.csv`
- Source documentation: `docs/DATA_SOURCES.md`
- Local sample: 1980-01 through 2023-09
- Rows after cleaning: 525
- Out-of-sample strategy months: 324

The S&P 500 is an index proxy. A live implementation would need ETF/futures execution data, borrow/financing assumptions if leverage is used, and explicit handling of dividends, taxes, margin, and contract rolls.

## Method

Each walk-forward fold uses:

1. 120 months of training data to estimate PE10 percentile thresholds.
2. 36 months of validation data to choose threshold candidates by validation Sharpe.
3. 36 months of test data to run the chosen rule out of sample.
4. A 36-month step size before retraining.

The trading signal is lagged by one month. The strategy does not use the current month's PE10 to trade the current month's return.

Exposure rule:

- Cheap regime: target higher equity exposure.
- Neutral regime: target moderate equity exposure.
- Expensive regime: target reduced equity exposure.

Risk controls:

- Volatility targeting at 12% annualized volatility.
- Maximum leverage of 1.25x.
- Drawdown control that de-risks when the strategy drawdown exceeds 20%.
- Transaction costs and slippage deducted on turnover.

## Results

Generated report: `reports/valuation_regime_study.md`

Key out-of-sample metrics:

- Strategy CAGR: 7.50%
- Benchmark CAGR: 10.09%
- Strategy Sharpe: 0.96
- Max drawdown: 34.43%
- Beta to benchmark: 0.52
- Annual alpha estimate: 2.09%
- Hit rate: 73.15%
- Profit factor: 2.30
- Average monthly turnover: 6.94%
- Total modeled cost: 1.57%
- 95% monthly VaR: 3.15%
- 95% monthly CVaR: 5.60%

Interpretation:

The strategy does not beat buy-and-hold S&P 500 CAGR over this sample. It does, however, materially reduce beta and volatility while producing a positive alpha estimate and solid Sharpe after transaction costs. That makes it better framed as a risk-managed allocation overlay than a pure return-maximizing strategy.

## Robustness

Cost robustness remains positive across the tested transaction-cost grid:

- 0 bps cost: 7.54% CAGR
- 5 bps cost: 7.50% CAGR
- 10 bps cost: 7.46% CAGR
- 25 bps cost: 7.33% CAGR
- 50 bps cost: 7.11% CAGR

The strategy is not turnover-heavy, so costs do not destroy the edge in this monthly setup. That said, the economic edge is not huge; weaker valuation regimes or poor ETF/futures implementation could erase it.

## Failure Analysis

Known weaknesses:

- Buy-and-hold wins on CAGR in the tested sample.
- The strategy can be underexposed during expensive momentum markets.
- The worst 12-month strategy return is still deeply negative, around -31.4%.
- High-volatility months have slightly negative average strategy returns.
- Drawdown duration reaches 58 months, which is psychologically and institutionally difficult.
- PE10 is slow moving and can stay expensive for years.
- Monthly data hides intramonth execution risk, gap risk, and implementation slippage.
- The S&P 500 index is not directly tradable; real implementation requires ETF/futures assumptions.
- Recent upstream rows after the Shiller extension point have incomplete dividend/earnings fields.

This is a useful risk-managed allocation study, not a claim of a standalone hedge-fund-grade alpha.

## What Would Improve It

- Use SPY total-return data or futures data with explicit financing.
- Add VIX and realized-volatility regime variables.
- Test alternative valuation variables such as earnings yield, dividend yield, real rates, and credit spreads.
- Add purged/embargoed validation windows for higher-frequency variants.
- Compare against 60/40, volatility targeting, and trend-following benchmarks.
- Add bootstrap confidence intervals for alpha and Sharpe.
- Add parameter sensitivity over train/validation/test window lengths.
- Add a live paper-trading mode that updates monthly after the latest data release.

## Conclusion

The study shows a real, reproducible quant research workflow: source data, cleaning, walk-forward training/validation/test splits, no-lookahead execution, risk controls, transaction costs, tear sheet, robustness, stress tests, and honest failure analysis. The result is not overclaimed: the strategy improves risk-adjusted behavior and reduces beta, but does not dominate buy-and-hold on CAGR.
