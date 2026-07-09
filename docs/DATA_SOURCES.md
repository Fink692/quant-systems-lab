# Data Sources

## Real Market Dataset

### Shiller/DataHub S&P 500 Monthly Dataset

- Cached file: `data/real/shiller_sp500_monthly.csv`
- Fetch script: `examples/fetch_shiller_sp500_data.py`
- Upstream URL: `https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv`
- Upstream repository: `https://github.com/datasets/s-and-p-500`
- Data provider notes: DataHub states the dataset is a tidied CSV version of Robert Shiller's S&P 500 dataset, with recent months extended from FRED S&P 500 data.
- Local sample window: January 1980 through September 2023.

Fields retained:

- `sp500`: monthly S&P 500 index level.
- `dividend`: Shiller monthly dividend series.
- `earnings`: Shiller earnings series.
- `cpi`: consumer price index.
- `long_rate`: long-term interest rate.
- `real_price`: inflation-adjusted price.
- `pe10`: Shiller CAPE/PE10 valuation measure.

Cleaning process:

1. Download upstream CSV.
2. Keep the listed market and valuation fields.
3. Parse dates and numeric columns.
4. Drop rows missing `sp500`, `pe10`, or `long_rate`.
5. Remove non-positive `sp500` and `pe10` rows.
6. Restrict to rows from 1980 onward so the strategy uses modern monetary/market regimes and avoids early PE10 warm-up zeros.

Limitations:

- The dataset is monthly, so it is suitable for allocation/regime research, not high-frequency execution research.
- The S&P 500 series is an index-level proxy, not directly tradable without ETF/futures implementation assumptions.
- Recent rows in the upstream file have zero dividend/earnings fields after the Shiller extension point; the strategy uses PE10 and price/rate fields, and the memo calls out this limitation.
- This dataset does not solve survivorship bias for single-stock equity strategies; it is intentionally used for an index-level valuation/regime study.

## Synthetic Datasets

The rest of the package still uses deterministic synthetic datasets for model correctness tests, including option chains, factor panels, credit curves, cointegrated assets, and exposure networks. Synthetic data remains useful for CI because it makes the tests reproducible without paid feeds or API keys.
