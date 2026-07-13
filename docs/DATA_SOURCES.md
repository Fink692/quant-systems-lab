# Data Sources

## Real Market Dataset

### LOBSTER AAPL Level-5 Demonstration Sample

- Instrument and venue: AAPL, NASDAQ.
- Session: 2012-06-21, regular market hours.
- Book representation: synchronized message and five-level order-book files.
- Official sample page: <https://data.lobsterdata.com/info/DataSamples.php>
- Output specification: <https://data.lobsterdata.com/info/DataStructure.php>
- Fetch script: `examples/fetch_lobster_sample.py`
- Local directory: `data/real/lobster_sample/` (ignored by Git).
- Canonical output: `data/processed/lobster_sample/` (ignored by Git).

Verified raw-file hashes:

| File | SHA-256 |
| --- | --- |
| `AAPL_2012-06-21_34200000_57600000_message_5.csv` | `8ca6fcbbf439c973d8ef74cb096a3aa340f66db816feea3ef09dcb7522bb625d` |
| `AAPL_2012-06-21_34200000_57600000_orderbook_5.csv` | `9a93bc2d754b5f7e02dd44cb112a19f86534212875e3b85b38ff99f6620b1f66` |

The sample contains 301,587 synchronized rows. Prices are stored by LOBSTER in units of $0.0001 and message times are seconds after exchange-local midnight. Ingestion converts the session from `America/New_York` to UTC nanoseconds and preserves the source row as the ordering key.

The dataset manifest fingerprint is `a985e2d3cb1e14abf10f9b8a60177d7fdcf7244574a3cc137c312e00cca1ecce`. The fingerprint covers provider, dataset identifier, source and license URLs, file sizes, and file hashes; acquisition timestamps and machine details do not change it.

Important limitations:

- The files are a public demonstration sample based on the official NASDAQ Historical TotalView-ITCH sample. They are used locally and are not redistributed in this repository.
- The sample is one session and cannot support a claim of persistent strategy profitability.
- Five visible levels are synchronized after each message, but complete depth and original queue history outside those levels are unavailable. Reconstruction mismatches at the Level-5 boundary are counted and reported.
- LOBSTER event time is available, but a distinct network receive timestamp is not. The canonical sample uses event time as receive time and treats latency as a sensitivity assumption.
- The illustrative fee schedule in the experiment configuration is not asserted to be the account-specific NASDAQ fee tier applicable to a particular 2012 participant.

The sample is appropriate for validating ingestion, reconstruction, chronology, replay, accounting, and reporting. A flagship paper still requires licensed multi-session L2 or L3 data and true receive-time semantics.

### Yahoo Finance Leveraged-ETF Daily Dataset

- Cached file: `data/real/leveraged_etf_adjusted.csv`
- Integrity metadata: `data/real/leveraged_etf_adjusted.metadata.json`
- Fetch script: `examples/fetch_leveraged_etf_data.py`
- Upstream endpoint: Yahoo Finance chart API, requested separately for TQQQ, QQQ, and BIL.
- Adjusted-close definition: <https://help.yahoo.com/kb/SLN28256.html>
- Historical-data help: <https://in.help.yahoo.com/kb/finance/download-historical-data-yahoo-finance-sln2311.html>
- TQQQ fund objective and leverage warning: <https://www.proshares.com/our-etfs/leveraged-and-inverse/tqqq>
- Local sample window: February 11, 2010 through July 10, 2026.

The study retains adjusted closes for TQQQ, QQQ, and BIL on their common trading dates. Yahoo documents adjusted close as accounting for applicable splits and dividend distributions. The fetch drops missing rows, sorts and deduplicates timestamps, excludes the current UTC date to avoid incomplete daily bars, and writes a SHA-256 checksum of the canonical LF-delimited CSV with the snapshot metadata.

Limitations:

- Yahoo Finance is a free public source, not a point-in-time institutional database or execution feed.
- Adjusted closes do not provide bid/ask quotes, intraday market impact, or the investor's actual fill price.
- TQQQ began in 2010, so its live history excludes the dot-com crash and 2008 financial crisis.
- TQQQ targets three times the Nasdaq-100's daily return before fees and expenses; multi-day returns can diverge substantially from three times the index return.
- Reproduction should retain the committed snapshot for exact results and use the fetch command only for a deliberately updated experiment.
- Users should review the upstream provider's current data terms before redistributing refreshed snapshots.

### QQQ/FRED Long-History Stress Inputs

- Cached file: `data/real/qqq_fred_stress_daily.csv`
- Integrity metadata: `data/real/qqq_fred_stress_daily.metadata.json`
- Fetch script: `examples/fetch_qqq_fred_stress_data.py`
- QQQ field: Yahoo Finance adjusted close.
- Financing field: [Federal Funds Effective Rate, FRED DFF](https://fred.stlouisfed.org/series/DFF), percent per annum.
- Local sample window: March 10, 1999 through July 10, 2026.

The fetch aligns forward-filled calendar-day DFF observations to completed QQQ trading sessions and writes a canonical LF-delimited CSV checksum. QQQ prices are observed; any 3x pre-TQQQ returns in the stress study are derived explicitly from QQQ returns, lagged DFF financing, and stated annual drag. They are never represented as observed ETF prices.

Limitations:

- The reconstruction cannot reproduce every swap, rebalance, tax, tracking, or market-price effect in TQQQ.
- Reconciliation against actual TQQQ finds 0.99894 daily-return correlation but a -2.38% annual actual-minus-synthetic mean, so the baseline reconstruction is optimistic.
- FRED DFF is used as a financing proxy and may not equal the fund's realized institutional financing rate.
- Yahoo data terms and the provider's current redistribution rules must be reviewed before refreshing or redistributing the snapshot.

### Forward Paper-Decision Snapshots

- Snapshot directory: `data/paper/`
- Ledger: `paper/leveraged_trend_decisions.jsonl`
- Recorder: `examples/record_leveraged_trend_paper.py`
- Signal source: Yahoo Finance adjusted closes.
- Independent close cross-check: Nasdaq.com ETF quote information.

Each dated snapshot contains exactly the data used for a prospective decision and is tied to the ledger by SHA-256. Nasdaq's completed 4:00 p.m. close is checked for TQQQ, QQQ, and BIL; a discrepancy above 5 bps prevents recording. The public Nasdaq website interface is not a contractual or licensed feed and may change, so provider failures are treated as failed recording runs rather than silently bypassed.

### Adjusted OHLC Execution Snapshot

- Cached file: `data/paper/execution_timing_ohlc_2026-07-13.csv`
- Integrity metadata: `data/paper/execution_timing_ohlc_2026-07-13.metadata.json`
- Fetch script: `examples/fetch_leveraged_etf_ohlc.py`
- Fields: adjusted TQQQ and BIL open and close, 4,128 common sessions through July 13, 2026.

Yahoo supplies raw open, raw close, and adjusted close. Adjusted open is calculated as `raw open * adjusted close / raw close`, applying the session's split/distribution adjustment factor consistently to both endpoints. The latest adjusted TQQQ close is reconciled within 5 bps to Nasdaq's explicitly completed close. These are daily endpoint proxies, not executable bid/ask quotes; the audit still models slippage rather than observing fills.

### Shiller/DataHub S&P 500 Monthly Dataset

- Cached file: `data/real/shiller_sp500_monthly.csv`
- Fetch script: `examples/fetch_shiller_sp500_data.py`
- Upstream URL: `https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv`
- Upstream repository: `https://github.com/datasets/s-and-p-500`
- Data provider notes: DataHub states the dataset is a tidied CSV version of Robert Shiller's S&P 500 dataset, with recent months extended from FRED S&P 500 data.
- Local sample window: January 1980 through September 2023.
- Provenance: the fetcher writes a sidecar manifest containing acquisition time, source URL, row range, and the cached CSV's SHA-256 hash.

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
- The source snapshot currently ends in September 2023. The manifest warns that it is not current or live; later third-party CAPE observations are not silently spliced into the published study.
- This dataset does not solve survivorship bias for single-stock equity strategies; it is intentionally used for an index-level valuation/regime study.

## Synthetic Datasets

The rest of the package still uses deterministic synthetic datasets for model correctness tests, including option chains, factor panels, credit curves, cointegrated assets, and exposure networks. Synthetic data remains useful for CI because it makes the tests reproducible without paid feeds or API keys.
