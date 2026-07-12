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
