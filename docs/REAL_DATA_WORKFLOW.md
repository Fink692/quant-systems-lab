# Real-Data-Compatible Price Panel Workflow

Quant Systems Lab is built around deterministic synthetic datasets so tests can run anywhere. For a real-data workflow, the simplest path is to export daily adjusted closes from your data vendor into a wide CSV and feed it into the package validators and portfolio/risk tools.

## Expected CSV Shape

```csv
date,SPY,QQQ,IWM,TLT,GLD
2024-01-02,472.65,402.17,198.83,99.21,191.44
2024-01-03,468.79,398.26,196.74,100.05,192.31
```

Rules:

- One date column.
- One strictly positive numeric price column per asset.
- No missing values after your chosen cleaning step.
- Prefer adjusted close prices when using equities or ETFs.
- Use at least several months of data for meaningful portfolio/risk estimates.

The committed file [price_panel_template.csv](https://github.com/Fink692/quant-systems-lab/blob/main/examples/price_panel_template.csv) is a small schema fixture, not a live market-data source.

## Run The Example

```powershell
$env:PYTHONPATH='src'
python examples\run_price_panel_example.py --prices examples\price_panel_template.csv
```

With a real exported CSV:

```powershell
$env:PYTHONPATH='src'
python examples\run_price_panel_example.py --prices C:\path\to\adjusted_closes.csv --date-column date
```

## What The Script Computes

- Schema validation through `load_price_panel_csv`.
- Log returns through `returns_from_prices`.
- Ledoit-Wolf covariance estimate.
- Minimum-variance portfolio weights.
- Risk-parity portfolio weights.
- Historical 95% VaR and CVaR for the minimum-variance portfolio.
- Static portfolio backtest total return and max drawdown.

## Why This Belongs In The Project

The synthetic workflows prove model correctness and keep CI deterministic. The real-data-compatible workflow shows how the same APIs can be pointed at actual market exports without changing package internals.

## Production Extensions

- Add vendor-specific ingestion adapters.
- Add corporate-action and calendar validation.
- Add train/test date splits and rolling covariance windows.
- Compare optimized portfolios against benchmark ETFs.
- Store run outputs as Markdown or JSON artifacts.
- Add model governance metadata: data source, timestamp, assumptions, parameters, and validation status.
