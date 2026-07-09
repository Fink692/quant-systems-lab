# Hiring Readiness Audit

This document maps the strongest hiring-signal requirements for a quant research repository to concrete artifacts in this repo.

| Requirement | Evidence |
| --- | --- |
| Real market data | `data/real/shiller_sp500_monthly.csv`, fetched by `examples/fetch_shiller_sp500_data.py` from the DataHub S&P 500/Shiller dataset. Source notes are in `docs/DATA_SOURCES.md`. |
| Economic strategy logic | `docs/RESEARCH_MEMO_VALUATION_REGIME.md` explains the valuation-regime hypothesis: low CAPE/PE10 implies better forward equity compensation; high CAPE/PE10 implies lower exposure. |
| Walk-forward research design | `src/quantlab/research/valuation_regime.py` uses train/validation/test folds with lagged signals and stitched out-of-sample history. |
| Transaction costs and execution assumptions | The strategy deducts turnover-based transaction cost and slippage. The memo is explicit that the S&P 500 index is an execution proxy and would need ETF/futures assumptions before live use. |
| Risk management | Volatility targeting, leverage caps, drawdown de-risking, turnover accounting, VaR/CVaR, drawdown duration, beta, alpha, and tracking error are implemented in the tear sheet. |
| Full performance tear sheet | `reports/valuation_regime_study.md` contains CAGR, Sharpe, Sortino, drawdown, Calmar, hit rate, profit factor, turnover, cost, beta, alpha, VaR/CVaR, regime breakdown, stress tests, and cost robustness. |
| Robustness checks | The report and code test a cost grid from 0 to 50 bps and summarize performance by valuation regime and stress scenario. |
| Failure analysis | `docs/RESEARCH_MEMO_VALUATION_REGIME.md` calls out underperformance versus buy-and-hold CAGR, expensive-market momentum risk, long drawdown duration, and monthly-data limitations. |
| Production-style reproducibility | `Makefile`, `Dockerfile`, `config/valuation_regime.json`, `pyproject.toml`, and CLI-style examples make the study reproducible from a clean environment. |
| Financial correctness tests | `tests/test_valuation_regime_research.py` checks no date leakage across folds, bounded exposures, finite risk metrics, cost accounting, robustness output, and report generation. |
| Honest scope control | The project keeps synthetic data for deterministic CI model tests and uses a separate real-data study for research evidence rather than mixing the two claims. |

## Reproduction Commands

```powershell
python -m pip install -e .[dev]
python examples/fetch_shiller_sp500_data.py --output data/real/shiller_sp500_monthly.csv
python examples/run_valuation_regime_study.py --data data/real/shiller_sp500_monthly.csv --config config/valuation_regime.json --output reports/valuation_regime_study.md
pytest
```

With GNU Make available:

```powershell
make install
make fetch-real-data
make reproduce-strategy
make test
```

With Docker:

```powershell
docker build -t quant-systems-lab .
docker run --rm quant-systems-lab
```
