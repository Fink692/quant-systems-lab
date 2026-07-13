# Hiring Readiness Audit

This document maps the strongest hiring-signal requirements for a quant research repository to concrete artifacts in this repo.

| Requirement | Evidence |
| --- | --- |
| Real market data | Sourced Shiller S&P 500 monthly data plus checksum-tracked TQQQ, QQQ, and BIL daily adjusted prices. Fetch and cleaning logic is in `examples/fetch_shiller_sp500_data.py` and `examples/fetch_leveraged_etf_data.py`; source notes are in `docs/DATA_SOURCES.md`. |
| Economic strategy logic | `docs/RESEARCH_MEMO_VALUATION_REGIME.md` explains valuation-regime allocation; `docs/RESEARCH_MEMO_LEVERAGED_TREND.md` explains trend persistence, volatility drag, and risk scaling for leveraged exposure. |
| Walk-forward research design | `src/quantlab/research/valuation_regime.py` uses train/validation/test folds with lagged signals and stitched out-of-sample history. |
| Transaction costs and execution assumptions | The strategy deducts turnover-based transaction cost and slippage. The memo is explicit that the S&P 500 index is an execution proxy and would need ETF/futures assumptions before live use. |
| Risk management | Volatility targeting, leverage caps, drawdown de-risking, turnover accounting, VaR/CVaR, drawdown duration, beta, alpha, and tracking error are implemented in the tear sheet. |
| Full performance tear sheet | `reports/valuation_regime_study.md` contains CAGR, Sharpe, Sortino, drawdown, Calmar, hit rate, profit factor, turnover, cost, beta, alpha, VaR/CVaR, regime breakdown, stress tests, and cost robustness. |
| Robustness checks | The studies include cost grids, 48-parameter sensitivity, regime and stress analysis, and a 2,000-sample moving-block bootstrap. |
| Failure analysis | Both research memos document underperformance, losing periods, selection risk, leverage/path dependence, execution limits, and uncertainty. The leveraged memo also records a failed 15.38% holdout candidate. |
| Production-style reproducibility | `Makefile`, `Dockerfile`, `config/valuation_regime.json`, `pyproject.toml`, and CLI-style examples make the study reproducible from a clean environment. |
| Financial correctness tests | `tests/test_valuation_regime_research.py` checks no date leakage across folds, bounded exposures, finite risk metrics, cost accounting, robustness output, and report generation. |
| Honest scope control | The project keeps synthetic data for deterministic CI model tests and uses a separate real-data study for research evidence rather than mixing the two claims. |

## Reproduction Commands

```powershell
python -m pip install -e .[dev]
python examples/fetch_shiller_sp500_data.py --output data/real/shiller_sp500_monthly.csv
python examples/run_valuation_regime_study.py --data data/real/shiller_sp500_monthly.csv --config config/valuation_regime.json --output reports/valuation_regime_study.md
python examples/run_leveraged_trend_study.py --data data/real/leveraged_etf_adjusted.csv --config config/leveraged_trend.json --output reports/leveraged_trend_study.md
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
