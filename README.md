# Quant Systems Lab

[![CI](https://github.com/Fink692/quant-systems-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Fink692/quant-systems-lab/actions/workflows/ci.yml)
[![Reproduce market making](https://github.com/Fink692/quant-systems-lab/actions/workflows/reproduce-market-making.yml/badge.svg)](https://github.com/Fink692/quant-systems-lab/actions/workflows/reproduce-market-making.yml)

Quant Systems Lab is a Python quantitative-research platform centered on a real-data, queue-aware market-making study. It also contains tested implementations of stochastic-volatility options, constrained RL trading, factor risk, robust portfolio optimization, rough volatility, statistical arbitrage, credit risk, volatility-surface arbitrage, and systemic-risk networks.

The flagship pipeline ingests real NASDAQ-derived order-book messages, validates provenance, reconstructs and reconciles the book, calibrates market descriptors, and compares five policies on a chronological held-out interval with explicit queue position, latency, fees, inventory limits, adverse selection, liquidation, and independent PnL accounting.

## Why This Project Matters

- Uses immutable experiment configurations, dataset hashes, append-only run records, chronological splits, and accounting invariants.
- Preserves negative results: all five policies lose money on the public sample, and the repository explicitly avoids presenting one session as persistent alpha.
- Includes deterministic synthetic workflows for model correctness plus real order-book and S&P 500 research paths.
- Ships with 195 tests, 88% measured coverage with an 85% CI floor, Python 3.11-3.13 CI, Ruff, Black, MyPy, dependency auditing, pre-commit, documentation builds, and benchmark regression checks.

## Flagship Result: Real-Data Market Making

The reproducible public-sample study processes 301,587 synchronized AAPL messages and Level-5 book states. Reconstruction exactly matches 89.17% of synchronized states; the remaining Level-5 boundary mismatches are counted and reseeded rather than hidden. Validation selects the queue assumption before the held-out comparison.

| Policy | Test net PnL | Max absolute inventory | Max drawdown |
| --- | ---: | ---: | ---: |
| Fixed spread | -130.63 | 99 | 142.23 |
| Avellaneda-Stoikov | -98.88 | 91 | 117.22 |
| Queue aware | -67.80 | 76 | 90.81 |
| Toxicity aware | -47.34 | 52 | 71.01 |
| Latency aware | -47.34 | 52 | 71.01 |

This is a pipeline-validation result, not a profitability claim. The next empirical gate is licensed multi-session L2/L3 data with true receive timestamps.

```bash
python -m pip install -e ".[dev,research]"
make fetch-order-book-data
make reproduce-market-making-sample
make market-making-notebook
make market-making-paper
make market-making-video
```

Artifacts:

- [Research paper PDF](output/pdf/queue_aware_market_making_sample_paper.pdf)
- [One-page tear sheet](output/pdf/queue_aware_market_making_tear_sheet.pdf)
- [Narrated research demo](output/video/queue_aware_market_making_demo.mp4)
- [Executed start-here notebook](notebooks/start_here_market_making.ipynb)
- [Generated study](reports/market_making_sample/study.md)
- [Data-quality report](reports/market_making_sample/data_quality.md)
- [Architecture](docs/FLAGSHIP_ARCHITECTURE.md)
- [Assumptions and limitations](docs/ASSUMPTIONS_AND_LIMITATIONS.md)
- [Model cards](docs/MODEL_CARDS.md)
- [Roadmap completion audit](docs/ROADMAP_COMPLETION_AUDIT.md)
- [Five-minute demo runbook](docs/FIVE_MINUTE_DEMO.md)

Launch the interactive dashboard after installing `.[dashboard]`:

```bash
make market-making-dashboard
```

## System Coverage

| Area | Implemented capabilities |
| --- | --- |
| Stochastic volatility options | Black-Scholes, Heston Fourier pricing/calibration, Bates jumps, SABR smile/surface calibration, SVI/SSVI, Greeks, density extraction, PDE/Monte Carlo baselines, variance reduction, delta hedging |
| Volatility surface arbitrage | Calendar, butterfly, vertical and price-bound checks, Dupire local volatility, interpolation-stability diagnostics, constrained surface repair |
| Market making | Price-level limit order book, Avellaneda-Stoikov quotes, queue-aware book-level agent simulation, Hawkes order flow, latency replay, fill calibration, toxicity metrics, PnL attribution |
| RL trading | Trading environments, tabular Q-learning, neural Q-learning with replay, softmax policy gradient, Lagrangian constrained policy gradient, walk-forward evaluation, drawdown/leverage controls |
| Factor risk | OLS factor model, rolling out-of-sample validation, style/sector/macro/cross-sectional factors, PCA factors, factor-mimicking portfolios, covariance shrinkage, VaR/CVaR backtesting |
| Portfolio optimization | Minimum variance, mean-variance, risk parity, risk budgeting, empirical CVaR, CDaR, Black-Litterman, Bayesian shrinkage, robust and ellipsoidal robust optimization, turnover constraints, stress testing |
| Rough volatility | Rough Bergomi path simulation/pricing, variogram Hurst estimation, ATM skew power-law calibration, option-chain proxy calibration |
| Statistical arbitrage | Engle-Granger and Johansen cointegration, OU diagnostics, Kalman dynamic hedge ratios, pair/basket backtests, cointegration networks, ranked pair selection |
| Credit risk | Merton/KMV structural default, hazard bootstrapping, Cox/logistic/CIR intensity models, risky bonds/CDS, migration matrices, Gaussian copula portfolio losses, tranches, CVA/wrong-way risk |
| Systemic risk | Contagion propagation, DebtRank, Eisenberg-Noe clearing, capital adequacy, centrality, fire-sale feedback, liquidity spirals, scenario and Monte Carlo stress tests |

## Architecture

```text
src/quantlab/
  options/          Derivatives pricing, calibration, surfaces, Greeks, hedging
  market_making/    Limit order book, execution, queue, latency, Hawkes flow
  rl/               Trading environments and risk-constrained learning agents
  risk/             Factor models, covariance, attribution, VaR validation
  portfolio/        Optimizers, robust allocation, stress and drawdown analytics
  rough_vol/        Rough Bergomi simulation, pricing, and calibration
  stat_arb/         Cointegration, Kalman hedge ratios, basket/pair backtests
  credit/           Structural/reduced-form credit risk, CVA, portfolios
  systemic/         Network contagion, clearing, capital, liquidity stress
  data/             Synthetic datasets and schema-checked loaders
  market_data/      Provider adapters, manifests, reconstruction, reconciliation
  workflows/        End-to-end deterministic demo suite
  reporting/        Markdown report generation
  research/         Frozen configs, registries, calibration, chronological studies
```

Tests live in `tests/` and are intentionally broad: most modules are exercised both directly and through workflow-level smoke tests.

## Quick Start

```powershell
python -m pip install -e ".[dev,research]"
pytest
quantlab demo-suite --seed 7
```

If editable install is not desired, the tests also add `src` to `PYTHONPATH` through `tests/conftest.py`.

## CLI Examples

```powershell
quantlab price-option --spot 100 --strike 100 --maturity 1 --rate 0.03 --volatility 0.2
quantlab implied-vol --price 9.4134 --spot 100 --strike 100 --maturity 1 --rate 0.03
quantlab market-maker-demo --steps 100 --seed 7
quantlab demo-suite --seed 7
quantlab surface-demo
quantlab risk-demo --seed 7
quantlab portfolio-demo --seed 7
quantlab data-demo --seed 7
quantlab demo-report --seed 7 --output examples/demo_report_seed7.md
```

## Valuation-Regime Research Study

The repo now includes a reproducible S&P 500 valuation-regime allocation study using the DataHub/Shiller monthly dataset.

```powershell
python examples/fetch_shiller_sp500_data.py --output data/real/shiller_sp500_monthly.csv
python examples/run_valuation_regime_study.py --data data/real/shiller_sp500_monthly.csv --config config/valuation_regime.json --output reports/valuation_regime_study.md
```

Artifacts:

- [Research memo](docs/RESEARCH_MEMO_VALUATION_REGIME.md)
- [Generated tear sheet](reports/valuation_regime_study.md)
- [Data source notes](docs/DATA_SOURCES.md)
- [Hiring readiness audit](docs/HIRING_READINESS_AUDIT.md)

The study uses train/validation/test walk-forward folds through September 2023, lagged valuation signals, transaction costs, slippage, volatility targeting, drawdown controls, block-bootstrap confidence intervals, deflated Sharpe, a fold-based probability-of-overfitting diagnostic, parameter-stability tables, a bond-sleeve scenario, and 60/40, volatility-targeted, volatility-matched, and beta-matched baselines. It is intentionally honest: the strategy reduces equity risk, but does not beat buy-and-hold CAGR or the simpler risk-matched baselines on Sharpe and drawdown.

## Verification

Current local verification:

```text
195 passed; 88.54% coverage
```

GitHub Actions runs formatting, linting, scoped static typing, strict documentation builds, dependency auditing, coverage, and the complete test suite across Python 3.11, 3.12, and 3.13.

## Example Output

- [Demo report, seed 7](examples/demo_report_seed7.md)
- [Market-making case study](docs/CASE_STUDY_MARKET_MAKING.md)
- [Flagship real-data market-making research plan](docs/FLAGSHIP_MARKET_MAKING_RESEARCH_PLAN.md)
- [Real-data valuation-regime study](docs/RESEARCH_MEMO_VALUATION_REGIME.md)
- [Valuation-regime tear sheet](reports/valuation_regime_study.md)
- [Real-data-compatible price panel workflow](docs/REAL_DATA_WORKFLOW.md)
- [Hiring readiness audit](docs/HIRING_READINESS_AUDIT.md)
- [Interview prep notes](docs/INTERVIEW_PREP.md)
- [Resume project brief](docs/PROJECT_BRIEF.md)
- [GitHub profile checklist](docs/PROFILE_CHECKLIST.md)

## Visual Artifacts

These charts are generated from the package with `python examples/generate_resume_artifacts.py --seed 7`.

![Queue-aware market-making PnL and inventory](examples/artifacts/market_making_pnl_inventory.svg)

![Synthetic volatility surface slices](examples/artifacts/volatility_surface_slices.svg)

![Factor risk contributions](examples/artifacts/factor_risk_contributions.svg)

## Resume Summary

Built a 195-test Python quant-finance research platform centered on a real-data queue-aware market-making study with event-level ingestion, reconstruction, chronological evaluation, latency/queue/fee sensitivity, immutable experiment provenance, independent PnL reconciliation, and five-policy comparison; supported by derivatives, portfolio, risk, credit, statistical-arbitrage, RL, and systemic-risk modules.

## Limitations and Next Extensions

The included public order-book sample covers one session and five visible levels, has no distinct receive timestamp, and cannot establish persistent profitability. A flagship empirical claim still requires licensed multi-session data, true receive times, provider-specific reconciliation, and a later untouched test period. Production deployment would additionally require exchange connectivity, operational controls, and independent model validation.

## License

MIT License. See [LICENSE](LICENSE).
