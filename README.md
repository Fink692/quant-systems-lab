# Quant Systems Lab

[![CI](https://github.com/Fink692/quant-systems-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Fink692/quant-systems-lab/actions/workflows/ci.yml)

Quant Systems Lab is a Python research platform that implements ten advanced quantitative-finance systems in one tested package: stochastic-volatility options, limit-order-book market making, constrained RL trading, Barra-style factor risk, robust portfolio optimization, rough volatility, statistical arbitrage, credit risk, volatility-surface arbitrage, and systemic-risk networks.

The project is designed as resume/interview evidence for applied mathematical finance work. It is not a toy trading bot: the code separates pricing models, calibration routines, simulation engines, portfolio/risk analytics, execution models, and reproducible workflows under `src/quantlab`.

## Why This Project Matters

- Covers derivatives pricing, market microstructure, reinforcement learning, econometrics, credit modeling, network contagion, and convex-style portfolio construction.
- Uses deterministic synthetic-data workflows so every major model family can be tested without depending on paid market data.
- Includes a CLI and report generator, making the package demonstrable from a terminal instead of only from isolated functions.
- Ships with 168 tests covering model behavior, calibration routines, risk diagnostics, workflow integration, and CLI outputs.

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
  workflows/        End-to-end deterministic demo suite
  reporting/        Markdown report generation
```

Tests live in `tests/` and are intentionally broad: most modules are exercised both directly and through workflow-level smoke tests.

## Quick Start

```powershell
python -m pip install -e .[dev]
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

## Verification

Current local verification:

```text
168 passed
```

GitHub Actions runs the same `pytest` suite on every push and pull request to `main`.

## Example Output

- [Demo report, seed 7](examples/demo_report_seed7.md)
- [Market-making case study](docs/CASE_STUDY_MARKET_MAKING.md)
- [Real-data-compatible price panel workflow](docs/REAL_DATA_WORKFLOW.md)
- [Interview prep notes](docs/INTERVIEW_PREP.md)
- [Resume project brief](docs/PROJECT_BRIEF.md)
- [GitHub profile checklist](docs/PROFILE_CHECKLIST.md)

## Visual Artifacts

These charts are generated from the package with `python examples/generate_resume_artifacts.py --seed 7`.

![Queue-aware market-making PnL and inventory](examples/artifacts/market_making_pnl_inventory.svg)

![Synthetic volatility surface slices](examples/artifacts/volatility_surface_slices.svg)

![Factor risk contributions](examples/artifacts/factor_risk_contributions.svg)

## Resume Summary

Built a tested Python quant-finance research platform covering stochastic-volatility options, market making, risk-constrained RL, Barra-style factor risk, robust portfolio optimization, credit/default modeling, statistical arbitrage, volatility-surface arbitrage, and systemic-risk contagion; packaged with CLI workflows, synthetic data validation, markdown reports, and 168 automated tests.

## Limitations and Next Extensions

The repository is research-grade scaffolding using deterministic synthetic datasets. It is suitable for demonstrating model implementation, numerical methods, workflow design, and test discipline. Production deployment would require live data connectors, execution-system integration, parameter governance, model-risk documentation, and independent calibration validation against real market datasets.

## License

MIT License. See [LICENSE](LICENSE).
