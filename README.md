# Quant Systems Lab

This repository is the beginning of a full advanced quant finance platform covering the ten project families from the goal file:

1. Stochastic volatility options pricing: Black-Scholes baseline, Heston Fourier pricing/calibration, SABR implied volatility calibration, and Bates jump-diffusion pricing/calibration.
2. Limit order book market making: order book state, execution simulation, Avellaneda-Stoikov quotes, and Hawkes order-flow dynamics.
3. Deep RL trading with risk constraints: trading environments with transaction costs, drawdown penalties, neural Q-learning, Lagrangian constrained policy-gradient, and walk-forward validation hooks.
4. Barra-style multi-factor risk model: factor exposures, factor covariance, residual/specific risk, covariance reconstruction.
5. Portfolio optimization under uncertainty: minimum variance, mean-variance, risk parity, and CVaR optimizers.
6. Rough volatility calibration: rough Bergomi Monte Carlo path generation and ATM proxy calibration from option smiles.
7. Statistical arbitrage with cointegration networks: Engle-Granger tests, OU spread diagnostics, z-score signals.
8. Credit risk/default probability: Merton structural default probability and reduced-form hazard tools.
9. Volatility surface arbitrage detector: calendar and butterfly arbitrage checks over option price grids.
10. Systemic risk network model: network exposures, contagion propagation, stability diagnostics, and Monte Carlo asset-shock simulation.

## Current Status

This is now more than a scaffold: it is a Python package under `src/quantlab` plus tests under `tests`.
The implementation includes first-pass production-shaped APIs for pricing, calibration, simulation, optimization, stress testing, and evaluation.
It also includes validated data loaders for option chains, price panels, return construction, and credit spread curves.

The goal is not a toy trading bot. The design keeps mathematical components separated from execution/simulation layers so that each area can mature independently.

## Implemented Workflows

- Options: Black-Scholes pricing/implied vol/Greeks, option-book stress and Greek aggregation, Heston Fourier pricing and calibration, Bates jump pricing and calibration, SABR smile and term-structure calibration, SVI smile calibration, SSVI surface construction and no-arbitrage sufficient checks, pricing diagnostics, delta-hedging simulation, Monte Carlo baselines with variance reduction, finite-difference PDE baseline, Dupire local volatility, risk-neutral density extraction, calendar/butterfly/vertical/bounds surface arbitrage checks, interpolation stability diagnostics, and constrained surface repair.
- Market making: price-level limit order book, Avellaneda-Stoikov quotes, queue-aware book-level agent simulation, execution probability/adverse-selection tools, latency slippage reports, path-based delayed-fill replay, spread/inventory/slippage PnL attribution, fill-intensity calibration, Hawkes buy/sell order-flow simulation, order-flow toxicity metrics, queue-position simulation, inventory diagnostics, and a stochastic fill simulator with inventory/PnL history.
- RL trading: single-asset and multi-asset portfolio trading environments, constant-weight and constant-mix baseline policies, momentum-rotation portfolio policy, simple policy search, neural Q-learning with replay, softmax and Lagrangian constrained policy-gradient baselines, tabular Q-learning baseline, walk-forward Q-learning evaluation, leverage/drawdown risk controls, volatility targeting, risk-adjusted reward helpers, performance metrics, backtest metrics, and walk-forward split generation.
- Risk: OLS factor model, rolling out-of-sample factor-model validation, macro surprise factor construction and stress testing, cross-sectional factor-return estimation, sector exposure construction, factor-neutral portfolio projection, factor-mimicking portfolios, EWMA and Ledoit-Wolf covariance estimation, covariance reconstruction/shrinkage, PCA statistical factors, style factor construction, portfolio factor-risk attribution, VaR/CVaR, component VaR, Kupiec and Christoffersen VaR backtesting, and Basel-style traffic-light diagnostics.
- Portfolio: minimum variance, mean-variance, risk parity, risk-budget optimization, empirical CVaR and CVaR attribution, conditional drawdown-at-risk analytics and optimization, efficient frontiers, Black-Litterman posterior estimates, Bayesian return shrinkage, robust mean-variance, ellipsoidal mean-uncertainty optimization, turnover-constrained optimization, resampled efficient weights, rolling backtests, and stress scenarios.
- Rough volatility: rough Bergomi path simulation, rough Bergomi option pricing, variogram-based roughness estimation, ATM-skew power-law roughness calibration, and rough Bergomi ATM proxy calibration from option chains.
- Statistical arbitrage: Engle-Granger and Johansen tests, Johansen basket strategy backtests, Kalman-filter dynamic hedge ratios, dynamic hedge backtests, OU diagnostics, rolling z-scores, pairwise cointegration networks, ranked pair selection, spread weights, pair-portfolio allocation/backtesting, mean-reversion signals, basket spreads, and spread strategy backtests.
- Credit: Merton default probability, KMV-style distance-to-default, structural Merton asset calibration, exponential survival calibration with censoring, Cox proportional-hazards survival modeling, CIR stochastic intensity simulation, covariate logistic hazard modeling, rating migration matrices, reduced-form survival/spread tools, risky zero-coupon and coupon bond pricing, spread duration/DV01, CDS par spread pricing, counterparty exposure profiles and CVA, Gaussian-copula portfolio default simulation, tranche loss analytics, and bootstrapped hazard curves.
- Systemic risk: contagion propagation, DebtRank-style impact propagation, Eisenberg-Noe clearing, exposure stability, capital adequacy/surcharge diagnostics, external asset stress, probabilistic Monte Carlo asset-shock simulation, multi-scenario shortfall aggregation, fire-sale feedback, liquidity spiral deleveraging, and exposure centrality.
- Data and reporting: schema validation and CSV loaders for option chains, price panels, and credit spread curves, price-to-return construction, synthetic datasets, CLI demos, and Markdown report generation.

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
quantlab demo-report --seed 7 --output reports/demo.md
```

## Milestones

- M1: Core analytics and pricing primitives with deterministic tests.
- M2: Market data loaders and volatility surface calibration examples.
- M3: Full Heston/SABR/Bates calibration workflows with optimizer diagnostics.
- M4: Limit order book simulator with latency, queue position, adverse selection, and market making agent.
- M5: Portfolio, risk, stat-arb, and systemic-risk notebooks backed by reusable package APIs.
- M6: CLI/dashboard layer for running pricing, calibration, backtests, and stress tests.
