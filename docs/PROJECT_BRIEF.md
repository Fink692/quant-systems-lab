# Quant Systems Lab Project Brief

## One-Line Pitch

A tested Python quant-finance platform implementing ten advanced institutional modeling systems across derivatives, market microstructure, portfolio construction, risk, credit, statistical arbitrage, and systemic contagion.

## What This Demonstrates

- Mathematical finance implementation: Heston, SABR, Bates, rough Bergomi, Merton/KMV, hazard curves, CDS pricing, CVA, and volatility-surface no-arbitrage diagnostics.
- Market microstructure modeling: price-level limit order book, queue-aware fills, Avellaneda-Stoikov quoting, latency replay, Hawkes order flow, toxicity, and PnL attribution.
- Machine learning for trading: tabular Q-learning, neural Q-learning, policy-gradient training, Lagrangian risk constraints, walk-forward validation, and realistic transaction-cost/drawdown penalties.
- Institutional risk analytics: Barra-style factor decomposition, covariance shrinkage, PCA/statistical factors, macro/style/sector factors, VaR/CVaR, Kupiec/Christoffersen validation, Basel traffic-light diagnostics.
- Portfolio engineering: robust optimization, Black-Litterman, Bayesian return shrinkage, risk parity, risk budgeting, CVaR, CDaR, turnover constraints, stress testing, and rolling backtests.
- System design: package layout, CLI workflows, deterministic synthetic data, schema validation, Markdown reporting, and broad automated tests.

## Technical Differentiators

1. Breadth with shared workflow coverage.
   The project is not ten disconnected scripts. `quantlab.workflows.demo_suite` runs an end-to-end deterministic workflow across all model families.

2. Model diagnostics, not just model outputs.
   The repo includes calibration objective values, pricing error reports, no-arbitrage checks, factor-model out-of-sample validation, VaR exception testing, and market-making PnL attribution.

3. Simulation realism where it matters.
   The market-making stack includes queue position, adverse selection, latency, Hawkes order flow, and inventory risk. The RL stack includes transaction costs, drawdown penalties, leverage controls, and constrained policy-gradient training.

4. Testable without paid data.
   Synthetic data generators create controlled option chains, price panels, factor panels, credit spread curves, cointegrated assets, and financial networks so the full package can be verified anywhere.

## Evidence of Completion

- Package: `src/quantlab`
- Tests: `tests`
- CLI: `quantlab demo-suite --seed 7`
- Report generation: `quantlab demo-report --seed 7 --output examples/demo_report_seed7.md`
- Visual artifacts: `python examples/generate_resume_artifacts.py --seed 7`
- Case study: `docs/CASE_STUDY_MARKET_MAKING.md`
- Real-data-compatible workflow: `examples/run_price_panel_example.py`
- Current local verification: `168 passed`
- Continuous integration: `.github/workflows/ci.yml`

## Resume Bullet

Built Quant Systems Lab, a tested Python platform implementing stochastic-volatility options pricing, queue-aware market making, risk-constrained RL trading, Barra-style factor risk, robust portfolio optimization, credit/default models, statistical arbitrage, volatility-surface arbitrage, and systemic-risk contagion, with CLI workflows, Markdown reports, synthetic-data validation, and 168 automated tests.

## Interview Talking Points

- How Heston and Bates prices are computed with characteristic functions and calibrated against synthetic option quotes.
- Why no-arbitrage volatility-surface diagnostics require calendar, butterfly, vertical, and bounds checks.
- How queue-aware market-making fills differ from simple probabilistic fill models.
- Why constrained RL trading should penalize drawdown/turnover during training rather than only clipping actions afterward.
- How rolling out-of-sample factor-model validation exposes risk-model overfit.
- Why robust and Bayesian portfolio methods are included alongside mean-variance optimization.
- How credit risk is modeled both structurally and with reduced-form intensities.
- How network contagion, clearing, fire-sale feedback, and liquidity spirals describe different systemic-risk channels.
