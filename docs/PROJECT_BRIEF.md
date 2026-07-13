# Quant Systems Lab Project Brief

## One-Line Pitch

A tested Python quant-finance platform implementing ten advanced institutional modeling systems across derivatives, market microstructure, portfolio construction, risk, credit, statistical arbitrage, and systemic contagion, plus two real-data allocation studies.

## What This Demonstrates

- Mathematical finance implementation: Heston, SABR, Bates, rough Bergomi, Merton/KMV, hazard curves, CDS pricing, CVA, and volatility-surface no-arbitrage diagnostics.
- Market microstructure modeling: price-level limit order book, queue-aware fills, Avellaneda-Stoikov quoting, latency replay, Hawkes order flow, toxicity, and PnL attribution.
- Machine learning for trading: tabular Q-learning, neural Q-learning, policy-gradient training, Lagrangian risk constraints, walk-forward validation, and realistic transaction-cost/drawdown penalties.
- Institutional risk analytics: Barra-style factor decomposition, covariance shrinkage, PCA/statistical factors, macro/style/sector factors, VaR/CVaR, Kupiec/Christoffersen validation, Basel traffic-light diagnostics.
- Portfolio engineering: robust optimization, Black-Litterman, Bayesian return shrinkage, risk parity, risk budgeting, CVaR, CDaR, turnover constraints, stress testing, and rolling backtests.
- Real-data research: sourced monthly and daily market data, chronological train/validation/holdout splits, transaction costs, slippage, volatility targeting, tear sheets, bootstrap uncertainty, robustness checks, and failure analysis.
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

5. Real-data research evidence.
   `src/quantlab.research.valuation_regime` runs a reproducible S&P 500 valuation-regime allocation study with lagged CAPE/PE10 signals and honest out-of-sample reporting. `src/quantlab.research.leveraged_trend` adds a daily TQQQ holdout study; `src/quantlab.research.leveraged_trend_stress` falsifies 20% persistence over 2000-2026; `src/quantlab.research.paper_trading` adds prospective hash-chained decisions.

## Evidence of Completion

- Package: `src/quantlab`
- Tests: `tests`
- CLI: `quantlab demo-suite --seed 7`
- Report generation: `quantlab demo-report --seed 7 --output examples/demo_report_seed7.md`
- Real-data study: `python examples/run_valuation_regime_study.py --data data/real/shiller_sp500_monthly.csv --config config/valuation_regime.json --output reports/valuation_regime_study.md`
- Leveraged trend study: `python examples/run_leveraged_trend_study.py --data data/real/leveraged_etf_adjusted.csv --config config/leveraged_trend.json --output reports/leveraged_trend_study.md`
- Long-history falsification: `python examples/run_leveraged_trend_stress.py --data data/real/qqq_fred_stress_daily.csv --actual data/real/leveraged_etf_adjusted.csv --config config/leveraged_trend_stress.json --output reports/leveraged_trend_long_history.md`
- Forward protocol: `docs/PAPER_TRADING_PROTOCOL.md`
- Visual artifacts: `python examples/generate_resume_artifacts.py --seed 7`
- Case study: `docs/CASE_STUDY_MARKET_MAKING.md`
- Real-data-compatible workflow: `examples/run_price_panel_example.py`
- Research memos: `docs/RESEARCH_MEMO_VALUATION_REGIME.md` and `docs/RESEARCH_MEMO_LEVERAGED_TREND.md`
- Hiring readiness audit: `docs/HIRING_READINESS_AUDIT.md`
- Current local verification: `204 passed`, 88.66% coverage
- Continuous integration: `.github/workflows/ci.yml`

## Resume Bullet

Built Quant Systems Lab, a 204-test Python platform centered on a real-data queue-aware market-making study with event ingestion, reconstruction, chronological evaluation, latency/queue/fee sensitivity, immutable provenance, and independent PnL reconciliation, supported by two reproducible real-data allocation studies, prospective hash-chained paper decisions, and derivatives, portfolio, risk, credit, statistical-arbitrage, RL, and systemic-risk modules.

## Interview Talking Points

- How Heston and Bates prices are computed with characteristic functions and calibrated against synthetic option quotes.
- Why no-arbitrage volatility-surface diagnostics require calendar, butterfly, vertical, and bounds checks.
- How queue-aware market-making fills differ from simple probabilistic fill models.
- Why constrained RL trading should penalize drawdown/turnover during training rather than only clipping actions afterward.
- How rolling out-of-sample factor-model validation exposes risk-model overfit.
- Why robust and Bayesian portfolio methods are included alongside mean-variance optimization.
- How credit risk is modeled both structurally and with reduced-form intensities.
- How network contagion, clearing, fire-sale feedback, and liquidity spirals describe different systemic-risk channels.
- Why the real-data valuation-regime study is framed as a risk-managed allocation overlay rather than an overclaimed alpha strategy.
- Why the leveraged trend study's 23.29% holdout CAGR is evidence for paper trading, not a promise of 20% annual returns.
