# Interview Prep

Use this as a quick prep sheet before sharing or discussing the repository.

## Project Pitch

Quant Systems Lab is a tested Python platform for advanced quantitative-finance systems. It implements derivatives pricing, volatility-surface diagnostics, queue-aware market making, risk-constrained RL trading, factor risk, robust portfolio optimization, statistical arbitrage, credit/default models, systemic-risk contagion, and two real-data allocation studies in one package with CLI workflows and automated tests.

## Best 60-Second Explanation

I built a research-grade quant-finance package that covers ten major modeling families plus two real-data allocation studies. The goal was not to make a black-box trading bot, but to demonstrate that I can implement the mathematical and systems pieces behind institutional quant workflows: stochastic-volatility pricing and calibration, market-microstructure simulation, constrained RL, factor risk decomposition, robust portfolio construction, credit risk, stat arb, volatility-surface arbitrage, contagion, and chronological holdout research with transaction costs. The repo includes deterministic synthetic-data workflows, daily and monthly real-market studies, generated reports, Docker/Make reproducibility, and CI-tested coverage across the stack.

## High-Value Talking Points

- Heston/Bates use characteristic-function pricing and optimizer-based calibration against option quotes.
- SABR/SVI/SSVI are used to construct smooth implied-volatility smiles and surfaces.
- Surface arbitrage checks include calendar, butterfly, vertical, and price-bound violations.
- The market maker has both a simple probabilistic simulator and a queue-aware limit-order-book simulator.
- Queue-aware fills require market-order flow to consume resting depth before the agent receives execution.
- RL environments include transaction costs, drawdown penalties, leverage controls, walk-forward validation, and constrained policy-gradient learning.
- Factor risk includes OLS exposures, covariance reconstruction, style/sector/macro/cross-sectional factors, PCA, attribution, and rolling out-of-sample validation.
- Portfolio optimizers include mean-variance, min variance, risk parity, CVaR, CDaR, Black-Litterman, Bayesian shrinkage, robust mean-variance, and turnover constraints.
- Credit risk is modeled structurally with Merton/KMV and reduced-form with hazard curves, Cox/logistic models, CIR intensities, CDS pricing, copula losses, tranches, and CVA.
- Systemic risk covers Eisenberg-Noe clearing, contagion, DebtRank, fire-sale feedback, liquidity spirals, scenario analysis, and Monte Carlo stress.
- The real-data study uses monthly S&P 500/Shiller data, lagged PE10 signals, train/validation/test walk-forward folds, transaction costs, volatility targeting, drawdown controls, a tear sheet, robustness checks, and failure analysis.

## Likely Questions

### Options And Volatility

Q: Why include multiple volatility models?

A: Different models answer different questions. Heston provides stochastic variance and characteristic-function pricing, Bates adds jumps for crash/skew behavior, SABR is useful for smile parameterization, and rough Bergomi captures rough-volatility behavior seen in short-dated skew scaling.

Q: What proves a surface is usable?

A: Smooth interpolation is not enough. The surface should avoid calendar arbitrage, butterfly arbitrage, vertical-spread violations, and price-bound violations. The repo includes diagnostics and constrained repair utilities for those checks.

### Market Making

Q: What is the gap between Avellaneda-Stoikov and a real order book?

A: Avellaneda-Stoikov gives an optimal quoting policy under stylized assumptions. A real book adds tick sizes, queue priority, cancellations, market-order flow, latency, adverse selection, and inventory path dependence. The repo models that second layer explicitly.

Q: Why does queue position matter?

A: A quote at the best bid does not fill just because a sell market order arrives. Existing resting quantity at that price fills first. The simulator records queue-ahead depth and only credits the agent after that depth is consumed.

### RL Trading

Q: Why constrained policy gradient?

A: Clipping actions after training is weaker than teaching the policy that drawdown and turnover violations have a cost. The Lagrangian trainer updates penalties based on observed violations, so constraints influence learning rather than only post-processing.

Q: How do you avoid overfitting?

A: The repo uses deterministic synthetic tests for correctness and walk-forward evaluation hooks for out-of-sample style testing. A production extension would add real market data, purged cross-validation, and transaction-cost calibration.

Q: What changed when you added real data?

A: I added a separate S&P 500 valuation-regime study rather than retrofitting every synthetic test. It fetches sourced monthly Shiller/DataHub data, uses lagged CAPE/PE10 signals, chooses thresholds only inside each training/validation fold, deducts costs and slippage, and reports both the good and bad news: lower beta and better risk-adjusted behavior, but lower CAGR than buy-and-hold.

### Q: Did the leveraged strategy really make 20% per year?

A: It produced a 23.29% CAGR on the January 2021 to July 2026 historical holdout after modeled costs. It did not earn 20% every calendar year, and a reconstructed 2000-2026 history using real QQQ and FRED rates produced only 15.13%. The reconstruction correlates 0.99894 with actual TQQQ returns but is optimistic by 2.38% annually, so the long-history failure is not a conservative-data artifact. I present the recent result as regime-dependent evidence worth forward-testing, not a promised return.

### Q: How are you preventing yourself from changing the model after seeing new returns?

A: The parameters are frozen under a versioned strategy ID. Each decision is timestamped for a later effective session, includes hashes of the exact configuration and source snapshot, and extends a hash-chained JSONL ledger. The recorder rejects duplicate dates, altered history, broken chains, and data-source disagreement. The first target was recorded after the July 13 close for July 14, before that return was known.

### Q: Can the close signal actually be traded at that same close?

A: No. The execution audit carries the old allocation overnight, changes to the frozen target at the next open, and then earns the new allocation's intraday return, with 10 bps turnover cost. The 2021 onward holdout is 25.53% CAGR with 23.06% maximum drawdown under that convention. The paper outcome ledger uses the same accounting and recomputes every result from completed OHLC.

### Factor Risk And Portfolio

Q: What makes this Barra-style?

A: It decomposes asset returns into factor exposures, factor covariance, and specific risk, then reconstructs covariance and attributes portfolio variance back to factors and assets. It also includes style, sector, macro, cross-sectional, and statistical PCA factors.

Q: Why robust optimization?

A: Mean-variance portfolios are fragile because expected returns are noisy. Robust and Bayesian methods shrink or penalize uncertain estimates so the optimizer is less dominated by estimation error.

### Credit And Systemic Risk

Q: Why both structural and reduced-form credit models?

A: Structural models connect default to balance-sheet asset value and leverage. Reduced-form models treat default arrival as an intensity process and are more directly tied to credit spreads, survival curves, CDS pricing, and covariates.

Q: What is systemic risk doing here?

A: The systemic module treats financial institutions as a network of exposures and assets. It can model direct contagion, clearing defaults, capital shortfalls, fire-sale feedback, liquidity spirals, and Monte Carlo default probability under shocks.

## Honest Limitations

- The repository is a research platform, not a production trading system.
- Most workflows use deterministic synthetic data so the project can be tested anywhere.
- The real-data valuation study is an index allocation proxy, not a directly tradable strategy without ETF/futures execution assumptions.
- The study improves risk-adjusted behavior in the tested sample, but it does not beat buy-and-hold S&P 500 CAGR.
- Live deployment would require market-data ingestion, execution adapters, calibration governance, model-risk documentation, and independent validation.
- The visual artifacts are explanatory examples, not performance claims.

## Strong Closing Line

The value of the project is that it shows I can turn mathematical finance concepts into tested, modular software with reproducible workflows and honest research memos, not just describe the models at a whiteboard.
