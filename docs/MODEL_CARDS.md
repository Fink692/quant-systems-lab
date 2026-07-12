# Model Cards

These cards define intended use, evidence, and failure boundaries for the repository's published research models. They are not investment recommendations or production-trading approvals.

## Market-Making Policy Family

| Field | Description |
| --- | --- |
| Models | Fixed spread, Avellaneda-Stoikov, queue aware, toxicity aware, latency aware |
| Intended use | Counterfactual comparison of quoting policies inside one shared historical replay engine |
| Data | Public LOBSTER AAPL Level-5 sample, 2012-06-21; 301,587 synchronized messages and snapshots |
| Training | Descriptive arrival, cancellation, execution, spread-transition, volatility, and adverse-selection calibration on the chronological training interval |
| Selection | Queue-ahead assumption selected on validation; policy comparison frozen before the test interval |
| Primary metrics | Net PnL after fees, fill rate, inventory, drawdown, adverse-selection cost, implementation shortfall, accounting error |
| Evidence | All five base policies lose money on the held-out interval; the result validates the pipeline, not persistent alpha |
| Main limitations | One session, five visible levels, boundary reseeding, no distinct receive timestamps, small-agent counterfactual assumption |
| Prohibited use | Live deployment, capacity claims, profitability claims, or latency calibration without licensed multi-session L2/L3 data |

The fixed-spread and Avellaneda-Stoikov policies are reference baselines. Queue-aware behavior requires displayed depth to deplete before an agent fill. Toxicity-aware behavior widens or suppresses quotes under signed-flow pressure. Latency-aware behavior applies the same controls to delayed observable state. All policies share fill, fee, inventory, liquidation, and cash-ledger logic so policy code cannot silently change execution assumptions.

## Valuation-Regime Allocation

| Field | Description |
| --- | --- |
| Model | Lagged Shiller CAPE/PE10 regime allocation with volatility scaling and drawdown control |
| Intended use | Research demonstration of chronological asset-allocation evaluation and risk attribution |
| Data | Monthly S&P 500, dividend, long-rate, and CAPE data; current study input ends September 2023 |
| Training | Rolling 120-month threshold estimation, 36-month validation, and up-to-36-month test folds |
| Selection | Nine cheap/expensive percentile pairs; final partial fold is retained when at least 12 months are available |
| Primary metrics | CAGR, volatility, Sharpe, drawdown, beta, alpha, turnover, costs, block-bootstrap intervals, deflated Sharpe, fold-based overfitting rate, and a duration-based bond-sleeve scenario |
| Evidence | Lower equity risk and positive point-estimate alpha, but alpha's 95% bootstrap interval crosses zero and simple risk-matched baselines have better Sharpe/drawdown |
| Main limitations | U.S.-only sample, index proxy, approximate bond return, slow valuation signal, monthly execution, no managed-futures comparator |
| Prohibited use | Claiming timing alpha, extrapolating to international markets, or treating the 60/40 proxy as a tradeable implementation |

## Governance Requirements

Any published model change must create a new immutable configuration and experiment record, preserve chronological train/validation/test separation, record the dataset fingerprint and random seed, reconcile independent accounting where applicable, and document both favorable and unfavorable results. Test-period-driven changes require a later untouched test period.
