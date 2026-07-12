# Flagship Research Plan: Queue-Aware Market Making on Real Order-Book Data

## Research claim

This project will test whether explicit queue state improves executable market-making outcomes after latency, fees, inventory risk, and adverse selection are modeled on a strictly out-of-sample event stream.

The primary claim is deliberately narrow: a queue-aware policy should improve risk-adjusted PnL or adverse-selection-adjusted fill quality relative to both a fixed-spread baseline and Avellaneda-Stoikov under the same market data, inventory limits, order size, fee schedule, and latency assumptions. Toxicity- and latency-aware variants are extensions of that core comparison, not substitutes for it.

## Locked decisions before data access

The following choices must be written into a copied, immutable version of `config/market_making_flagship.example.json` before the test period is examined:

- One venue and one highly liquid instrument for the first paper.
- L3 data when order identifiers and priority rules are available; otherwise L2 with queue position treated as an explicit uncertainty scenario.
- UTC nanosecond timestamps, exchange sequence numbers, regular-session boundaries, tick size, lot size, fee tier, and trading calendar.
- Chronological train, validation, and test windows with no overlapping messages or parameter fitting across boundaries.
- The five strategy definitions, inventory limits, order size, quote update rules, latency scenarios, and queue-ahead assumptions.
- Primary endpoint, secondary endpoints, statistical tests, and failure criteria.

The example configuration is not a publishable experiment: its provider, instrument, dates, and zero dataset hash are placeholders.

## Data acquisition and provenance

Use a source whose license permits local research, derived artifacts, and public description of the methodology. Record the provider, product name, venue, instrument, coverage dates, book level, timestamp semantics, sequence semantics, known outages, corrections, and redistribution restrictions in `docs/DATA_SOURCES.md`.

Raw files remain read-only. Each acquired file receives a SHA-256 digest. A dataset manifest lists the raw file digests, ingestion code commit, canonical schema version, row counts, first and last sequence numbers, session counts, and exclusions. The combined canonical dataset receives the digest recorded in the frozen experiment config.

Data selection must not depend on realized strategy performance. Instrument and date selection may depend on data quality and liquidity criteria defined before running strategy comparisons.

## Canonical event contract

The canonical loader in `quantlab.data` accepts an event-ordered L2/L3 table with these required fields:

| Field | Meaning | Contract |
| --- | --- | --- |
| `timestamp_ns` | Exchange event time | Non-negative integer UTC nanoseconds; non-decreasing |
| `sequence_number` | Exchange/feed ordering key | Non-negative integer; unique and strictly increasing within a canonical stream |
| `event_type` | Book mutation | `add`, `cancel`, `modify`, or `trade` |
| `side` | Resting-book side affected | `bid` or `ask` |
| `price` | Event price | Positive and on the instrument tick grid |
| `quantity` | Added, removed, modified, or traded quantity | Positive |

Optional fields include `order_id` for L3 priority reconstruction and `receive_timestamp_ns` for observed feed latency. When present, receive time cannot precede exchange time. Venue-specific messages must be transformed without silently rounding timestamps, prices, or quantities.

Before research begins, adapter-specific tests must cover snapshot/reset behavior, sequence gaps, duplicate messages, out-of-order packets, trading halts, auctions, crossed or locked states, order replacement semantics, partial executions, hidden liquidity, and session rollover. The current general contract intentionally does not guess how a particular venue represents those cases.

## Event-by-event reconstruction

The reconstruction engine will:

1. Initialize from a full-depth snapshot or a documented empty-book boundary.
2. Apply messages strictly by sequence number.
3. Maintain price-time priority for L3, or aggregated depth for L2.
4. Reconcile periodic snapshots without overwriting unexplained differences.
5. Emit top-of-book and depth-state checks after every mutation.
6. Quarantine sessions with unresolved gaps, negative depth, invalid cancels, or unexplained crossed books.

Acceptance requires exact reproduction of provider snapshots at reconciliation points. Results will report the fraction of sessions and messages excluded and the reason for each exclusion.

## Chronological research design

### Train

Estimate descriptive and execution parameters only: market-order arrival intensity, cancellation and modification intensity, size distributions, spread transition probabilities, volatility, Hawkes parameters, queue depletion, fill probability, short-horizon price response, and toxicity features. Diagnostics are stratified by time of day and predeclared market regimes.

### Validation

Choose a single specification for each strategy, including risk aversion, horizon, quote distance bounds, inventory skew, toxicity threshold, update frequency, and any regularization. Sensitivity analysis may influence the locked choice here.

### Test

Run the frozen configurations once. No parameter changes are allowed after aggregate test outcomes are viewed. Any subsequent change creates a new experiment identifier and treats the old test interval as development data; it must use a later untouched test period.

Embargo intervals at split boundaries must be at least the longest feature lookback plus the maximum modeled latency. Intraday state does not cross session boundaries unless explicitly modeled and declared.

## Strategy comparison

All policies receive the same event stream, starting inventory, capital convention, order size, inventory cap, venue rules, fee schedule, latency scenario, and cancellation mechanics.

1. **Fixed spread:** symmetric quotes at a predeclared number of ticks, with only a hard inventory limit.
2. **Avellaneda-Stoikov:** reservation-price and optimal-spread quotes calibrated on train and selected on validation.
3. **Queue aware:** Avellaneda-Stoikov augmented with queue-ahead state, expected depletion, and fill probability.
4. **Toxicity aware:** queue-aware quotes widened, skewed, or suppressed using only contemporaneously available order-flow toxicity features.
5. **Latency aware:** toxicity-aware decision logic evaluated against stale book state and explicit outbound/cancel latency.

For L2 data, the queue-aware comparison is reported across the full predeclared queue-ahead grid. A single favorable queue assumption cannot be presented as the headline result.

## Calibration targets

Calibration tables will compare predicted and realized quantities on validation and test data:

- Arrival, cancellation, modification, and trade intensity by side, spread, depth, time of day, and regime.
- Fill probability by queue-ahead depth, quote distance, resting time, and latency.
- Queue depletion and replenishment distributions.
- Spread duration and transition probabilities.
- Mid-price response 10 ms, 100 ms, 1 s, and 5 s after fills.
- Adverse selection conditional on side, imbalance, signed flow, volatility, and toxicity bucket.
- Hawkes residual and time-rescaling diagnostics, compared with a Poisson baseline.

Calibration error must be shown, not only fitted parameter values. Independent simple benchmark implementations are retained for fill probabilities, Poisson arrivals, and mark-to-market accounting.

## Outcomes

The primary endpoint is net PnL divided by a predeclared inventory-risk measure, with uncertainty estimated by session-level block bootstrap. Raw PnL alone is not sufficient.

Required secondary outcomes are:

- Gross and net PnL after maker/taker fees and fixed order charges.
- Realized spread and implementation shortfall against decision-time and arrival-time mid prices.
- Fill rate, fill ratio, time to fill, cancel-to-fill ratio, and missed fills.
- Adverse-selection cost at all declared response horizons.
- Mean, maximum, and time-weighted absolute inventory; inventory variance and limit breaches.
- Daily loss distribution, expected shortfall, worst session, maximum drawdown, and recovery time.
- PnL attribution to spread capture, fees, inventory mark-to-market, adverse selection, and liquidation.
- Quote uptime, message rate, turnover, and capital usage.

Report paired differences between strategies on identical sessions. Confidence intervals use session-level or stationary block bootstrap rather than treating individual messages as independent observations. Effect sizes, interval estimates, and multiple-comparison handling accompany p-values.

## Sensitivity and failure analysis

At minimum, rerun the locked test across:

- The configured latency grid for market-data observation, order entry, and cancellation.
- The configured queue-ahead grid, or empirically reconstructed L3 priority.
- Actual fees plus adverse fee/rebate scenarios.
- Order size, inventory cap, tick size regime, and quote-update throttling.
- High/low volatility, wide/tight spread, high/low toxicity, open/midday/close, and trend/reversal regimes.
- Sequence-gap exclusions and conservative hidden-liquidity assumptions.
- Forced end-of-session liquidation and no-liquidation accounting.

Failure cases are first-class results. The paper will show sessions where queue awareness loses, inventory becomes concentrated, cancellations arrive too late, fills predict price moves against the strategy, or estimated intensities fail out of sample.

## Governance and reproducibility

Every publishable run records:

- Git commit, Python version, dependency lock digest, operating system, random seed, and experiment-config fingerprint.
- Raw and canonical dataset hashes, provenance, schema version, adapter version, exclusions, and validation report.
- Exact command, start/end time, generated artifact hashes, and pass/fail status.
- Model card covering intended use, training inputs, assumptions, limitations, and prohibited claims.
- Assumption register and a change log explaining any departure from the preregistered plan.

Published result directories are append-only and use the experiment identifier plus config fingerprint. Generated reports must refuse to combine artifacts with different dataset or config fingerprints.

## Delivery sequence

### Milestone 0 — contracts and preregistration

- Select and license a data source.
- Copy and complete the versioned experiment configuration.
- Write provider adapter tests and a dataset manifest generator.
- Lock research questions, endpoints, splits, and failure criteria.

Exit: one canonical sample session validates, reconstructs, and matches source snapshots; the dataset and config fingerprints are recorded.

### Milestone 1 — reconstruction and calibration

- Build L2/L3 reconstruction and reconciliation.
- Produce data-quality and market-descriptive reports.
- Calibrate arrivals, cancellations, fills, spread dynamics, toxicity, and short-horizon adverse selection on train only.

Exit: validation-period calibration plots and benchmark checks pass predeclared tolerances.

### Milestone 2 — comparable replay engine

- Connect all five policies to one event-driven execution interface.
- Add queue priority, fees, latency, rate limits, rejects, and liquidation.
- Verify cash, inventory, fills, and attribution with deterministic fixtures and independent accounting.

Exit: identical actions produce identical fills and PnL across the primary and benchmark accounting paths.

### Milestone 3 — locked evaluation

- Freeze the selected configurations after validation.
- Run the untouched test once, then run the predeclared sensitivities and bootstrap.
- Generate tables, plots, failure-case replays, and the full audit manifest.

Exit: all headline numbers trace to immutable data, code, configuration, and artifact hashes.

### Milestone 4 — demonstration package

- Publish the paper PDF, tear sheet, start-here notebook, architecture diagram, and five-minute demo.
- Add one command that validates prerequisites and reproduces all distributable artifacts from licensed local data.

Exit: a clean environment can reproduce the reported artifact hashes, subject to data-license access.

## Stop/go criteria

Stop and report a data-quality study instead of a strategy paper if sequence integrity, snapshot reconciliation, or timestamp semantics cannot be established. Do not claim queue-priority evidence from L2 data without sensitivity bounds. Do not claim alpha if improvements disappear after matching inventory risk, quote uptime, fees, and latency. A null or negative result remains publishable when the reconstruction, controls, and failure analysis are sound.

## First reproducible command

After replacing every placeholder and ingesting licensed data, the intended contract check is:

```bash
python -m pytest tests/test_market_making_research_contracts.py
```

The next implementation should add a provider-specific ingestion command that writes the canonical events, manifest, validation report, and SHA-256 digest without modifying raw source files.
