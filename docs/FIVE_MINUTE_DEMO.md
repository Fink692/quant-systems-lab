# Five-Minute Demo Runbook

This is the recording script for a concise repository walkthrough. Record the terminal and browser at 1080p; do not show raw licensed data rows.

## 0:00-0:30 - Research question

Show the repository README and say:

> Quant Systems Lab tests whether explicit queue state improves executable market-making outcomes after latency, fees, inventory risk, and adverse selection. The research compares five policies on the same chronological event stream.

## 0:30-1:15 - Reproducibility contract

Open `config/lobster_sample_experiment.json` and point out:

- Dataset fingerprint
- Train, validation, embargo, and test periods
- Five fixed strategies
- Latency, queue, and fee grids
- Execution limits, model parameters, and bootstrap settings

Run:

```bash
make quality
```

## 1:15-2:00 - Real-data integrity

Open `reports/market_making_sample/data_quality.md`. Explain sequence checks, UTC conversion, synchronized snapshots, reconstruction matching, Level-5 boundary mismatches, and why the sample is not sufficient for an alpha claim.

## 2:00-3:00 - Shared replay engine

Open `docs/FLAGSHIP_ARCHITECTURE.md`, then show `src/quantlab/market_making/replay.py`. Explain:

- Stale observations under latency
- Queue-ahead depletion before fills
- Partial fills and inventory limits
- Maker/taker fees
- Forced liquidation
- Independent fill-ledger accounting

## 3:00-4:10 - Results and failure cases

Run:

```bash
jupyter notebook notebooks/start_here_market_making.ipynb
```

Show the policy table and sensitivity chart. Explicitly state that unfavorable or unstable results are retained and that the public sample validates the pipeline rather than profitability.

## 4:10-4:40 - Governance

Show the generated experiment `record.json` and explain the config fingerprint, dataset fingerprint, Git commit, seed, command, artifact hashes, and append-only directory behavior.

## 4:40-5:00 - Close

Show the paper and say:

> The next empirical milestone is a licensed multi-session feed with true receive timestamps. The software, chronology, audit trail, and failure-reporting path are ready for that dataset.

End on the one-command workflow:

```bash
make fetch-order-book-data reproduce-market-making-sample
```
