# Frozen Defensive-Momentum Paper Protocol

## Purpose

This protocol prospectively evaluates the exploratory monthly row from the defensive-momentum study. The historical monthly result was inspected after the broader frequency grid, so it is not validated evidence. Freezing it now prevents future parameter changes from being mistaken for a clean test.

This is paper research. It does not submit orders or recommend leverage.

## Frozen Model

- Strategy ID: `defensive-momentum-monthly-v1`.
- Assets: QQQ, GLD, and TLT; residual cash may be negative when leverage exceeds 1x.
- Signal: highest 252-session momentum among assets above their 200-session moving average.
- Risk: 25% annualized target using 63 completed returns, capped at 1.5x.
- Schedule: the final completed session of each calendar month supplies the signal.
- Execution: the target becomes effective no earlier than the following session's open.
- Friction: 10 bps times gross weight turnover and 1% annual drag on exposure above 1x.

Any parameter change requires a new strategy ID and ledger. The historical broad-grid failure remains unchanged.

## Genesis Decision

The genesis record was created at `2026-07-13T21:54:11.010175+00:00`, after the July 13 completed close and before July 14 returns existed. It uses the June 30 completed month-end signal:

- QQQ momentum: 35.0046%; above its 200-session average.
- QQQ annualized volatility: 23.7601%.
- GLD and TLT: ineligible because each was below its 200-session average.
- July 14 paper target: 105.2183% QQQ and -5.2183% cash financing.
- Record hash: `c18dfa9c2e4512338f37bec87a0104b65638fd833b7275a3a5fda4868c7df1ce`.

This record proves only that the decision existed before the return. It does not prove profitability.

## Integrity Controls

The exact 320-session input snapshot and frozen configuration are SHA-256 linked from the append-only JSONL record. The validator recomputes the record hash and chain, requires increasing as-of sessions, and rejects duplicates or alterations. Current Yahoo adjusted closes for QQQ, GLD, and TLT were independently reconciled with Nasdaq completed closes; recording fails above 5 bps disagreement.

Future rows are ignored when recomputing an as-of decision. During an unfinished month, the calculation uses the last completed prior month-end rather than treating the current session as month-end.

## Recording Command

Run after a completed month-end signal is available and specify the actual next trading session explicitly:

```powershell
python examples/record_defensive_momentum_paper.py `
  --source data/real/defensive_momentum_ohlc.csv `
  --source-metadata data/real/defensive_momentum_ohlc.metadata.json `
  --snapshot data/paper/defensive_momentum_inputs_YYYY-MM-DD.csv `
  --metadata data/paper/defensive_momentum_inputs_YYYY-MM-DD.metadata.json `
  --ledger paper/defensive_momentum_decisions.jsonl `
  --effective-session YYYY-MM-DD `
  --config config/defensive_momentum_paper.json
```

## Evidence Gate

The next required artifact is completed-session outcome scoring using adjusted OHLC and the same next-open convention. A handful of sessions cannot establish a 20% process. Reporting must retain drawdown, leverage, financing, turnover, source failures, and all missing observations alongside return.
