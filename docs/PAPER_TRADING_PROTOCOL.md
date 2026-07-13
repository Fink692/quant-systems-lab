# Frozen Leveraged-Trend Paper Protocol

## Purpose

This protocol creates prospective evidence for `leveraged-trend-v1`. Historical results are insufficient to establish persistent 20% returns, so the model parameters are frozen and each next-session exposure is recorded before its return is known.

This is paper research. It does not submit orders or recommend that capital be traded.

## Frozen Model

- Risky instrument: TQQQ.
- Cash instrument: BIL.
- Trend rule: TQQQ adjusted close above its 200-session moving average.
- Risk rule: 30% target annualized volatility using the prior 21 completed returns.
- Exposure bounds: 0% to 100% TQQQ; residual exposure in BIL.
- Timing: a completed-session signal becomes effective no earlier than the following market session.
- Configuration: `config/leveraged_trend_paper.json`, whose canonical hash is stored in every decision.

Changing any field creates a new strategy identifier. Past records are never rewritten.

## Source Controls

The signal source is Yahoo Finance adjusted-close history. Every append independently queries Nasdaq.com for the completed 4:00 p.m. close of TQQQ, QQQ, and BIL. The append is rejected if:

- Nasdaq does not explicitly identify a completed close.
- The three symbols do not share one completed session.
- Yahoo does not contain that same session.
- Any latest source difference exceeds 5 bps.
- The effective session is not strictly later than the as-of session.

The Nasdaq website endpoint is an independent public cross-check, not a licensed execution feed, and may change without notice.

## Tamper Evidence

Each JSONL record includes:

- UTC creation timestamp.
- As-of and effective sessions.
- Frozen-configuration SHA-256.
- Input-snapshot SHA-256.
- Signal price, moving average, realized volatility, and target weights.
- Independent closes and source differences.
- Previous-record hash and current-record hash.

The ledger validator recomputes every hash, verifies the chain, and requires strictly increasing as-of sessions. Duplicate, older, altered, or detached records are rejected.

## First Decision

The genesis record was created at `2026-07-13T21:02:30.793027+00:00`, after Nasdaq reported the July 13 completed close and before the July 14 session:

- TQQQ close: $72.64.
- 200-session moving average: $58.040916.
- Trailing annualized volatility: 88.3878%.
- Trend state: on.
- July 14 paper target: 33.9413% TQQQ and 66.0587% BIL.
- Record hash: `56d0096707615f550ec3acf5b9265b88d51bc7312ab33a78289aa3dac574201f`.

The decision is evidence of process, not performance. It must remain unchanged after July 14 returns become available.

## Recording Command

Run only after the intended as-of session has a confirmed completed Nasdaq close:

```powershell
python examples/record_leveraged_trend_paper.py `
  --snapshot data/paper/leveraged_trend_inputs_YYYY-MM-DD.csv `
  --metadata data/paper/leveraged_trend_inputs_YYYY-MM-DD.metadata.json `
  --ledger paper/leveraged_trend_decisions.jsonl `
  --effective-session YYYY-MM-DD `
  --config config/leveraged_trend_paper.json
```

The effective session is explicit rather than guessed by a simplified holiday calendar.

## Evaluation Gate

Performance evaluation begins only after decisions and subsequent completed returns exist. A credible update will report net paper returns, turnover, modeled and quote-based slippage, drawdown, source failures, missing decisions, and every configuration change. No 20% claim should be made from a handful of forward sessions.
