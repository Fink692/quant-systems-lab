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
- Execution: the target changes at the effective session's open; the prior target earns any intervening overnight return.
- Cost: 10 bps times absolute TQQQ-weight turnover, charged at the effective open.
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

Completed outcomes are stored in a separate hash-chained ledger. Validation links every outcome to its frozen decision and recomputes exposure, overnight and intraday return, turnover, cost, and net return from the recorded adjusted OHLC values. A rehashed but mathematically inconsistent outcome is rejected.

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

## Outcome Scoring

After the effective session has a unique completed Nasdaq historical row, score the oldest pending decision:

```powershell
python examples/score_leveraged_trend_paper.py `
  --decisions paper/leveraged_trend_decisions.jsonl `
  --outcomes paper/leveraged_trend_outcomes.jsonl `
  --total-cost-bps 10
```

The genesis outcome begins in cash immediately before the effective open, so it has no pre-entry overnight return. Later outcomes hold the previous target from its recorded close to the new effective open, switch to the new target at that open, then earn the new target's intraday return. The scorer refuses missing, duplicate, future, or unlinked sessions. As of July 13, 2026, the July 14 genesis outcome does not yet exist and must not be inferred.

## Execution-Timing Audit

The frozen historical strategy was recomputed using executable next-open timing and adjusted TQQQ/BIL OHLC through July 13, 2026. On the untouched 2021 onward holdout, next-open CAGR was **25.53%**, Sharpe was **0.96**, and maximum drawdown was **23.06%** after the same 10 bps turnover cost. The corresponding close-to-close CAGR was 22.84%. See `reports/leveraged_trend_execution_timing.md` and its checksum-tracked input snapshot in `data/paper/`.

## Evaluation Gate

Performance evaluation begins only after decisions and subsequent completed returns exist. A credible update will report net paper returns, turnover, modeled and quote-based slippage, drawdown, source failures, missing decisions, and every configuration change. No 20% claim should be made from a handful of forward sessions.
