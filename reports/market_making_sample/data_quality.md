# Market-Data Quality Report

## Verdict

**PASS** — AAPL on 2012-06-21. This report describes data integrity and reconstruction only; it is not strategy evidence.

## Integrity

| Metric | Value |
| --- | ---: |
| Canonical messages | 301,587 |
| Synchronized snapshots | 301,587 |
| Duration | 6.50 hours |
| Sequence gaps | 0 |
| Duplicate sequences | 0 |
| Non-monotonic timestamps | 0 |
| Duplicate timestamps | 14,194 |
| Crossed snapshots | 0 |
| Reconstruction match rate | 89.17% |
| Reconstruction mismatches | 32,673 |
| Invalid cancel/execution attempts | 0 |

## Market description

| Metric | Value |
| --- | ---: |
| Median spread | 0.1500 |
| 99th-percentile spread | 0.3700 |
| Median top-of-book depth | 205.00 |

## Event counts

| Event | Count |
| --- | ---: |
| delete | 120,451 |
| hidden_execution | 11,332 |
| partial_cancel | 2,324 |
| submission | 143,822 |
| visible_execution | 23,658 |

## Interpretation

LOBSTER Level-N samples contain synchronized top-N snapshots but not the full depth outside the requested range. A level can therefore enter the visible range without its original resting order having appeared in the truncated message history. The replay deliberately records and reseeds those boundary mismatches instead of silently treating the truncated feed as a complete L3 book.
