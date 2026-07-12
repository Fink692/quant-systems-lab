# Assumptions and Limitations Register

## Flagship market-making research

| ID | Assumption or limitation | Consequence | Control |
| --- | --- | --- | --- |
| MM-001 | LOBSTER public samples expose synchronized Level-N state, not complete depth outside N levels. | A level may enter the observed range without its original queue history. | Count reconciliation mismatches, reseed explicitly, and do not claim exact L3 queue priority from the sample. |
| MM-002 | Exchange timestamps are used as receive timestamps in the public sample. | Empirical feed latency cannot be estimated. | Treat latency as a predeclared sensitivity and require true receive timestamps for the final paper. |
| MM-003 | Queue position is unknown in Level-2 or truncated data. | Fill estimates depend on an assumption. | Report the complete configured queue-ahead grid. |
| MM-004 | Agent orders are counterfactual and do not alter the historical market. | Large simulated orders would violate the small-agent assumption. | Cap order size relative to displayed depth and report participation. |
| MM-005 | A single demonstration session cannot establish persistent profitability. | Bootstrap intervals describe intraday path variation only. | Label sample results as pipeline validation and require many untouched sessions for the paper. |
| MM-006 | Fees and rebates vary by venue, tier, and date. | Net PnL can be materially misstated. | Freeze the documented fee schedule and run adverse fee scenarios. |
| MM-007 | Hidden liquidity and priority rules are incompletely observed. | Real fills may differ from queue replay. | Report hidden executions separately and compare conservative queue assumptions. |
| MM-008 | End-of-session positions are forcibly liquidated at the displayed touch. | Tail costs may still be optimistic during stressed liquidity. | Attribute liquidation separately and add stressed liquidation scenarios before publication. |

## General research controls

- Results are research evidence, not investment advice or a production execution system.
- Synthetic tests prove deterministic software behavior, not empirical validity.
- Every published result must identify its dataset hash, configuration fingerprint, code commit, seed, exclusions, and limitations.
- Any test-period-driven model change creates a new experiment and requires a later untouched test period.
