# Defensive Multi-Asset Momentum Exploration

Real adjusted QQQ/GLD/TLT OHLC and official FRED rates; completed-close signals execute at the next open.

**All-grid selected evaluation 20% CAGR threshold: NOT MET.**

## Development-Selected Parameters

- Momentum: 63 sessions.
- Trend: 200 sessions.
- Target volatility: 25%.
- Maximum leverage: 1.5x.
- Rebalance: weekly.

## Selected-Model Period Results

| period | cagr | sharpe | max_drawdown | observations |
| --- | --- | --- | --- | --- |
| development | 0.147869 | 0.728649 | 0.406150 | 3021.000000 |
| evaluation | 0.080582 | 0.442121 | 0.415987 | 2393.000000 |
| full | 0.117626 | 0.595970 | 0.415987 | 5414.000000 |

## Best Monthly Subfamily Row

| momentum_days | trend_days | target_volatility | max_leverage | rebalance_frequency | development_cagr | development_sharpe | development_max_drawdown | development_observations | evaluation_cagr | evaluation_sharpe | evaluation_max_drawdown | evaluation_observations | full_cagr | full_sharpe | full_max_drawdown | full_observations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 252 | 200 | 0.250000 | 1.500000 | monthly | 0.138549 | 0.673427 | 0.460715 | 3021.000000 | 0.300123 | 1.191253 | 0.271366 | 2393.000000 | 0.207329 | 0.908748 | 0.460715 | 5414.000000 |

The monthly row is exploratory: it was observed after the broader weekly/monthly evaluation was inspected. It must not replace the all-grid selection or be called prospective evidence.

## Full-History Cost Sensitivity

| cost_bps | cagr | sharpe | max_drawdown | observations |
| --- | --- | --- | --- | --- |
| 5.000000 | 0.130363 | 0.644964 | 0.391863 | 5414.000000 |
| 10.000000 | 0.117626 | 0.595970 | 0.415987 | 5414.000000 |
| 25.000000 | 0.080170 | 0.448458 | 0.496516 | 5414.000000 |
| 50.000000 | 0.020199 | 0.203246 | 0.713584 | 5414.000000 |
