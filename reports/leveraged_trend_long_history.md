# Leveraged Trend Long-History Falsification

Real QQQ adjusted closes and FRED DFF rates; pre-inception 3x returns are explicitly reconstructed, not observed.

**Long-history 20% CAGR threshold: NOT MET.**

## Period Results

| period | cagr | sharpe | max_drawdown | observations |
| --- | --- | --- | --- | --- |
| dotcom_and_gfc | 0.0395801 | 0.289212 | 0.39058 | 2515 |
| pre_tqqq | 0.0270126 | 0.232022 | 0.39058 | 2542 |
| tqqq_pre_holdout | 0.225866 | 0.84986 | 0.377527 | 2742 |
| published_holdout | 0.254211 | 0.956858 | 0.236349 | 1385 |
| full_history | 0.151345 | 0.671383 | 0.39058 | 6669 |

## Reconstruction Reconciliation

- **Overlap Observations**: 4126
- **Daily Return Correlation**: 0.99894
- **Actual Minus Synthetic Annual Mean**: -0.0237977
- **Annualized Tracking Error**: 0.0295165

## Drag Sensitivity

| annual_drag | cagr | sharpe | max_drawdown | observations |
| --- | --- | --- | --- | --- |
| 0.0095 | 0.151345 | 0.671383 | 0.39058 | 6669 |
| 0.015 | 0.151854 | 0.675317 | 0.431008 | 6669 |
| 0.025 | 0.143086 | 0.646605 | 0.480569 | 6669 |
| 0.04 | 0.139593 | 0.636781 | 0.504192 | 6669 |
