# Leveraged Trend Execution-Timing Audit

Frozen parameters; signals use completed closes and next-open returns use adjusted OHLC.

| period | convention | cagr | sharpe | max_drawdown | observations |
| --- | --- | --- | --- | --- | --- |
| train | next_open | 0.113628 | 0.524449 | 0.347164 | 1510.000000 |
| train | close_to_close | 0.098548 | 0.472898 | 0.376571 | 1510.000000 |
| validation | next_open | 0.413105 | 1.306496 | 0.334927 | 1007.000000 |
| validation | close_to_close | 0.401444 | 1.287045 | 0.329156 | 1007.000000 |
| holdout | next_open | 0.255255 | 0.960074 | 0.230589 | 1386.000000 |
| holdout | close_to_close | 0.228396 | 0.883912 | 0.242183 | 1386.000000 |
| full | next_open | 0.235634 | 0.887492 | 0.347164 | 3903.000000 |
| full | close_to_close | 0.217124 | 0.832739 | 0.376571 | 3903.000000 |
