# Strategy Comparison Summary

Generated: 2026-02-09

Note: This page is a historical snapshot from the legacy research pipeline and is retained for reference only.
Current production workflow uses the ledger-native allocator in `main.py`.

Source report: [reports/strategy_comparison_2026-02-09.md](../reports/strategy_comparison_2026-02-09.md)

## Common Window (apples-to-apples)
- Window: 2017-10-30 -> 2026-02-05

| Strategy | Sharpe | Final Equity | CAGR | Vol | Max DD | Calmar | Data Points |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zscore_throttle_tuned | 0.52 | 4.0904 | 18.56% | 16.06% | -26.71% | 0.69 | 2078 |
| zscore_throttle_cut50 | 0.52 | 3.9671 | 18.12% | 15.73% | -26.25% | 0.69 | 2078 |
| regime_blend_ief20 | 0.39 | 2.0214 | 8.88% | 17.30% | -40.95% | 0.22 | 2022 |
| regime_blend_vix29 | 0.34 | 1.8739 | 7.89% | 17.00% | -39.32% | 0.20 | 2022 |

## Full Window (per strategy)

| Strategy | Start | End | Sharpe | Final Equity | CAGR | Vol | Max DD | Calmar | Data Points |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zscore_throttle_tuned | 2013-01-02 | 2026-02-05 | 0.57 | 4.0904 | 11.35% | 14.51% | -26.71% | 0.42 | 3294 |
| zscore_throttle_cut50 | 2013-01-02 | 2026-02-05 | 0.56 | 3.9671 | 11.09% | 14.26% | -26.25% | 0.42 | 3294 |
| regime_blend_ief20 | 2017-10-30 | 2026-02-05 | 0.39 | 2.0214 | 8.88% | 17.30% | -40.95% | 0.22 | 2022 |
| regime_blend_vix29 | 2017-10-30 | 2026-02-05 | 0.34 | 1.8739 | 7.89% | 17.00% | -39.32% | 0.20 | 2022 |
