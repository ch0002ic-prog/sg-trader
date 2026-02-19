# V2 Promotion Verification

1) Confirm package files exist
- reports/v2_promotion_selection.json
- reports/v2_active_policy_overrides.csv
- reports/v2_promotion_package.md

2) Confirm winner matches leaderboard rank 1
- Compare winner.label in reports/v2_promotion_selection.json
- Against rank=1 label in reports/regime_optimizer_v2_safe_leaderboard.csv

3) Confirm active overrides source
- active csv should match reports/regime_optimizer_v224_ndvol120_safe_overrides.csv

4) Optional re-run checks
- scripts/regime_portfolio_optimizer_v2.py (same args as safe run)
- reports/regime_optimizer_override_necessity_summary.csv
