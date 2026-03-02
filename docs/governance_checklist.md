# Governance Checklist

## Monthly Review

- Confirm ledger integrity: run `python main.py --dedup-ledger` and check for corruption backups.
- Verify daily heartbeat coverage for the month and confirm entry types are populated.
- Review signal cadence using `python main.py --signal-health-report`.
- Review monitoring alerts from `python main.py --monitoring-report` and investigate spikes.
- Reconcile paper PnL with attribution: `python main.py --pnl-dashboard` and `python main.py --attribution-report`.
- Confirm data freshness thresholds and no stale data alerts.
- Review execution logs for missing correlation IDs and module tags.
- Archive monthly report outputs (`reports/iras_report_YYYY-MM.md` and `.json`).
- Record the current validation baseline (vvix quantile ~0.95, alpha ~5.2-5.25, vvix ~177.5-180) and note any deviations.

## Change Log

- Date: 2026-02-09
- Change summary: Added strategy comparison summary page and documented tuned zscore throttle defaults.
- Files/modules touched: docs/strategy_comparison.md, docs/runbook.md
- Config or env changes: None
- Backtest validation run (grid, quantiles): Not run
- Monitoring alerts observed: None
- Rollback plan: Revert doc updates
- Approver: TBD

- Date:
- Change summary:
- Files/modules touched:
- Config or env changes:
- Backtest validation run (grid, quantiles):
- Monitoring alerts observed:
- Rollback plan:
- Approver:
