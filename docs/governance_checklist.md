# Governance Checklist

## Monthly Review

- Confirm ledger integrity: validate JSON parse and schema consistency of `fortress_alpha_ledger.json`.
- Verify daily heartbeat coverage for the month and confirm entry types are populated.
- Review current tradable universe using `python main.py --list-tickers`.
- Reconcile execution approvals and replays under `reports/execution_plans/`.
- Generate and review portfolio dashboard outputs using `python main.py --portfolio-dashboard`.
- Confirm data freshness thresholds and no stale data alerts.
- Review execution logs for missing correlation IDs and module tags.
- Archive monthly report outputs relevant to current flow (`reports/ledger_universe_allocation.json`, `reports/portfolio_dashboard_all.*`).
- Record current allocator settings and operational overrides.
- Verify repository branch protection requires passing GitHub checks `smoke` and `unit-gates` before merge.

## Change Log

- Date: 2026-02-09
- Change summary: Added strategy comparison summary page and documented tuned zscore throttle defaults.
- Files/modules touched: docs/strategy_comparison.md, docs/runbook.md
- Config or env changes: None
- Allocation run parameters: Not recorded
- Replay approvals audited: Not recorded
- Rollback plan: Revert doc updates
- Approver: TBD

- Date:
- Change summary:
- Files/modules touched:
- Config or env changes:
- Allocation run parameters:
- Replay approvals audited:
- Rollback plan:
- Approver:
