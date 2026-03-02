# Operational Runbook

## Daily

- Run `python main.py` (or scheduled job) and review console output.
- Confirm heartbeat and daily summary are logged.
- Run `python main.py --monitoring-report` and check for alerts.
- Review the latest monitoring markdown report in `reports/monitoring_report_YYYY-MM-DD.md`.
- If alerts fire, investigate data sources and signal spike causes.

## Weekly

- Review signal cadence: `python main.py --signal-health-report`.
- Check paper execution logs for missing module tags or correlation IDs.

## Monthly

- Generate IRAS report: `python main.py --monthly-report`.
- Run the governance checklist in [docs/governance_checklist.md](docs/governance_checklist.md).
- Archive `reports/` outputs for the month.

## Ad hoc

- Run backtest validation after parameter changes: `python main.py --backtest-validate`.
- Recommended validation band (current baseline):
	- vvix quantile ~0.95
	- alpha spread ~5.2-5.25
	- vvix safe threshold ~177.5-180
	- Latest deduped table: [reports/backtest_top_unique.md](reports/backtest_top_unique.md)
- If schema changes are made, run `python main.py --migrate-ledger`.
- Strategy defaults (zscore throttle): trend-on min leverage 0.9, trend-off VIX cap 32, zscore cut 0.5.
