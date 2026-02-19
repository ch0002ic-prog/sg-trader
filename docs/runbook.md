# Operational Runbook

## Daily

- Run `python main.py` (or scheduler) to produce the latest ledger-native allocation proposal.
- If using `STRATEGY_PROFILE_DEFAULT` in `.env`, remember CLI `--strategy-profile` takes precedence when explicitly provided.
- Run `python main.py --healthcheck` before market open for environment sanity.
- Confirm `reports/ledger_universe_allocation.json` is refreshed.
- Spot-check extracted universe with `python main.py --list-tickers`.
- If execution replay is planned, verify an approval file exists under `reports/execution_plans/`.

## Weekly

- Generate portfolio health snapshots:
	- `python main.py --portfolio-dashboard`
	- `python main.py --portfolio-dashboard --portfolio-skip-recent`
- Review execution replay entries for broker overrides and hash/approval integrity.
- Review artifacts from GitHub Actions workflow `Weekly Walkforward Scan` (`weekly-walkforward-scan`) and confirm profile ranking has not materially drifted.

## Monthly

- Run the governance checklist in [docs/governance_checklist.md](docs/governance_checklist.md).
- Run the release checklist in [docs/release_checklist.md](docs/release_checklist.md) before major merges/promotions.
- Archive `reports/` outputs for the month.
- Run retention helper (dry-run, then apply) to archive older untracked runtime report files:
	- `bash scripts/reports_retention.sh --keep 20 --dry-run`
	- `bash scripts/reports_retention.sh --keep 20 --apply`
- Run monthly profile concentration guardrails (using non-mutating `/tmp` reports):
	- Generate profile snapshots:
		- `python main.py --strategy-profile normal --lookback-days 63 --no-log --report-path /tmp/sg_trader_strategy_normal.json`
		- `python main.py --strategy-profile defensive --lookback-days 63 --no-log --report-path /tmp/sg_trader_strategy_defensive.json`
		- `python main.py --strategy-profile aggressive --lookback-days 63 --no-log --report-path /tmp/sg_trader_strategy_aggressive.json`
	- Compute concentration metrics (`top3_concentration`, `effective_n = 1/sum(w^2)`) and alert if:
		- `top3_concentration > 0.70`, or
		- `effective_n < 4.50`
- Record current allocation settings used in production (`--lookback-days`, `--risk-free-rate`, `--max-weight`, `--top-n`).
- Confirm branch protection requires both checks from [.github/workflows/ci-smoke.yml](.github/workflows/ci-smoke.yml): `smoke` and `unit-gates`.

## Ad hoc

- Approve a plan before replay:
	- `python main.py --execution-approve <plan_id_or_path> --execution-approve-reason "reviewed"`
- Replay an approved plan deterministically:
	- `python main.py --execution-replay <plan_id_or_path> --paper-seed 1`
- For incident recovery only, skip replay risk checks:
	- `python main.py --execution-replay <plan_id_or_path> --paper-seed 1 --execution-replay-skip-risk`
- List supported brokers:
	- `python main.py --list-brokers`
- Run consolidated CI probe:
	- `bash scripts/ci_smoke.sh`
- Run focused unit gates:
	- `bash scripts/unit_gates.sh`
- Run full local pre-merge CI:
	- `bash scripts/local_ci.sh`
- Run strict local pre-merge CI (enforce smoke JSON `ok=true`):
	- `bash scripts/local_ci.sh --strict`
- Run local pre-merge CI with machine-readable summary output:
	- `bash scripts/local_ci.sh --json`
	- `bash scripts/local_ci.sh --strict --json`
- Parse machine-readable summary to compact status text:
	- `bash scripts/local_ci.sh --strict --json | python scripts/local_ci_parse.py`
- In GitHub Actions (`unit-gates`), diagnostics payload is saved as `reports/local_ci_result.json` and uploaded as artifact `local-ci-diagnostics`.
- In GitHub Actions (`smoke`), smoke payload is saved as `reports/ci_smoke_summary.json` and uploaded as artifact `ci-smoke-summary`.
- Summarize both artifacts in one line:
	- `python scripts/print_ci_artifact_summary.py --smoke-path reports/ci_smoke_summary.json --local-ci-path reports/local_ci_result.json`
- `unit-gates` prints this combined summary line directly in the workflow log for quick triage.

## Exit Codes (for CI and Ops)

- `0`: success
- `2`: invalid allocator input
- `3`: no tradable tickers in ledger
- `4`: no computable metrics from market data
- `5`: plan/approval/dashboard validation failure
- `6`: execution replay validation/execution failure
- `8`: replay blocked by risk gate
- `9`: healthcheck failure
