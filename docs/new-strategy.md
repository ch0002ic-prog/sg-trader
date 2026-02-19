# New Strategy Spec (Ledger-Native)

## Quick Start (5 Commands)

```bash
# 1) Run allocator with defaults
python main.py

# 2) Inspect tradable ledger-derived universe
python main.py --list-tickers

# 3) Generate portfolio dashboard
python main.py --portfolio-dashboard

# 4) Generate an execution plan
python main.py --execution-plan --execution-broker paper --paper-symbol SPX_PUT --paper-side SELL --paper-qty 1 --paper-reference-price 1.25

# 5) Approve, then replay deterministically
python main.py --execution-approve <plan_id_or_path> --execution-approve-reason "reviewed"
python main.py --execution-replay <plan_id_or_path> --paper-seed 1
```

For CI automation wrappers and machine-readable command flows, see `README.md` and `docs/runbook.md`.

## Purpose

This document defines the current production strategy workflow for `sg-trader`.
It replaces exploratory notes with an implementation spec aligned to the active code path.

Primary objective: propose allocations that optimize growth relative to a $1 baseline while preserving operational controls.

## Scope

In scope:
- Ledger-native ticker universe construction
- Risk-adjusted scoring and capped allocation proposals
- Execution governance via plan approval and replay
- Operational dashboards and controls

Out of scope:
- Direct live order routing
- Broker API development
- Legacy backtest/monitoring command families removed from `main.py`

## Non-Negotiable Constraint

Ticker universe must be sourced solely from `fortress_alpha_ledger.json`.

No external allowlists are permitted in the strategy runtime.

## System Entry Point

Current entrypoint: `main.py`.

Modes:
1. Allocation mode (default): computes proposal report from ledger-derived symbols
2. Execution governance mode: approval/replay utilities
3. Operational reporting mode: portfolio dashboard generation

## Data Contract

### Universe Extraction

Universe extraction reads:
- top-level `ticker`
- nested `details` recursively for keys like `ticker` and `symbol`

Symbols classified as synthetic/non-tradable are excluded from allocation (for example `PORTFOLIO`, `SPX_PUT`, `S-REIT_BASKET`, `N/A`).

### Market Data

For each tradable ticker, close series are fetched using existing market utilities with cache fallback.

Minimum history requirement: `lookback_days + 1` rows.

## Allocation Logic

For each ticker:
- Compute lookback return over configured window
- Compute annualized volatility from log returns
- Compute risk-adjusted score:

$$
\text{score}_i = \frac{R_i - r_f \cdot (L/252)}{\max(\sigma_i, \epsilon)}
$$

where:
- $R_i$ = lookback return
- $r_f$ = annual risk-free rate input
- $L$ = lookback window in days
- $\sigma_i$ = annualized volatility
- $\epsilon$ = small floor to avoid divide-by-zero

Selection and sizing:
- Rank tickers by score descending
- Optionally filter by minimum score (`--min-score`)
- Optionally filter by volatility ceiling (`--max-annualized-volatility`)
- Optionally filter by lookback drawdown ceiling (`--max-lookback-drawdown`)
- Optionally apply soft score penalties (`--score-volatility-penalty`, `--score-drawdown-penalty`) to reduce preference for high-risk names without hard exclusion
- Optionally apply regime-aware defensive overlay (`--regime-aware-defaults`) that can tighten effective `top_n` and `max_weight` when median volatility is elevated or median score is weak
- Optionally apply a preset strategy profile (`--strategy-profile` in `none|normal|defensive|aggressive`) to set coherent defaults; explicit CLI flags override profile defaults
- Optional environment fallback `STRATEGY_PROFILE_DEFAULT` is used only when `--strategy-profile` is not explicitly provided
- Keep top `N`
- Convert positive scores to weights
- Apply iterative max-weight cap
- Renormalize to 1.0
- Convert to dollar allocation against `initial_wealth` (default $1)

## Execution Governance

### Plan Generation

`--execution-plan` creates a signed execution-plan artifact under `reports/execution_plans/`.
The plan includes `plan_hash` used by approval and replay validation.

### Plan Approval

`--execution-approve` validates plan payload and hash, then writes approval artifact.

### Replay

`--execution-replay` enforces:
- Plan hash validation
- Plan age limits
- Approval existence and approval hash match
- Optional approval age limit
- Broker validation (with optional override)
- Deterministic seed requirement unless explicitly bypassed
- Kill-switch guard unless `--execution-replay-skip-risk`

Replay outputs execution summary to console and uses existing broker abstractions.

## Strategy-Critical Commands

- Run allocation: `python main.py`
- Inspect ledger-derived tradable universe: `python main.py --list-tickers`
- Generate execution plan: `python main.py --execution-plan --execution-broker paper --paper-symbol SPX_PUT --paper-side SELL --paper-qty 1 --paper-reference-price 1.25`
- Approve plan: `python main.py --execution-approve <plan_id_or_path> --execution-approve-reason "reviewed"`
- Replay approved plan deterministically: `python main.py --execution-replay <plan_id_or_path> --paper-seed 1`
- Generate portfolio dashboard: `python main.py --portfolio-dashboard`

For operational health checks, CI wrappers, machine-readable endpoints, and diagnostics commands, refer to `README.md` and `docs/runbook.md`.

## Outputs

Core artifacts:
- `reports/ledger_universe_allocation.json`
- `reports/execution_plans/execution_plan_<id>.json`
- `reports/execution_plans/execution_approval_<id>.json`
- `reports/portfolio_dashboard_all.json`
- `reports/portfolio_dashboard_all.md`
- `reports/portfolio_dashboard_all.csv`
- `reports/portfolio_dashboard_all_correlations.csv`

## Validation Baseline

Operational baseline to keep green:
- `bash scripts/unit_gates.sh`

Optional strategy tuning support:
- `python scripts/walkforward_profile_scan.py --profiles normal,defensive,aggressive --lookback-days 63 --forward-days 21 --windows 6`

## Current Tuning Snapshot (2026-02-19)

Multi-lookback walk-forward scans (10 windows each):

```bash
# 63-day lookback
python scripts/walkforward_profile_scan.py \
	--profiles normal,defensive,aggressive \
	--lookback-days 63 \
	--forward-days 21 \
	--windows 10

# 126-day lookback
python scripts/walkforward_profile_scan.py \
	--profiles normal,defensive,aggressive \
	--lookback-days 126 \
	--forward-days 21 \
	--windows 10 \
	--out-csv reports/walkforward_profile_scan_lb126.csv \
	--out-md reports/walkforward_profile_scan_lb126.md \
	--out-detail-csv reports/walkforward_profile_scan_lb126_detail.csv

# 252-day lookback
python scripts/walkforward_profile_scan.py \
	--profiles normal,defensive,aggressive \
	--lookback-days 252 \
	--forward-days 21 \
	--windows 10 \
	--out-csv reports/walkforward_profile_scan_lb252.csv \
	--out-md reports/walkforward_profile_scan_lb252.md \
	--out-detail-csv reports/walkforward_profile_scan_lb252_detail.csv
```

Observed ranking summary:
- 63-day: `aggressive` > `defensive` > `normal`
- 126-day: `defensive` > `normal` > `aggressive`
- 252-day: `aggressive` > `normal` > `defensive`
- Stability points (3/2/1 by average-return rank): `aggressive=7`, `defensive=6`, `normal=5`

Current recommendation (evidence-based, subject to periodic re-scan):
- Keep `--strategy-profile aggressive` as default for now.
- Optional runtime default without CLI flag: set `STRATEGY_PROFILE_DEFAULT=aggressive` in environment/.env (CLI `--strategy-profile` still takes precedence).
- Switch rule: change default to `defensive` only if two consecutive scheduled scans show `defensive` leading both 63-day and 126-day lookbacks.

### Monthly Checkpoint (2026-02-20)

Latest non-mutating monthly check (10 windows per lookback):
- 63-day: `aggressive` > `defensive` > `normal`
- 126-day: `defensive` > `normal` > `aggressive`
- 252-day: `aggressive` > `normal` > `defensive`
- Stability points (3/2/1 by average-return rank): `aggressive=7`, `defensive=6`, `normal=5`

Concentration guardrails (all pass):
- `normal`: top3 concentration `0.6787`, effective_n `4.73` (OK)
- `defensive`: top3 concentration `0.6724`, effective_n `4.86` (OK)
- `aggressive`: top3 concentration `0.6521`, effective_n `5.13` (OK)

Decision: keep `aggressive` default (switch rule not triggered).

Operationalization note:
- Use `bash scripts/monthly_strategy_check.sh` to generate the monthly decision pack automatically.
- Guardrail alerting is enforced by `scripts/monthly_strategy_guardrail_alert.py` (monthly pack exits non-zero on breach unless soft-fail mode is used).

This strategy spec intentionally does not duplicate CI workflow wiring details.
For CI diagnostics/summary behavior, refer to `README.md` and `docs/runbook.md`.

## Risk Notes

This repository remains a decision-support and paper-execution framework.

Before any production capital deployment:
- Add separate operational risk controls
- Add independent reconciliation checks
- Add strict secrets/config management
- Add environment-level alerting and audit trails

## Change Control

Any future strategy changes must preserve:
1. Ledger-only ticker universe rule
2. Backward-compatible execution governance semantics unless explicitly versioned
3. Deterministic replay behavior for auditable operations
