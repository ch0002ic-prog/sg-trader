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
- Keep top `N`
- Convert positive scores to weights
- Apply iterative max-weight cap
- Renormalize to 1.0
- Convert to dollar allocation against `initial_wealth` (default $1)

## Execution Governance

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
- `reports/execution_plans/execution_approval_<id>.json`
- `reports/portfolio_dashboard_all.json`
- `reports/portfolio_dashboard_all.md`

## Validation Baseline

Operational baseline to keep green:
- `bash scripts/unit_gates.sh`

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
