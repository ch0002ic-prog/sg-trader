# sg-trader

![ci-smoke](https://github.com/ch0002ic-prog/sg-trader/actions/workflows/ci-smoke.yml/badge.svg)
![weekly-walkforward](https://github.com/ch0002ic-prog/sg-trader/actions/workflows/weekly-walkforward.yml/badge.svg)

Decision-support engine for the ergodic barbell strategy. This project fetches market data, computes signals, and logs compliance-friendly entries. Execution remains manual.

## Architecture

- `main.py` is the single CLI entry point for allocation runs, execution governance (`plan`/`approve`/`replay`), dashboard generation, and health/discovery probes.
- `fortress_alpha_ledger.json` is the only source of ticker universe truth for strategy runtime.
- `sg_trader/` contains domain modules for config, market data/signal utilities, execution adapters, logging, and dashboard/report helpers.
- `reports/` stores generated artifacts (allocation reports, execution plans/approvals, dashboard outputs, CI diagnostics JSON).
- `scripts/` contains operator/CI wrappers (`ci_smoke`, `unit_gates`, `local_ci`) and parsers/summarizers used in automation.
- `.github/workflows/ci-smoke.yml` runs two required checks: `smoke` and `unit-gates`.

## Setup

1. Create/activate your virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Configure Telegram alerts:

   ```bash
   export TELEGRAM_TOKEN="your_bot_token"
   export TELEGRAM_CHAT_ID="your_chat_id"
   ```

   Or add a `.env` file:

   ```bash
   TELEGRAM_TOKEN=your_bot_token
   TELEGRAM_CHAT_ID=your_chat_id
   TELEGRAM_TIMEOUT_SECONDS=10
   TELEGRAM_RETRIES=3
   TELEGRAM_BACKOFF_SECONDS=1
   EXECUTION_PLAN_MAX_AGE_HOURS=24
   ```
The script uses the MAS eServices Benchmark Prices & Yields page as the source for the latest 6-month T-bill yield.
If the MAS yield cannot be retrieved, the run will abort (no fallback to a static rate).
If MAS is unavailable, the last cached yield can be used if it is fresh; a staleness warning is logged when cache is used.
Startup will fail if key configuration values are invalid.

To enable drift monitoring, add your current allocation snapshot in .env:

```bash
ALLOC_FORTRESS=0.70
ALLOC_ALPHA=0.29
ALLOC_SHIELD=0.01
ALLOC_GROWTH=0.00
DRIFT_BAND=0.05
GROWTH_TICKER=QQQ
GROWTH_MA_DAYS=200

# Optional log/data tuning
DEDUP_WINDOW_SECONDS=300
MAS_MONTHS_BACK=2
CACHE_DIR=.cache
MAS_CACHE_MAX_AGE_HOURS=12
MAS_CACHE_WARN_PCT=0.8
PAPER_MAX_POSITION_QTY=5
PAPER_MAX_DAILY_LOSS=0
PAPER_KILL_SWITCH=false
PAPER_INITIAL_CAPITAL=100000
DATA_FRESHNESS_DAYS=2
MARKET_CACHE_MAX_AGE_HOURS=48
MARKET_CACHE_WARN_PCT=0.8
PAPER_MAX_DAILY_TRADES=10
PAPER_MAX_NOTIONAL=100000
PAPER_VOL_KILL_THRESHOLD=40
STRATEGY_PROFILE_DEFAULT=aggressive
PNL_DOWNSIDE_MIN_DAYS=10
MONITORING_WINDOW_DAYS=7
MONITORING_SPIKE_MULT=2
MONITORING_MIN_DAYS=3
MONITORING_REGIME_WINDOW_DAYS=60
MONITORING_REGIME_ZSCORE=2
MONITORING_REGIME_CRITICAL_ZSCORE=3
MONITORING_REGIME_MIN_POINTS=20
MONITORING_SIGNAL_DRIFT_WINDOW_DAYS=3
MONITORING_SIGNAL_DRIFT_LOW_MULT=0.5
MONITORING_ALERT_THROTTLE_HOURS=6
MONITORING_ALERTS_ENABLED=false
```

## Usage

Quick onboarding guide: [docs/new-strategy.md](docs/new-strategy.md) (see “Quick Start (5 Commands)”).

CLI discovery endpoints:

```bash
python main.py --version
python main.py --cli-capabilities-json
python main.py --healthcheck
python main.py --healthcheck-json
```

Run the ledger-native allocation workflow:

```bash
python main.py
```

This uses tickers extracted solely from `fortress_alpha_ledger.json`.

Control allocation inputs:

```bash
python main.py \
  --strategy-profile normal \
  --lookback-days 63 \
  --risk-free-rate 0.0365 \
  --min-score 0.0 \
  --max-annualized-volatility 0.35 \
  --max-lookback-drawdown 0.25 \
  --score-volatility-penalty 0.5 \
  --score-drawdown-penalty 1.0 \
  --regime-aware-defaults \
  --regime-volatility-threshold 0.30 \
  --regime-score-threshold 0.0 \
  --regime-defensive-top-n 6 \
  --regime-defensive-max-weight 0.20 \
  --max-weight 0.30 \
  --top-n 10 \
  --initial-wealth 1 \
  --report-path reports/ledger_universe_allocation.json
```

Profiles available: `none`, `normal`, `defensive`, `aggressive`.
`STRATEGY_PROFILE_DEFAULT` sets the default profile only when `--strategy-profile` is not provided.
Explicit CLI flags override profile defaults.

List extracted tradable tickers:

```bash
python main.py --list-tickers
```

Include synthetic symbols (`PORTFOLIO`, `SPX_PUT`, etc.) in ticker listing:

```bash
python main.py --list-tickers --include-synthetic
```

List available execution brokers:

```bash
python main.py --list-brokers
```

Generate an execution plan:

```bash
python main.py --execution-plan \
  --execution-broker paper \
  --paper-symbol SPX_PUT \
  --paper-side SELL \
  --paper-qty 1 \
  --paper-reference-price 1.25
```

Capture only the generated plan ID (for scripting/CI):

```bash
PLAN_ID=$(python main.py --execution-plan \
  --execution-broker paper \
  --paper-symbol SPX_PUT \
  --paper-side SELL \
  --paper-qty 1 \
  --paper-reference-price 1.25 \
  --execution-plan-id-only)
echo "$PLAN_ID"
```

Approve and capture only the approved plan ID:

```bash
APPROVED_ID=$(python main.py --execution-approve "$PLAN_ID" \
  --execution-approve-reason "ci-approval" \
  --execution-approve-id-only)
echo "$APPROVED_ID"
```

Run end-to-end CI smoke in one command (plan + approve + replay):

```bash
python main.py --execution-ci-smoke \
  --execution-broker paper \
  --paper-symbol SPX_PUT \
  --paper-side SELL \
  --paper-qty 1 \
  --paper-reference-price 1.25 \
  --paper-seed 1
```

Machine-readable CI smoke payload:

```bash
python main.py --execution-ci-smoke --execution-ci-smoke-json \
  --execution-broker paper \
  --paper-symbol SPX_PUT \
  --paper-side SELL \
  --paper-qty 1 \
  --paper-reference-price 1.25 \
  --paper-seed 1
```

Approve an execution plan JSON:

```bash
python main.py --execution-approve <plan_id_or_path> \
  --execution-approve-reason "Manual approval after review"
```

Replay an approved execution plan:

```bash
python main.py --execution-replay <plan_id_or_path> \
  --paper-seed 1 \
  --execution-approval-max-age-hours 12
```

Machine-readable replay payload:

```bash
python main.py --execution-replay <plan_id_or_path> \
  --execution-replay-json \
  --paper-seed 1
```

Replay with broker override:

```bash
python main.py --execution-replay <plan_id_or_path> \
  --paper-seed 1 \
  --execution-replay-broker paper
```

Generate portfolio dashboard outputs:

```bash
python main.py --portfolio-dashboard
python main.py --portfolio-dashboard --portfolio-start 2026-01-01 --portfolio-end 2026-02-06
python main.py --portfolio-dashboard --portfolio-skip-recent
```

Run tests:

```bash
python -m unittest
python -m unittest discover -s tests
```

CI gate (focused execution workflow checks):

```bash
bash scripts/unit_gates.sh
```

In GitHub Actions, `unit-gates` emits `reports/local_ci_result.json` (uploaded as artifact `local-ci-diagnostics`) via:

```bash
bash scripts/local_ci.sh --strict --json > reports/local_ci_result.json
python scripts/local_ci_parse.py --input reports/local_ci_result.json
```

`unit-gates` also prints a one-line combined artifact summary in the Actions log (using `reports/ci_smoke_summary.json` and `reports/local_ci_result.json`):

```bash
python scripts/print_ci_artifact_summary.py \
  --smoke-path reports/ci_smoke_summary.json \
  --local-ci-path reports/local_ci_result.json
```

One-command CI wrapper (healthcheck + execution smoke):

```bash
bash scripts/ci_smoke.sh
```

One-command local pre-merge gates (smoke + focused unit gates):

```bash
bash scripts/local_ci.sh
```

Strict mode (also validates smoke JSON `ok=true` shape before unit gates):

```bash
bash scripts/local_ci.sh --strict
```

JSON mode (emit one final machine-readable status line):

```bash
bash scripts/local_ci.sh --json
bash scripts/local_ci.sh --strict --json
```

Parse that JSON into a compact CI log line (and preserve exit code semantics):

```bash
bash scripts/local_ci.sh --strict --json | python scripts/local_ci_parse.py
```

Summarize both CI artifacts in one line:

```bash
python scripts/print_ci_artifact_summary.py \
  --smoke-path reports/ci_smoke_summary.json \
  --local-ci-path reports/local_ci_result.json
```

Archive older untracked runtime report artifacts (safe retention, dry-run by default):

```bash
bash scripts/reports_retention.sh --keep 20 --dry-run
bash scripts/reports_retention.sh --keep 20 --apply
```

Run walk-forward profile scan (strategy tuning):

```bash
python scripts/walkforward_profile_scan.py \
  --profiles normal,defensive,aggressive \
  --lookback-days 63 \
  --forward-days 21 \
  --windows 6
```

Optional environment overrides:

```bash
LEDGER_PATH=./fortress_alpha_ledger.json \
EXECUTION_BROKER=paper \
PAPER_SYMBOL=SPX_PUT \
PAPER_SIDE=SELL \
PAPER_QTY=1 \
PAPER_REFERENCE_PRICE=1.25 \
PAPER_SEED=1 \
ENFORCE_ROBUSTNESS_GATE=1 \
ROBUSTNESS_STRATEGY_PROFILE=aggressive \
ROBUSTNESS_MAX_TOP3_CONCENTRATION=0.70 \
ROBUSTNESS_MIN_EFFECTIVE_N=4.0 \
bash scripts/ci_smoke.sh
```

Regime-threshold mode (use different robustness thresholds for normal vs defensive regimes):

```bash
ENFORCE_ROBUSTNESS_GATE=1 \
ROBUSTNESS_THRESHOLD_MODE=regime \
ROBUSTNESS_MAX_TOP3_CONCENTRATION_NORMAL=0.70 \
ROBUSTNESS_MIN_EFFECTIVE_N_NORMAL=4.0 \
ROBUSTNESS_MAX_TOP3_CONCENTRATION_DEFENSIVE=0.75 \
ROBUSTNESS_MIN_EFFECTIVE_N_DEFENSIVE=3.8 \
bash scripts/ci_smoke.sh
```

Robustness calibration outputs are generated at:

- `reports/robustness_threshold_calibration_latest.json`
- `reports/robustness_threshold_calibration_latest.md`

Latest calibration recommendation (36-scenario grid):

- `ROBUSTNESS_MAX_TOP3_CONCENTRATION=0.8644`
- `ROBUSTNESS_MIN_EFFECTIVE_N=3.4433`

Use these as promotion-pipeline starting points if current defaults are too strict for your scenario mix.

Investigation cadence recommendation:

- Always run Tier-1 checks (`bash scripts/ci_smoke.sh` + focused unit tests).
- Run deep calibration grids only when guardrails are unstable, for example when robustness breaches happen for 2+ consecutive promotion runs.

Write combined JSON summary to a file (useful in CI artifacts):

```bash
CI_SMOKE_SUMMARY_PATH=reports/ci_smoke_summary.json bash scripts/ci_smoke.sh
```

## Exit Codes

`main.py` uses stable non-zero exit codes for automation:

- `0`: success
- `2`: invalid allocator argument (for example bad `--lookback-days`, `--top-n`, `--max-weight`, `--initial-wealth`)
- `3`: no tradable tickers extracted from ledger
- `4`: ticker metrics could not be computed (insufficient/missing market data)
- `5`: plan/approval/dashboard validation error (input/payload/hash/date issues)
- `6`: execution replay validation or broker execution error
- `7`: allocator robustness gate failed (top-3 concentration and/or effective-n threshold breach)
- `8`: risk-gate blocked execution replay/CI smoke (for example kill switch)
- `9`: healthcheck failed

## Outputs

- Ledger file: `fortress_alpha_ledger.json`
- Allocation report: `reports/ledger_universe_allocation.json` (or custom `--report-path`)
- Portfolio dashboard files when requested:
  - `reports/portfolio_dashboard_all.json`
  - `reports/portfolio_dashboard_all.md`
  - `reports/portfolio_dashboard_all.csv`
  - `reports/portfolio_dashboard_all_correlations.csv`

Note: `external` is a stub adapter that must be implemented before live execution.

Key flag:
- `--no-log` disables ledger writes for allocation proposal mode.

## Notes

This is a decision-support tool. It does not place trades automatically.

## Governance

See the governance checklist in [docs/governance_checklist.md](docs/governance_checklist.md).
Release checklist: [docs/release_checklist.md](docs/release_checklist.md)

Operational runbook: [docs/runbook.md](docs/runbook.md)
Strategy comparison summary: [docs/strategy_comparison.md](docs/strategy_comparison.md)
