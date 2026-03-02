# sg-trader

![tests](https://github.com/OWNER/REPO/actions/workflows/tests.yml/badge.svg)

Replace `OWNER/REPO` with your GitHub org/repo to activate the badge.

Decision-support engine for the ergodic barbell strategy. This project fetches market data, computes signals, and logs compliance-friendly entries. Execution remains manual.

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

Run the daily signal check:

```bash
python main.py
```

Scheduler-friendly CLI options:

```bash
python main.py \
  --alpha-spread-threshold 5 \
  --vvix-safe-threshold 110 \
  --reit-spread-threshold 3.1 \
  --risk-free-rate 3.65 \
  --dte 180 \
   --shield-dte-remaining 45 \
   --shield-profit-multiple 6.5 \
  --dry-run

```

Generate an auditable execution plan (no execution, includes risk checks):

```bash
python main.py --execution-plan \
   --paper-symbol SPX_PUT \
   --paper-side SELL \
   --paper-qty 1 \
   --paper-reference-price 1.25 \
   --paper-slippage-bps 5
```

Approve a saved execution plan (logs approval + writes approval JSON):

```bash
python main.py --execution-approve <plan_id_or_path> \
   --execution-approve-reason "Manual approval after review"
```

Execute a paper trade only when a matching plan is approved:

```bash
python main.py --paper-execution \
   --execution-plan-id <plan_id> \
   --execution-plan-max-age-hours 24 \
   --execution-approval-max-age-hours 12 \
   --execution-plan-slippage-tolerance-bps 2 \
   --execution-plan-latency-tolerance-ms 5 \
   --paper-symbol SPX_PUT \
   --paper-side SELL \
   --paper-qty 1 \
   --paper-reference-price 1.25 \
   --paper-post-trade-reports \
   --paper-post-trade-monthly
```

Replay an approved execution plan (paper execution) and refresh reports:

```bash
python main.py --execution-replay <plan_id_or_path> \
   --paper-seed 1 \
   --execution-approval-max-age-hours 12 \
   --paper-mark-price 1.25 \
   --paper-post-trade-reports \
   --paper-post-trade-monthly \
   --report-month 2026-02
```

Override the broker during replay:

```bash
python main.py --execution-replay <plan_id_or_path> \
   --execution-replay-broker paper
```

Note: a warning like "Execution replay broker override: <plan> -> <override>" is emitted
whenever the replay broker differs from the plan broker.

Skip execution risk checks during replay (not recommended outside incident recovery):

```bash
python main.py --execution-replay <plan_id_or_path> \
   --paper-seed 1 \
   --execution-replay-skip-risk
```

Allow replay without a deterministic seed:

```bash
python main.py --execution-replay <plan_id_or_path> \
   --execution-replay-allow-random
```

Log a paper PnL snapshot using a mark price:

```bash
python main.py --paper-pnl-snapshot \
   --paper-symbol SPX_PUT \
   --paper-mark-price 1.24
```

One-time ledger cleanup (daily de-dup):

```bash
python main.py --dedup-ledger
```

Backfill entry types on historical ledger entries:

```bash
python main.py --backfill-entry-types
```

Backfill module attribution for execution entries:

```bash
python main.py --backfill-exec-modules
```

Rewrite today's daily summary entry:

```bash
python main.py --refresh-daily-summary
```

Refresh today's summary and regenerate the report:

```bash
python main.py --refresh-daily-summary
python main.py --monthly-report
```

Run the backtest (VectorBT required):

```bash
python main.py --backtest
```

Run backtest validation (stress tests + sensitivity):

```bash
python main.py --backtest-validate
python main.py --backtest-validate --bt-alpha-grid 3,5,7 --bt-vvix-grid 90,110,130
python main.py --backtest-validate --bt-use-primary
python main.py --backtest-validate --bt-run-tag baseline_2026_02_06
python main.py --backtest-validate --bt-max-drawdown 0.25
```

Run the standard validation suite (primary + tight + min-trades variants):

```bash
python main.py --validation-suite
python main.py --validation-suite --suite-tag suite_2026_02_06
```

Refresh validation + slippage + attribution + monthly reports in one run:

```bash
python main.py --refresh-reports
python main.py --refresh-reports --bt-use-primary --bt-run-tag nightly_2026_02_06
```

Summarize sweep results (top-N, optional dedupe/plot):

```bash
python scripts/backtest_sweep_summary.py --top 10 --min-trades 40
python scripts/backtest_sweep_summary.py --dedupe --plot reports/backtest_tradeoff.png
python scripts/backtest_sweep_summary.py --dedupe --out-md reports/backtest_sweep_summary.md
```

The summary output includes a robust score (r/vol penalized for low trade counts)
and a drawdown-penalized score.
Validation outputs can be archived under `reports/validation_runs/<tag>` using `--bt-run-tag`.

Latest deduped top-10 table: [reports/backtest_top_unique.md](reports/backtest_top_unique.md)

Run unit tests:

```bash
python -m unittest
python -m unittest discover -s tests
```

Or use the Makefile shortcut:

```bash
make test
```

CI note: if you add automation, run `python -m unittest discover -s tests` in your pipeline.

Recommended parameter band (from the latest validation sweeps):

```bash
# Center point
python main.py --backtest-validate \
   --bt-vvix-quantile 0.95 \
   --bt-alpha-grid 5.2 \
   --bt-vvix-grid 178

# Band (flat plateau)
# vvix quantile ~0.95
# alpha spread ~5.2-5.25
# vvix safe threshold ~177.5-180
```

Generate the IRAS monthly report from the ledger:

```bash
python main.py --monthly-report
python main.py --monthly-report --report-month 2026-02
python main.py --monthly-report --report-month last
```

Generate a paper PnL report:

```bash
python main.py --paper-pnl-report
python main.py --paper-pnl-report --paper-pnl-date 2026-02-06
```

Paper PnL reports include both unrealized and realized PnL entries.

Generate a PnL dashboard report (equity curve + drawdowns):

```bash
python main.py --pnl-dashboard
python main.py --pnl-dashboard --pnl-start 2026-01-01 --pnl-end 2026-02-06
```

If Telegram credentials are configured, the PnL dashboard run will send a
portfolio performance summary message. Use `--pnl-dashboard-no-telegram` to
skip the message or `--pnl-dashboard-telegram` to force sending.

Override the downside sample requirement for Sortino:

```bash
python main.py --pnl-dashboard --pnl-downside-min-days 10
```

Note: values below 2 may produce unstable Sortino results.

Export a standalone performance JSON summary:

```bash
python main.py --pnl-dashboard --pnl-performance-json
```

PnL dashboard output files:

- `reports/pnl_dashboard_all.json`
- `reports/pnl_dashboard_all.md`
- `reports/pnl_dashboard_all_performance.csv`
- `reports/pnl_dashboard_all_performance.json` (when `--pnl-performance-json` is used)

Backfill PAPER_PNL entries from a simple strategy (Yahoo Finance marks):

```bash
python scripts/backfill_paper_pnl_from_strategy.py \
   --strategy buy_hold \
   --start 2013-01-01 \
   --end 2026-02-06
```

Strategy choices:

- `buy_hold`
- `monthly_rebalance`
- `volatility_timing`
- `momentum`
- `trend_vix`
- `risk_parity`
- `dual_momentum`
- `combined`

Note: `SPX_PUT` marks are proxied with `^SPX`, and the REIT basket uses `CLR.SI`.

To append directly to the live ledger (use with care):

```bash
python scripts/backfill_paper_pnl_from_strategy.py \
   --strategy monthly_rebalance \
   --start 2013-01-01 \
   --end 2026-02-06 \
   --append
```

Generate a portfolio dashboard report (module equity + correlations):

```bash
python main.py --portfolio-dashboard
python main.py --portfolio-dashboard --portfolio-start 2026-01-01 --portfolio-end 2026-02-06
python main.py --portfolio-dashboard --portfolio-skip-recent
```

Portfolio dashboard output files:

- `reports/portfolio_dashboard_all.json`
- `reports/portfolio_dashboard_all.md`
- `reports/portfolio_dashboard_all.csv`
- `reports/portfolio_dashboard_all_correlations.csv`

Skip recent trend sections in the markdown:

```bash
python main.py --portfolio-dashboard --portfolio-skip-recent
```

Generate a signal health report (signal frequency + stale data counts):

```bash
python main.py --signal-health-report
python main.py --signal-health-report --signal-start 2026-01-01 --signal-end 2026-02-06
```

Generate a slippage report (reference vs fill):

```bash
python main.py --slippage-report
python main.py --slippage-report --slip-start 2026-01-01 --slip-end 2026-02-06
```

Latest slippage report (markdown): [reports/slippage_report_all.md](reports/slippage_report_all.md)

Generate a PnL attribution report (by module):

```bash
python main.py --attribution-report
python main.py --attribution-report --attr-start 2026-01-01 --attr-end 2026-02-06
```

Latest attribution report (markdown): [reports/attribution_report_all.md](reports/attribution_report_all.md)

Generate a monitoring report:

```bash
python main.py --monitoring-report
```

Latest monitoring report (markdown): [reports/monitoring_report_YYYY-MM-DD.md](reports/monitoring_report_YYYY-MM-DD.md)

Generate monitoring report with Telegram alerts:

```bash
python main.py --monitoring-report --monitoring-alerts
```

Log a manual execution entry:

```bash
python main.py --log-execution \
   --exec-category Execution \
   --exec-ticker SPX_PUT \
   --exec-action ROLL \
   --exec-rationale "Rolled hedge at 45 DTE" \
   --exec-tags "yield_enhancement,shield" \
   --exec-details-json '{"dte_remaining":45,"strike":3000}' \
   --exec-correlation-id <from_signal>
```

Simulate a paper execution:

```bash
python main.py --paper-execution \
   --execution-broker paper \
   --paper-symbol SPX_PUT \
   --paper-side BUY \
   --paper-qty 1 \
   --paper-reference-price 1.25 \
   --paper-module shield \
   --paper-slippage-bps 5 \
   --paper-latency-ms 150 \
   --paper-seed 42 \
   --paper-mark-price 1.35
```

Dry-run a paper execution (no slippage):

```bash
python main.py --paper-execution \
   --execution-broker dry-run \
   --paper-symbol SPX_PUT \
   --paper-side BUY \
   --paper-qty 1 \
   --paper-reference-price 1.25
```

List available execution brokers:

```bash
python main.py --list-brokers
```

Note: `external` is a stub adapter that must be implemented before live execution.

Log a manual fill entry:

```bash
python main.py --manual-fill \
   --manual-symbol SPX_PUT \
   --manual-side SELL \
   --manual-qty 1 \
   --manual-fill-price 1.55 \
   --manual-reference-price 1.60 \
   --manual-commission 0.50 \
   --manual-venue "BrokerX" \
   --manual-notes "Manual fill after broker execution" \
   --exec-correlation-id <from_signal>
```

Key flags:
- `--dry-run` prints alerts and skips Telegram + logging.
- `--no-log` disables the JSON ledger writes.

## Outputs

- Ledger file: `fortress_alpha_ledger.json`
- Console output: Alpha/Fortress signals, shield strike estimate, and the CAGR projection

The ledger includes a daily compliance heartbeat plus detailed signal metadata and Shield strike estimates.
It also includes a daily summary entry with key signal flags and risk-free rate metadata.
Entries are tagged as decision or execution to support IRAS reporting.
The daily summary now captures action/category counts, last action timestamps, and core market data.
Market data recency is logged daily to help detect stale Yahoo/VIX/VVIX inputs.
The run will emit a warning and log `DATA_STALE` when recency exceeds `DATA_FRESHNESS_DAYS`.

## Notes

This is a decision-support tool. It does not place trades automatically.

## Governance

See the governance checklist in [docs/governance_checklist.md](docs/governance_checklist.md).

Operational runbook: [docs/runbook.md](docs/runbook.md)
Strategy comparison summary: [docs/strategy_comparison.md](docs/strategy_comparison.md)
