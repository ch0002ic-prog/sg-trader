import argparse
import hashlib
import json
import shutil
from pathlib import Path
from datetime import datetime
import uuid

from sg_trader.config import AppConfig, load_config, validate_config
from sg_trader.logging_utils import (
    backfill_entry_types,
    backfill_execution_modules,
    backfill_schema_version,
    check_ledger_consistency,
    dedup_ledger,
    log_daily_heartbeat,
    log_daily_summary,
    log_transaction,
    migrate_ledger,
    load_ledger,
    send_telegram_alert,
)
from sg_trader.backtest import run_backtest
from sg_trader.execution import (
    DryRunBroker,
    ExecutionRequest,
    ExecutionRouter,
    ExternalBrokerStub,
    ManualBroker,
    PaperBroker,
)
from sg_trader.paper_pnl import apply_fill, compute_unrealized_pnl, read_positions
from sg_trader.pnl_report import write_paper_pnl_report
from sg_trader.pnl_dashboard import (
    build_pnl_dashboard,
    format_pnl_performance_message,
    write_pnl_performance_json,
    write_pnl_dashboard,
)
from sg_trader.portfolio_dashboard import write_portfolio_dashboard
from sg_trader.signal_health import write_signal_health_report
from sg_trader.slippage_report import write_slippage_report
from sg_trader.attribution_report import write_attribution_report
from sg_trader.monitoring import write_monitoring_report
from sg_trader.backtest_validation import (
    PRIMARY_ALPHA_GRID,
    PRIMARY_VVIX_GRID,
    PRIMARY_VVIX_QUANTILE,
    write_validation_report,
)
from sg_trader.projection import project_initial_investment
from sg_trader.rebalancing import AllocationSnapshot, check_allocation_drift
from sg_trader.reporting import generate_monthly_report
from sg_trader.signals import (
    calculate_shield_strike,
    check_fortress_rebalance,
    check_market_data_quality,
    fetch_close_series,
    fetch_mas_tbill_yield,
    get_growth_signal,
    get_market_signals,
)


def _infer_module_tag(symbol: str, config: AppConfig) -> str | None:
    upper = symbol.upper()
    if upper == "SPX_PUT":
        return "shield"
    if upper == "S-REIT_BASKET":
        return "fortress"
    if upper in {"^SPX", "SPX", "SPY"}:
        return "alpha"
    if upper == config.growth_ticker.upper():
        return "growth"
    return None


def build_parser(config: AppConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the sg-trader decision-support checks.",
    )
    parser.add_argument(
        "--alpha-spread-threshold",
        type=float,
        default=config.alpha_spread_threshold,
    )
    parser.add_argument(
        "--vvix-safe-threshold",
        type=float,
        default=config.vvix_safe_threshold,
    )
    parser.add_argument(
        "--reit-spread-threshold",
        type=float,
        default=config.reit_spread_threshold,
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=config.risk_free_rate,
    )
    parser.add_argument(
        "--dte",
        type=int,
        default=180,
    )
    parser.add_argument(
        "--shield-dte-remaining",
        type=int,
        default=None,
        help="Remaining DTE for the current Shield hedge (manual input).",
    )
    parser.add_argument(
        "--shield-profit-multiple",
        type=float,
        default=None,
        help="Current Shield profit multiple (manual input).",
    )
    parser.add_argument(
        "--log-execution",
        action="store_true",
        help="Log a manual execution entry and exit.",
    )
    parser.add_argument(
        "--manual-fill",
        action="store_true",
        help="Log a manual fill entry with price/qty and exit.",
    )
    parser.add_argument(
        "--manual-symbol",
        type=str,
        default="",
        help="Manual fill symbol.",
    )
    parser.add_argument(
        "--manual-side",
        type=str,
        default="BUY",
        help="Manual fill side (BUY or SELL).",
    )
    parser.add_argument(
        "--manual-qty",
        type=float,
        default=0.0,
        help="Manual fill quantity.",
    )
    parser.add_argument(
        "--manual-fill-price",
        type=float,
        default=0.0,
        help="Manual fill price.",
    )
    parser.add_argument(
        "--manual-reference-price",
        type=float,
        default=0.0,
        help="Optional reference price for slippage analysis.",
    )
    parser.add_argument(
        "--manual-commission",
        type=float,
        default=0.0,
        help="Optional commission for the manual fill.",
    )
    parser.add_argument(
        "--manual-venue",
        type=str,
        default="",
        help="Optional execution venue or broker name.",
    )
    parser.add_argument(
        "--manual-notes",
        type=str,
        default="",
        help="Optional notes for the manual fill.",
    )
    parser.add_argument(
        "--exec-category",
        type=str,
        default="Execution",
        help="Execution category label.",
    )
    parser.add_argument(
        "--exec-ticker",
        type=str,
        default="N/A",
        help="Execution ticker symbol.",
    )
    parser.add_argument(
        "--exec-action",
        type=str,
        default="MANUAL",
        help="Execution action label (e.g., BUY, SELL, ROLL).",
    )
    parser.add_argument(
        "--exec-rationale",
        type=str,
        default="Manual execution logged.",
        help="Execution rationale text.",
    )
    parser.add_argument(
        "--exec-tags",
        type=str,
        default="",
        help="Comma-separated tags for the execution entry.",
    )
    parser.add_argument(
        "--exec-details-json",
        type=str,
        default="",
        help="JSON string with extra execution details.",
    )
    parser.add_argument(
        "--exec-correlation-id",
        type=str,
        default="",
        help="Correlation ID to link executions to signals.",
    )
    parser.add_argument(
        "--paper-execution",
        action="store_true",
        help="Simulate an execution with the paper broker.",
    )
    parser.add_argument(
        "--execution-plan",
        action="store_true",
        help="Generate an auditable execution plan using paper inputs.",
    )
    parser.add_argument(
        "--execution-approve",
        type=str,
        default="",
        help="Approve a saved execution plan by ID or path.",
    )
    parser.add_argument(
        "--execution-approve-reason",
        type=str,
        default="",
        help="Optional approval rationale to log.",
    )
    parser.add_argument(
        "--execution-replay",
        type=str,
        default="",
        help="Replay an approved execution plan by ID or path.",
    )
    parser.add_argument(
        "--execution-replay-broker",
        type=str,
        default="",
        help="Optional broker override for execution replay.",
    )
    parser.add_argument(
        "--execution-replay-skip-risk",
        action="store_true",
        help="Skip execution risk checks during replay.",
    )
    parser.add_argument(
        "--execution-replay-allow-random",
        action="store_true",
        help="Allow replay without a deterministic paper seed.",
    )
    parser.add_argument(
        "--execution-plan-id",
        type=str,
        default="",
        help="Execution plan ID required for paper execution approval checks.",
    )
    parser.add_argument(
        "--execution-plan-max-age-hours",
        type=float,
        default=None,
        help="Optional max age (hours) for execution plans before paper execution.",
    )
    parser.add_argument(
        "--execution-approval-max-age-hours",
        type=float,
        default=None,
        help="Optional max age (hours) for execution approvals.",
    )
    parser.add_argument(
        "--execution-plan-slippage-tolerance-bps",
        type=float,
        default=0.0,
        help="Allowed slippage bps mismatch when validating plan inputs.",
    )
    parser.add_argument(
        "--execution-plan-latency-tolerance-ms",
        type=int,
        default=0,
        help="Allowed latency ms mismatch when validating plan inputs.",
    )
    parser.add_argument(
        "--list-brokers",
        action="store_true",
        help="List available execution brokers and exit.",
    )
    parser.add_argument(
        "--execution-broker",
        type=str,
        default="paper",
        help="Execution broker name for simulated runs.",
    )
    parser.add_argument(
        "--paper-symbol",
        type=str,
        default="SPX_PUT",
        help="Paper execution symbol.",
    )
    parser.add_argument(
        "--paper-side",
        type=str,
        default="BUY",
        help="Paper execution side (BUY or SELL).",
    )
    parser.add_argument(
        "--paper-qty",
        type=float,
        default=1.0,
        help="Paper execution quantity.",
    )
    parser.add_argument(
        "--paper-reference-price",
        type=float,
        default=1.0,
        help="Paper execution reference price.",
    )
    parser.add_argument(
        "--paper-slippage-bps",
        type=float,
        default=5.0,
        help="Paper execution slippage in bps.",
    )
    parser.add_argument(
        "--paper-latency-ms",
        type=int,
        default=150,
        help="Paper execution latency in milliseconds.",
    )
    parser.add_argument(
        "--paper-seed",
        type=int,
        default=None,
        help="Optional seed for deterministic paper execution.",
    )
    parser.add_argument(
        "--paper-module",
        type=str,
        default="",
        help="Optional module tag for paper execution (alpha/fortress/shield/growth).",
    )
    parser.add_argument(
        "--paper-mark-price",
        type=float,
        default=None,
        help="Optional mark price to compute paper PnL.",
    )
    parser.add_argument(
        "--paper-pnl-snapshot",
        action="store_true",
        help="Log a paper PnL snapshot for the current position.",
    )
    parser.add_argument(
        "--paper-post-trade-reports",
        action="store_true",
        help="After paper execution, refresh paper PnL + slippage + attribution reports.",
    )
    parser.add_argument(
        "--paper-post-trade-monthly",
        action="store_true",
        help="After paper execution, refresh the monthly report.",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run the VectorBT backtest and print summary stats.",
    )
    parser.add_argument(
        "--backtest-validate",
        action="store_true",
        help="Run backtest stress tests and parameter sensitivity.",
    )
    parser.add_argument(
        "--validation-suite",
        action="store_true",
        help="Run the standard validation suite and emit a summary markdown.",
    )
    parser.add_argument(
        "--refresh-reports",
        action="store_true",
        help="Run validation + slippage + attribution + monthly reports in order.",
    )
    parser.add_argument(
        "--bt-alpha-grid",
        type=str,
        default="3,5,7",
        help="Comma-separated alpha spread thresholds for validation.",
    )
    parser.add_argument(
        "--bt-vvix-grid",
        type=str,
        default="90,110,130",
        help="Comma-separated VVIX thresholds for validation.",
    )
    parser.add_argument(
        "--bt-vix-quantile",
        type=float,
        default=0.75,
        help="Quantile for high VIX regime (0-1).",
    )
    parser.add_argument(
        "--bt-vvix-quantile",
        type=float,
        default=0.75,
        help="Quantile for high VVIX regime (0-1).",
    )
    parser.add_argument(
        "--bt-use-primary",
        action="store_true",
        help="Use the primary validation grid defaults.",
    )
    parser.add_argument(
        "--bt-min-trades",
        type=int,
        default=40,
        help="Minimum trades required for filtered validation rankings.",
    )
    parser.add_argument(
        "--bt-max-drawdown",
        type=float,
        default=0.25,
        help="Optional max drawdown threshold (0-1) for filtered validation rankings.",
    )
    parser.add_argument(
        "--bt-run-tag",
        type=str,
        default="",
        help="Optional tag to archive validation outputs under reports/validation_runs/.",
    )
    parser.add_argument(
        "--suite-tag",
        type=str,
        default="",
        help="Optional tag used to namespace validation suite outputs.",
    )
    parser.add_argument(
        "--monthly-report",
        action="store_true",
        help="Generate the IRAS monthly report from the ledger.",
    )
    parser.add_argument(
        "--refresh-daily-summary",
        action="store_true",
        help="Rewrite today's daily summary entry.",
    )
    parser.add_argument(
        "--dedup-ledger",
        action="store_true",
        help="Remove duplicate ledger entries (one per day).",
    )
    parser.add_argument(
        "--backfill-entry-types",
        action="store_true",
        help="Backfill decision/execution entry types in the ledger.",
    )
    parser.add_argument(
        "--backfill-schema-version",
        action="store_true",
        help="Backfill ledger schema versions.",
    )
    parser.add_argument(
        "--backfill-exec-modules",
        action="store_true",
        help="Backfill module attribution for execution entries.",
    )
    parser.add_argument(
        "--migrate-ledger",
        action="store_true",
        help="Run all ledger migrations (entry type, module, schema).",
    )
    parser.add_argument(
        "--report-month",
        type=str,
        default="",
        help="Report month in YYYY-MM format (defaults to current month).",
    )
    parser.add_argument(
        "--paper-pnl-report",
        action="store_true",
        help="Generate a paper PnL report for today.",
    )
    parser.add_argument(
        "--paper-pnl-date",
        type=str,
        default="",
        help="Paper PnL report date in YYYY-MM-DD (defaults to today).",
    )
    parser.add_argument(
        "--pnl-dashboard",
        action="store_true",
        help="Generate a paper PnL dashboard report.",
    )
    parser.add_argument(
        "--pnl-start",
        type=str,
        default="",
        help="PnL dashboard start date in YYYY-MM-DD (optional).",
    )
    parser.add_argument(
        "--pnl-end",
        type=str,
        default="",
        help="PnL dashboard end date in YYYY-MM-DD (optional).",
    )
    parser.add_argument(
        "--pnl-dashboard-telegram",
        action="store_true",
        help="Send the PnL dashboard performance summary to Telegram.",
    )
    parser.add_argument(
        "--pnl-dashboard-no-telegram",
        action="store_true",
        help="Skip sending the PnL dashboard summary to Telegram.",
    )
    parser.add_argument(
        "--pnl-downside-min-days",
        type=int,
        default=None,
        help="Minimum downside sample days required for Sortino.",
    )
    parser.add_argument(
        "--pnl-performance-json",
        action="store_true",
        help="Export the PnL performance summary as JSON.",
    )
    parser.add_argument(
        "--portfolio-dashboard",
        action="store_true",
        help="Generate a portfolio dashboard report.",
    )
    parser.add_argument(
        "--portfolio-start",
        type=str,
        default="",
        help="Portfolio dashboard start date in YYYY-MM-DD (optional).",
    )
    parser.add_argument(
        "--portfolio-end",
        type=str,
        default="",
        help="Portfolio dashboard end date in YYYY-MM-DD (optional).",
    )
    parser.add_argument(
        "--portfolio-skip-recent",
        action="store_true",
        help="Skip recent trend sections in the portfolio dashboard markdown.",
    )
    parser.add_argument(
        "--signal-health-report",
        action="store_true",
        help="Generate a signal health report.",
    )
    parser.add_argument(
        "--signal-start",
        type=str,
        default="",
        help="Signal report start date in YYYY-MM-DD (optional).",
    )
    parser.add_argument(
        "--signal-end",
        type=str,
        default="",
        help="Signal report end date in YYYY-MM-DD (optional).",
    )
    parser.add_argument(
        "--slippage-report",
        action="store_true",
        help="Generate a paper slippage report.",
    )
    parser.add_argument(
        "--slip-start",
        type=str,
        default="",
        help="Slippage report start date in YYYY-MM-DD (optional).",
    )
    parser.add_argument(
        "--slip-end",
        type=str,
        default="",
        help="Slippage report end date in YYYY-MM-DD (optional).",
    )
    parser.add_argument(
        "--attribution-report",
        action="store_true",
        help="Generate a PnL attribution report by module.",
    )
    parser.add_argument(
        "--attr-start",
        type=str,
        default="",
        help="Attribution report start date in YYYY-MM-DD (optional).",
    )
    parser.add_argument(
        "--attr-end",
        type=str,
        default="",
        help="Attribution report end date in YYYY-MM-DD (optional).",
    )
    parser.add_argument(
        "--monitoring-report",
        action="store_true",
        help="Generate a monitoring report with daily checks.",
    )
    parser.add_argument(
        "--monitoring-alerts",
        action="store_true",
        help="Send monitoring alerts via Telegram when issues are detected.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument(
        "--print-env",
        action="store_true",
        help="Print whether Telegram env vars are loaded (redacts values).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    cache_path = Path(__file__).resolve().parent / "sg_trader" / "__pycache__"
    if cache_path.exists():
        try:
            shutil.rmtree(cache_path)
        except OSError as exc:
            print(f"Failed to clear cache at {cache_path}: {exc}")

    config = load_config()
    parser = build_parser(config)
    args = parser.parse_args(argv)

    config.alpha_spread_threshold = args.alpha_spread_threshold
    config.vvix_safe_threshold = args.vvix_safe_threshold
    config.reit_spread_threshold = args.reit_spread_threshold
    config.risk_free_rate = args.risk_free_rate

    config_errors = validate_config(config)
    if config_errors:
        print("Config validation failed:")
        for error in config_errors:
            print(f"- {error}")
        return 7

    if args.print_env:
        token_status = "set" if config.telegram_token else "missing"
        chat_status = "set" if config.telegram_chat_id else "missing"
        print(f"TELEGRAM_TOKEN: {token_status}")
        print(f"TELEGRAM_CHAT_ID: {chat_status}")

    if args.list_brokers:
        router = ExecutionRouter()
        router.register("paper", PaperBroker())
        router.register_adapter(ManualBroker())
        router.register_adapter(DryRunBroker())
        router.register_adapter(ExternalBrokerStub())
        print("Available brokers:")
        for name in router.list_brokers():
            print(f"- {name}")
        return 0

    def _resolve_validation_args() -> tuple[str, str, float]:
        if args.bt_use_primary:
            return PRIMARY_ALPHA_GRID, PRIMARY_VVIX_GRID, PRIMARY_VVIX_QUANTILE
        return args.bt_alpha_grid, args.bt_vvix_grid, args.bt_vvix_quantile

    def _archive_validation_outputs(run_tag: str) -> None:
        if not run_tag:
            return
        archive_dir = Path("reports") / "validation_runs" / run_tag
        archive_dir.mkdir(parents=True, exist_ok=True)
        for name in [
            "backtest_validation.json",
            "backtest_validation.md",
            "backtest_top_unique.csv",
            "backtest_top_unique.md",
        ]:
            src = Path("reports") / name
            if src.exists():
                shutil.copy2(src, archive_dir / name)

    def _write_suite_summary(
        suite_dir: Path,
        entries: list[dict[str, object]],
    ) -> Path:
        lines = [
            "# Validation Suite Summary",
            "",
            "run | mode | grid | min_trades | max_drawdown | top_row",
            "--- | --- | --- | --- | --- | ---",
        ]
        for entry in entries:
            lines.append(
                f"{entry['run']} | {entry['mode']} | {entry['grid']} | "
                f"{entry['min_trades']} | {entry['max_drawdown']} | {entry['top_row']}"
            )
        path = suite_dir / "validation_suite_summary.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _extract_top_row(path: Path) -> str:
        if not path.exists():
            return "missing"
        lines = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#") and "---" not in line
        ]
        if len(lines) < 2:
            return "missing"
        return lines[1]

    def _run_validation(
        run_label: str,
        alpha_grid: str,
        vvix_grid: str,
        vvix_quantile: float,
        min_trades: int,
        max_drawdown: float | None,
    ) -> Path:
        return write_validation_report(
            config,
            "reports",
            alpha_grid=alpha_grid,
            vvix_grid=vvix_grid,
            vix_quantile=args.bt_vix_quantile,
            vvix_quantile=vvix_quantile,
            min_trades=min_trades,
            max_drawdown=max_drawdown,
        )

    if args.dedup_ledger:
        removed, backup = dedup_ledger(config)
        print(f"Removed {removed} duplicate entries.")
        print(f"Backup: {backup}")
        return 0

    if args.backfill_entry_types:
        updated, backup = backfill_entry_types(config)
        print(f"Backfilled entry_type on {updated} entries.")
        print(f"Backup: {backup}")
        return 0

    if args.backfill_schema_version:
        updated, backup = backfill_schema_version(config)
        print(f"Backfilled schema_version on {updated} entries.")
        print(f"Backup: {backup}")
        return 0

    if args.backfill_exec_modules:
        updated, backup = backfill_execution_modules(config)
        print(f"Backfilled module on {updated} execution entries.")
        print(f"Backup: {backup}")
        return 0

    if args.migrate_ledger:
        updates, backup = migrate_ledger(config)
        print(
            "Ledger migrations applied: "
            f"entry_type={updates['entry_type']}, "
            f"modules={updates['modules']}, "
            f"schema_version={updates['schema_version']}"
        )
        print(f"Backup: {backup}")
        return 0

    if args.paper_pnl_report:
        date_key = args.paper_pnl_date.strip() or None
        path = write_paper_pnl_report(config, "reports", date_key=date_key)
        print(f"Paper PnL report generated: {path}")
        return 0

    if args.pnl_dashboard:
        start_key = args.pnl_start.strip() or None
        end_key = args.pnl_end.strip() or None
        try:
            if args.pnl_downside_min_days is not None:
                if args.pnl_downside_min_days < 2:
                    print(
                        "Warning: --pnl-downside-min-days < 2 may produce "
                        "unstable Sortino results."
                    )
                config.pnl_downside_min_days = args.pnl_downside_min_days
            start_date = (
                datetime.strptime(start_key, "%Y-%m-%d").date()
                if start_key
                else None
            )
            end_date = (
                datetime.strptime(end_key, "%Y-%m-%d").date()
                if end_key
                else None
            )
            dashboard = build_pnl_dashboard(
                config,
                start_date=start_date,
                end_date=end_date,
            )
            path = write_pnl_dashboard(
                config,
                "reports",
                start_date=start_key,
                end_date=end_key,
                dashboard=dashboard,
            )
        except ValueError as exc:
            print(f"Invalid PnL dashboard date: {exc}")
            return 5
        if args.pnl_performance_json:
            suffix = "all"
            if start_key or end_key:
                suffix = f"{start_key or 'start'}_{end_key or 'end'}"
            perf_path = write_pnl_performance_json(
                dashboard.summary.get("performance", {}),
                Path("reports") / f"pnl_dashboard_{suffix}_performance.json",
            )
            print(f"PnL performance JSON generated: {perf_path}")
        send_summary = args.pnl_dashboard_telegram or (
            not args.pnl_dashboard_no_telegram
            and config.telegram_token
            and config.telegram_chat_id
        )
        if send_summary:
            message = format_pnl_performance_message(dashboard.summary)
            send_telegram_alert(message, config)
        print(f"PnL dashboard generated: {path}")
        return 0

    if args.portfolio_dashboard:
        start_key = args.portfolio_start.strip() or None
        end_key = args.portfolio_end.strip() or None
        try:
            path = write_portfolio_dashboard(
                config,
                "reports",
                start_date=start_key,
                end_date=end_key,
                include_recent=not args.portfolio_skip_recent,
            )
        except ValueError as exc:
            print(f"Invalid portfolio dashboard date: {exc}")
            return 5
        print(f"Portfolio dashboard generated: {path}")
        return 0

    if args.signal_health_report:
        start_key = args.signal_start.strip() or None
        end_key = args.signal_end.strip() or None
        try:
            path = write_signal_health_report(
                config,
                "reports",
                start_date=start_key,
                end_date=end_key,
            )
        except ValueError as exc:
            print(f"Invalid signal report date: {exc}")
            return 5
        print(f"Signal health report generated: {path}")
        return 0

    if args.slippage_report:
        start_key = args.slip_start.strip() or None
        end_key = args.slip_end.strip() or None
        try:
            path = write_slippage_report(
                config,
                "reports",
                start_date=start_key,
                end_date=end_key,
            )
        except ValueError as exc:
            print(f"Invalid slippage report date: {exc}")
            return 5
        print(f"Slippage report generated: {path}")
        return 0

    if args.attribution_report:
        start_key = args.attr_start.strip() or None
        end_key = args.attr_end.strip() or None
        try:
            path = write_attribution_report(
                config,
                "reports",
                start_date=start_key,
                end_date=end_key,
            )
        except ValueError as exc:
            print(f"Invalid attribution report date: {exc}")
            return 5
        print(f"Attribution report generated: {path}")
        return 0

    if args.monitoring_report:
        path = write_monitoring_report(config, "reports")
        if args.monitoring_alerts or config.monitoring_alerts_enabled:
            try:
                content = json.loads(Path(path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                content = None
            if content:
                alerts = content.get("alerts", [])
                if alerts:
                    message = "MONITORING ALERTS\n" + "\n".join(alerts)
                    send_telegram_alert(message, config)
        print(f"Monitoring report generated: {path}")
        return 0

    if args.refresh_reports:
        alpha_grid, vvix_grid, vvix_quantile = _resolve_validation_args()
        run_tag = args.bt_run_tag.strip()
        if not run_tag:
            run_tag = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        try:
            path = write_validation_report(
                config,
                "reports",
                alpha_grid=alpha_grid,
                vvix_grid=vvix_grid,
                vix_quantile=args.bt_vix_quantile,
                vvix_quantile=vvix_quantile,
                min_trades=args.bt_min_trades,
                max_drawdown=args.bt_max_drawdown,
            )
        except (ValueError, RuntimeError) as exc:
            print(f"Backtest validation failed: {exc}")
            return 3
        _archive_validation_outputs(run_tag)
        print(f"Backtest validation report generated: {path}")

        start_key = args.slip_start.strip() or None
        end_key = args.slip_end.strip() or None
        try:
            path = write_slippage_report(
                config,
                "reports",
                start_date=start_key,
                end_date=end_key,
            )
        except ValueError as exc:
            print(f"Invalid slippage report date: {exc}")
            return 5
        print(f"Slippage report generated: {path}")

        start_key = args.attr_start.strip() or None
        end_key = args.attr_end.strip() or None
        try:
            path = write_attribution_report(
                config,
                "reports",
                start_date=start_key,
                end_date=end_key,
            )
        except ValueError as exc:
            print(f"Invalid attribution report date: {exc}")
            return 5
        print(f"Attribution report generated: {path}")

        month_key = args.report_month.strip() or None
        md_path, json_path = generate_monthly_report(
            ledger_path=config.log_file,
            output_dir="reports",
            month_key=month_key,
        )
        print("Monthly report generated:")
        print(f"- {md_path}")
        print(f"- {json_path}")
        return 0

    if args.validation_suite:
        suite_tag = args.suite_tag.strip() or datetime.now().strftime("suite_%Y%m%d_%H%M%S")
        suite_dir = Path("reports") / "validation_suites" / suite_tag
        suite_dir.mkdir(parents=True, exist_ok=True)
        summary_rows: list[dict[str, object]] = []

        primary_alpha, primary_vvix, primary_q = (
            PRIMARY_ALPHA_GRID,
            PRIMARY_VVIX_GRID,
            PRIMARY_VVIX_QUANTILE,
        )
        tight_alpha = "5.0,5.05,5.1,5.15,5.2,5.25,5.3,5.35,5.4,5.45,5.5"
        tight_vvix = "170,172.5,175,177.5,180,182.5,185,187.5,190"

        variants = [
            {"label": "primary", "grid": "primary", "alpha": primary_alpha, "vvix": primary_vvix, "q": primary_q, "min_trades": 40},
            {"label": "primary_min50", "grid": "primary", "alpha": primary_alpha, "vvix": primary_vvix, "q": primary_q, "min_trades": 50},
            {"label": "tight", "grid": "tight", "alpha": tight_alpha, "vvix": tight_vvix, "q": 0.95, "min_trades": 40},
        ]

        for variant in variants:
            _run_validation(
                variant["label"],
                variant["alpha"],
                variant["vvix"],
                variant["q"],
                variant["min_trades"],
                args.bt_max_drawdown,
            )
            run_tag = f"{suite_tag}_{variant['label']}"
            _archive_validation_outputs(run_tag)
            top_row = _extract_top_row(Path("reports") / "backtest_top_unique.md")
            summary_rows.append(
                {
                    "run": run_tag,
                    "mode": "suite",
                    "grid": variant["grid"],
                    "min_trades": variant["min_trades"],
                    "max_drawdown": args.bt_max_drawdown,
                    "top_row": top_row,
                }
            )

        summary_path = _write_suite_summary(suite_dir, summary_rows)
        print(f"Validation suite complete. Summary: {summary_path}")
        return 0

    if args.backtest_validate:
        alpha_grid, vvix_grid, vvix_quantile = _resolve_validation_args()
        try:
            path = write_validation_report(
                config,
                "reports",
                alpha_grid=alpha_grid,
                vvix_grid=vvix_grid,
                vix_quantile=args.bt_vix_quantile,
                vvix_quantile=vvix_quantile,
                min_trades=args.bt_min_trades,
                max_drawdown=args.bt_max_drawdown,
            )
        except ValueError as exc:
            print(f"Invalid backtest validation args: {exc}")
            return 5
        except RuntimeError as exc:
            print(f"Backtest validation failed: {exc}")
            return 3
        _archive_validation_outputs(args.bt_run_tag.strip())
        print(f"Backtest validation report generated: {path}")
        return 0

    if args.manual_fill:
        if args.no_log or args.dry_run:
            print("Manual fill logging skipped (dry-run or no-log).")
            return 0
        symbol = args.manual_symbol.strip()
        if not symbol:
            print("Manual fill requires --manual-symbol.")
            return 5
        if args.manual_qty <= 0:
            print("Manual fill requires --manual-qty > 0.")
            return 5
        if args.manual_fill_price <= 0:
            print("Manual fill requires --manual-fill-price > 0.")
            return 5
        side = args.manual_side.strip().upper()
        if side not in {"BUY", "SELL"}:
            print("Manual fill side must be BUY or SELL.")
            return 5
        module_tag = _infer_module_tag(symbol, config)
        tags = ["manual", "execution"]
        if module_tag:
            tags.append(module_tag)
        details: dict[str, float | str | None] = {
            "quantity": args.manual_qty,
            "fill_price": args.manual_fill_price,
            "side": side,
            "reference_price": args.manual_reference_price or None,
            "commission": args.manual_commission or None,
            "venue": args.manual_venue.strip() or None,
            "module": module_tag or None,
            "correlation_id": args.exec_correlation_id or None,
        }
        log_transaction(
            "Execution",
            symbol,
            f"MANUAL_{side}",
            args.manual_notes.strip() or "Manual fill logged.",
            config,
            tags=tags,
            details=details,
        )
        print("Manual fill entry logged.")
        return 0

    if args.log_execution:
        if args.no_log or args.dry_run:
            print("Execution logging skipped (dry-run or no-log).")
            return 0
        tags = [tag.strip() for tag in args.exec_tags.split(",") if tag.strip()]
        details = {}
        if args.exec_details_json:
            try:
                details = json.loads(args.exec_details_json)
            except json.JSONDecodeError as exc:
                print(f"Invalid --exec-details-json: {exc}")
                return 5
        log_transaction(
            args.exec_category,
            args.exec_ticker,
            args.exec_action,
            args.exec_rationale,
            config,
            tags=tags,
            details={
                **details,
                "correlation_id": args.exec_correlation_id or None,
            },
        )
        print("Execution entry logged.")
        return 0

    if args.paper_pnl_snapshot:
        if args.paper_mark_price is None:
            print("Paper PnL snapshot requires --paper-mark-price.")
            return 5
        positions = read_positions(Path(config.cache_dir))
        position = positions.get(args.paper_symbol)
        if position is None:
            print(f"Paper PnL snapshot blocked: no position for {args.paper_symbol}.")
            return 6
        pnl = compute_unrealized_pnl(position, args.paper_mark_price)
        print(
            "Paper PnL snapshot: "
            f"{args.paper_symbol} mark {args.paper_mark_price:.4f} | "
            f"unrealized {pnl:.4f}"
        )
        if not args.no_log and not args.dry_run:
            module_tag = args.paper_module.strip().lower()
            if not module_tag:
                inferred = _infer_module_tag(args.paper_symbol, config)
                if inferred:
                    module_tag = inferred
            module_tags = ["paper", "execution"]
            if module_tag:
                module_tags.append(module_tag)
            log_transaction(
                "Execution",
                args.paper_symbol,
                "PAPER_PNL",
                "Paper PnL snapshot.",
                config,
                tags=module_tags,
                details={
                    "quantity": position.quantity,
                    "avg_price": position.avg_price,
                    "mark_price": args.paper_mark_price,
                    "unrealized_pnl": pnl,
                    "module": module_tag or None,
                    "correlation_id": args.exec_correlation_id or None,
                },
            )
        return 0

    def evaluate_execution_risks(
        symbol: str,
        side: str,
        quantity: float,
        reference_price: float,
    ) -> dict[str, object]:
        reasons: list[str] = []
        vix_value = None
        if config.paper_kill_switch:
            reasons.append("Kill switch enabled.")
        if config.paper_vol_kill_threshold > 0:
            try:
                vix_latest = fetch_close_series(config.vix_ticker)
            except Exception:
                vix_latest = None
            if vix_latest is not None and not vix_latest.empty:
                vix_value = float(vix_latest.iloc[-1])
                if vix_value >= config.paper_vol_kill_threshold:
                    reasons.append(
                        "Volatility kill switch. "
                        f"VIX {vix_value:.2f} >= {config.paper_vol_kill_threshold}"
                    )
        positions = read_positions(Path(config.cache_dir))
        current = positions.get(symbol)
        current_qty = current.quantity if current else 0.0
        signed_qty = quantity if side.upper() == "BUY" else -quantity
        projected_qty = current_qty + signed_qty
        if abs(projected_qty) > config.paper_max_position_qty:
            reasons.append(
                "Position limit exceeded. "
                f"limit {config.paper_max_position_qty}, projected {projected_qty:.4f}"
            )
        notional = abs(quantity * reference_price)
        if notional > config.paper_max_notional:
            reasons.append(
                "Notional limit exceeded. "
                f"notional {notional:,.2f} > limit {config.paper_max_notional:,.2f}"
            )
        daily_loss = None
        if config.paper_max_daily_loss > 0:
            today = datetime.now().strftime("%Y-%m-%d")
            entries = load_ledger(config)
            daily_loss = 0.0
            for entry in entries:
                if entry.get("action") != "PAPER_PNL":
                    continue
                timestamp = str(entry.get("timestamp", ""))
                if not timestamp.startswith(today):
                    continue
                details = entry.get("details", {})
                pnl = details.get("unrealized_pnl")
                if isinstance(pnl, (int, float)):
                    daily_loss += float(pnl)
            if daily_loss <= -config.paper_max_daily_loss:
                reasons.append(
                    "Daily loss limit reached. "
                    f"limit {config.paper_max_daily_loss}, current {daily_loss:.4f}"
                )
        daily_trades = None
        if config.paper_max_daily_trades > 0:
            today = datetime.now().strftime("%Y-%m-%d")
            entries = load_ledger(config)
            daily_trades = 0
            for entry in entries:
                if entry.get("action") not in {"PAPER_BUY", "PAPER_SELL"}:
                    continue
                timestamp = str(entry.get("timestamp", ""))
                if not timestamp.startswith(today):
                    continue
                daily_trades += 1
            if daily_trades >= config.paper_max_daily_trades:
                reasons.append(
                    "Daily trade limit reached. "
                    f"limit {config.paper_max_daily_trades}, current {daily_trades}"
                )
        return {
            "blocked": bool(reasons),
            "reasons": reasons,
            "vix_value": vix_value,
            "projected_qty": projected_qty,
            "notional": notional,
            "daily_loss": daily_loss,
            "daily_trades": daily_trades,
        }

    def _load_execution_plan(plan_ref: str) -> dict[str, object] | None:
        plan_path = Path(plan_ref)
        if not plan_path.suffix:
            plan_path = (
                Path("reports")
                / "execution_plans"
                / f"execution_plan_{plan_ref}.json"
            )
        if not plan_path.exists():
            print(f"Execution plan not found: {plan_path}")
            return None
        try:
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"Invalid execution plan JSON: {exc}")
            return None
        if not isinstance(payload, dict):
            print("Invalid execution plan payload.")
            return None
        payload["_path"] = str(plan_path)
        return payload

    def _verify_execution_plan(plan: dict[str, object]) -> tuple[bool, str]:
        if "plan_hash" not in plan:
            return False, "Missing plan_hash."
        plan_copy = {k: v for k, v in plan.items() if k not in {"plan_hash", "_path"}}
        plan_payload = json.dumps(plan_copy, sort_keys=True)
        computed = hashlib.sha256(plan_payload.encode("utf-8")).hexdigest()
        if computed != plan.get("plan_hash"):
            return False, "Plan hash mismatch."
        return True, ""

    def _plan_age_hours(plan: dict[str, object]) -> float | None:
        timestamp = str(plan.get("timestamp", ""))
        if not timestamp:
            return None
        try:
            plan_dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
        return (datetime.now() - plan_dt).total_seconds() / 3600.0

    def _approval_age_hours(approval: dict[str, object]) -> float | None:
        timestamp = str(approval.get("timestamp", ""))
        if not timestamp:
            return None
        try:
            approval_dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
        return (datetime.now() - approval_dt).total_seconds() / 3600.0

    def _load_execution_approval(plan_id: str) -> dict[str, object] | None:
        approval_path = (
            Path("reports")
            / "execution_plans"
            / f"execution_approval_{plan_id}.json"
        )
        if not approval_path.exists():
            print(f"Execution approval not found: {approval_path}")
            return None
        try:
            payload = json.loads(approval_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"Invalid execution approval JSON: {exc}")
            return None
        if not isinstance(payload, dict):
            print("Invalid execution approval payload.")
            return None
        payload["_path"] = str(approval_path)
        return payload

    def _validate_plan_matches_args(
        plan: dict[str, object],
        slippage_tolerance_bps: float,
        latency_tolerance_ms: int,
    ) -> list[str]:
        issues: list[str] = []
        if str(plan.get("symbol")) != args.paper_symbol:
            issues.append("Plan symbol does not match execution symbol.")
        if str(plan.get("side", "")).upper() != args.paper_side.upper():
            issues.append("Plan side does not match execution side.")
        if float(plan.get("quantity", 0.0)) != float(args.paper_qty):
            issues.append("Plan quantity does not match execution quantity.")
        if float(plan.get("reference_price", 0.0)) != float(args.paper_reference_price):
            issues.append("Plan reference price does not match execution reference price.")
        slippage_delta = abs(
            float(plan.get("slippage_bps", 0.0)) - float(args.paper_slippage_bps)
        )
        if slippage_delta > slippage_tolerance_bps:
            issues.append("Plan slippage does not match execution slippage.")
        latency_delta = abs(int(plan.get("latency_ms", 0)) - int(args.paper_latency_ms))
        if latency_delta > latency_tolerance_ms:
            issues.append("Plan latency does not match execution latency.")
        if str(plan.get("broker", "")) != args.execution_broker:
            issues.append("Plan broker does not match execution broker.")
        return issues

    if args.execution_plan:
        risk = evaluate_execution_risks(
            args.paper_symbol,
            args.paper_side,
            args.paper_qty,
            args.paper_reference_price,
        )
        slip_factor = max(0.0, args.paper_slippage_bps) / 10000.0
        if args.paper_side.upper() == "BUY":
            expected_min = args.paper_reference_price
            expected_max = args.paper_reference_price * (1 + slip_factor)
        else:
            expected_min = args.paper_reference_price * (1 - slip_factor)
            expected_max = args.paper_reference_price
        plan_id = uuid.uuid4().hex
        correlation_id = args.exec_correlation_id or uuid.uuid4().hex
        plan = {
            "plan_id": plan_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "broker": args.execution_broker,
            "symbol": args.paper_symbol,
            "side": args.paper_side.upper(),
            "quantity": args.paper_qty,
            "reference_price": args.paper_reference_price,
            "slippage_bps": args.paper_slippage_bps,
            "latency_ms": args.paper_latency_ms,
            "expected_fill_range": {
                "min": expected_min,
                "max": expected_max,
            },
            "risk": risk,
            "correlation_id": correlation_id,
        }
        plan_payload = json.dumps(plan, sort_keys=True)
        plan["plan_hash"] = hashlib.sha256(plan_payload.encode("utf-8")).hexdigest()
        output_dir = Path("reports") / "execution_plans"
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"execution_plan_{plan_id}.json"
        out_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        if not args.no_log and not args.dry_run:
            log_transaction(
                "Execution",
                args.paper_symbol,
                "EXECUTION_PLAN",
                "Execution plan generated.",
                config,
                tags=["execution", "plan"],
                details={
                    "plan_id": plan_id,
                    "plan_hash": plan["plan_hash"],
                    "broker": args.execution_broker,
                    "side": args.paper_side.upper(),
                    "quantity": args.paper_qty,
                    "reference_price": args.paper_reference_price,
                    "slippage_bps": args.paper_slippage_bps,
                    "latency_ms": args.paper_latency_ms,
                    "risk": risk,
                    "correlation_id": correlation_id,
                },
            )
        print(f"Execution plan saved: {out_path}")
        if risk["blocked"]:
            print("Execution plan blocked:")
            for reason in risk["reasons"]:
                print(f"- {reason}")
        return 0

    if args.execution_approve:
        plan = _load_execution_plan(args.execution_approve.strip())
        if plan is None:
            return 5
        ok, error = _verify_execution_plan(plan)
        if not ok:
            print(f"Execution plan verification failed: {error}")
            return 5
        plan_id = str(plan.get("plan_id", ""))
        approval = {
            "plan_id": plan_id,
            "plan_hash": plan.get("plan_hash"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "reason": args.execution_approve_reason.strip() or None,
            "plan_path": plan.get("_path"),
        }
        output_dir = Path("reports") / "execution_plans"
        output_dir.mkdir(parents=True, exist_ok=True)
        approval_path = output_dir / f"execution_approval_{plan_id}.json"
        approval_path.write_text(json.dumps(approval, indent=2), encoding="utf-8")
        if not args.no_log and not args.dry_run:
            log_transaction(
                "Execution",
                str(plan.get("symbol", "")),
                "EXECUTION_APPROVED",
                args.execution_approve_reason.strip()
                or "Execution plan approved.",
                config,
                tags=["execution", "approval"],
                details=approval,
            )
        print(f"Execution approval saved: {approval_path}")
        return 0

    if args.execution_replay:
        plan = _load_execution_plan(args.execution_replay.strip())
        if plan is None:
            return 6
        ok, error = _verify_execution_plan(plan)
        if not ok:
            print(f"Execution replay blocked: {error}")
            return 6
        plan_id = str(plan.get("plan_id", ""))
        if not plan_id:
            print("Execution replay blocked: missing plan ID.")
            return 6
        max_age_hours = (
            args.execution_plan_max_age_hours
            if args.execution_plan_max_age_hours is not None
            else config.execution_plan_max_age_hours
        )
        plan_age = _plan_age_hours(plan)
        if plan_age is None:
            print("Execution replay blocked: invalid plan timestamp.")
            return 6
        if plan_age > max_age_hours:
            print(
                "Execution replay blocked: plan expired. "
                f"age {plan_age:.2f}h > limit {max_age_hours:.2f}h"
            )
            return 6
        approval = _load_execution_approval(plan_id)
        if approval is None:
            print("Execution replay blocked: approval not found.")
            return 6
        approval_max_age = args.execution_approval_max_age_hours
        if approval_max_age is not None:
            approval_age = _approval_age_hours(approval)
            if approval_age is None:
                print("Execution replay blocked: invalid approval timestamp.")
                return 6
            if approval_age > approval_max_age:
                print(
                    "Execution replay blocked: approval expired. "
                    f"age {approval_age:.2f}h > limit {approval_max_age:.2f}h"
                )
                return 6
        if approval.get("plan_hash") != plan.get("plan_hash"):
            print("Execution replay blocked: approval hash mismatch.")
            return 6
        symbol = str(plan.get("symbol", ""))
        side = str(plan.get("side", "")).upper()
        quantity = float(plan.get("quantity", 0.0))
        reference_price = float(plan.get("reference_price", 0.0))
        slippage_bps = float(plan.get("slippage_bps", 0.0))
        latency_ms = int(plan.get("latency_ms", 0))
        override_broker = args.execution_replay_broker.strip()
        broker = override_broker or str(plan.get("broker", "")) or args.execution_broker
        if override_broker:
            plan_broker = str(plan.get("broker", ""))
            if plan_broker and plan_broker != override_broker:
                print(
                    "Execution replay broker override: "
                    f"{plan_broker} -> {override_broker}"
                )
        if not symbol or side not in {"BUY", "SELL"} or quantity <= 0:
            print("Execution replay blocked: invalid plan inputs.")
            return 6
        if not args.execution_replay_allow_random and args.paper_seed is None:
            print("Execution replay blocked: missing --paper-seed.")
            return 6
        if args.execution_replay_skip_risk:
            risk = {"blocked": False, "reasons": []}
            print("Execution replay risk checks skipped.")
        else:
            risk = evaluate_execution_risks(symbol, side, quantity, reference_price)
            if risk["blocked"]:
                print("Execution replay blocked:")
                for reason in risk["reasons"]:
                    print(f"- {reason}")
                return 8
        router = ExecutionRouter()
        router.register("paper", PaperBroker())
        router.register_adapter(ManualBroker())
        router.register_adapter(DryRunBroker())
        router.register_adapter(ExternalBrokerStub())
        request = ExecutionRequest(
            symbol=symbol,
            side=side,
            quantity=quantity,
            reference_price=reference_price,
            slippage_bps=slippage_bps,
            latency_ms=latency_ms,
            seed=args.paper_seed,
        )
        try:
            result = router.execute(broker, request)
        except (ValueError, RuntimeError) as exc:
            print(str(exc))
            return 6
        position, realized_pnl = apply_fill(
            Path(config.cache_dir),
            result.symbol,
            result.side,
            result.quantity,
            result.fill_price,
        )
        print(
            "Execution replay: "
            f"{result.side} {result.quantity} {result.symbol} "
            f"@ {result.fill_price:.4f} (ref {result.reference_price:.4f})"
        )
        print(
            "Paper position: "
            f"qty {position.quantity:.4f} @ avg {position.avg_price:.4f}"
        )
        if not args.no_log and not args.dry_run:
            module_tag = args.paper_module.strip().lower()
            if not module_tag:
                inferred = _infer_module_tag(result.symbol, config)
                if inferred:
                    module_tag = inferred
            module_tags = ["paper", "execution", "replay"]
            if module_tag:
                module_tags.append(module_tag)
            log_transaction(
                "Execution",
                result.symbol,
                f"PAPER_{result.side.upper()}",
                "Paper execution replayed.",
                config,
                tags=module_tags,
                details={
                    "quantity": result.quantity,
                    "reference_price": result.reference_price,
                    "fill_price": result.fill_price,
                    "slippage_bps": result.slippage_bps,
                    "latency_ms": result.latency_ms,
                    "module": module_tag or None,
                    "plan_id": plan_id,
                    "correlation_id": plan.get("correlation_id")
                    or args.exec_correlation_id
                    or None,
                },
            )
            if realized_pnl != 0.0:
                log_transaction(
                    "Execution",
                    result.symbol,
                    "PAPER_REALIZED",
                    "Paper execution replay realized PnL.",
                    config,
                    tags=module_tags,
                    details={
                        "realized_pnl": realized_pnl,
                        "quantity": result.quantity,
                        "avg_price": position.avg_price,
                        "module": module_tag or None,
                        "plan_id": plan_id,
                        "correlation_id": plan.get("correlation_id")
                        or args.exec_correlation_id
                        or None,
                    },
                )
        if args.paper_mark_price is not None:
            pnl = compute_unrealized_pnl(position, args.paper_mark_price)
            print(
                "Paper PnL: "
                f"mark {args.paper_mark_price:.4f} | unrealized {pnl:.4f}"
            )
            if not args.no_log and not args.dry_run:
                log_transaction(
                    "Execution",
                    result.symbol,
                    "PAPER_PNL",
                    "Paper PnL snapshot.",
                    config,
                    tags=module_tags,
                    details={
                        "quantity": position.quantity,
                        "avg_price": position.avg_price,
                        "mark_price": args.paper_mark_price,
                        "unrealized_pnl": pnl,
                        "module": module_tag or None,
                        "plan_id": plan_id,
                        "correlation_id": plan.get("correlation_id")
                        or args.exec_correlation_id
                        or None,
                    },
                )
        if args.paper_post_trade_reports:
            write_paper_pnl_report(config, "reports")
            write_slippage_report(config, "reports")
            write_attribution_report(config, "reports")
        if args.paper_post_trade_monthly:
            month_key = args.report_month.strip() or None
            md_path, json_path = generate_monthly_report(
                ledger_path=config.log_file,
                output_dir="reports",
                month_key=month_key,
            )
            print("Monthly report generated:")
            print(f"- {md_path}")
            print(f"- {json_path}")
        return 0

    if args.paper_execution:
        plan_id = args.execution_plan_id.strip()
        if not plan_id:
            print("Paper execution blocked: missing --execution-plan-id.")
            return 6
        plan = _load_execution_plan(plan_id)
        if plan is None:
            return 6
        ok, error = _verify_execution_plan(plan)
        if not ok:
            print(f"Paper execution blocked: {error}")
            return 6
        max_age_hours = (
            args.execution_plan_max_age_hours
            if args.execution_plan_max_age_hours is not None
            else config.execution_plan_max_age_hours
        )
        plan_age = _plan_age_hours(plan)
        if plan_age is None:
            print("Paper execution blocked: invalid plan timestamp.")
            return 6
        if plan_age > max_age_hours:
            print(
                "Paper execution blocked: plan expired. "
                f"age {plan_age:.2f}h > limit {max_age_hours:.2f}h"
            )
            return 6
        approval = _load_execution_approval(str(plan.get("plan_id", plan_id)))
        if approval is None:
            print("Paper execution blocked: approval not found.")
            return 6
        approval_max_age = args.execution_approval_max_age_hours
        if approval_max_age is not None:
            approval_age = _approval_age_hours(approval)
            if approval_age is None:
                print("Paper execution blocked: invalid approval timestamp.")
                return 6
            if approval_age > approval_max_age:
                print(
                    "Paper execution blocked: approval expired. "
                    f"age {approval_age:.2f}h > limit {approval_max_age:.2f}h"
                )
                return 6
        if approval.get("plan_hash") != plan.get("plan_hash"):
            print("Paper execution blocked: approval hash mismatch.")
            return 6
        plan_issues = _validate_plan_matches_args(
            plan,
            args.execution_plan_slippage_tolerance_bps,
            args.execution_plan_latency_tolerance_ms,
        )
        if plan_issues:
            print("Paper execution blocked: plan mismatch.")
            for issue in plan_issues:
                print(f"- {issue}")
            return 6
        risk = evaluate_execution_risks(
            args.paper_symbol,
            args.paper_side,
            args.paper_qty,
            args.paper_reference_price,
        )
        if risk["blocked"]:
            print("Paper execution blocked:")
            for reason in risk["reasons"]:
                print(f"- {reason}")
            return 8
        router = ExecutionRouter()
        router.register("paper", PaperBroker())
        router.register_adapter(ManualBroker())
        router.register_adapter(DryRunBroker())
        router.register_adapter(ExternalBrokerStub())
        request = ExecutionRequest(
            symbol=args.paper_symbol,
            side=args.paper_side,
            quantity=args.paper_qty,
            reference_price=args.paper_reference_price,
            slippage_bps=args.paper_slippage_bps,
            latency_ms=args.paper_latency_ms,
            seed=args.paper_seed,
        )
        try:
            result = router.execute(args.execution_broker, request)
        except (ValueError, RuntimeError) as exc:
            print(str(exc))
            return 6
        position, realized_pnl = apply_fill(
            Path(config.cache_dir),
            result.symbol,
            result.side,
            result.quantity,
            result.fill_price,
        )
        print(
            "Paper execution: "
            f"{result.side} {result.quantity} {result.symbol} "
            f"@ {result.fill_price:.4f} (ref {result.reference_price:.4f})"
        )
        print(
            "Paper position: "
            f"qty {position.quantity:.4f} @ avg {position.avg_price:.4f}"
        )
        if not args.no_log and not args.dry_run:
            module_tag = args.paper_module.strip().lower()
            if not module_tag:
                inferred = _infer_module_tag(result.symbol, config)
                if inferred:
                    module_tag = inferred
            module_tags = ["paper", "execution"]
            if module_tag:
                module_tags.append(module_tag)
            log_transaction(
                "Execution",
                result.symbol,
                f"PAPER_{result.side.upper()}",
                "Paper execution simulated.",
                config,
                tags=module_tags,
                details={
                    "quantity": result.quantity,
                    "reference_price": result.reference_price,
                    "fill_price": result.fill_price,
                    "slippage_bps": result.slippage_bps,
                    "latency_ms": result.latency_ms,
                    "module": module_tag or None,
                    "correlation_id": args.exec_correlation_id or None,
                },
            )
            if realized_pnl != 0.0:
                log_transaction(
                    "Execution",
                    result.symbol,
                    "PAPER_REALIZED",
                    "Paper execution realized PnL.",
                    config,
                    tags=module_tags,
                    details={
                        "realized_pnl": realized_pnl,
                        "quantity": result.quantity,
                        "avg_price": position.avg_price,
                        "module": module_tag or None,
                        "correlation_id": args.exec_correlation_id or None,
                    },
                )
        if args.paper_mark_price is not None:
            pnl = compute_unrealized_pnl(position, args.paper_mark_price)
            print(
                "Paper PnL: "
                f"mark {args.paper_mark_price:.4f} | unrealized {pnl:.4f}"
            )
            if not args.no_log and not args.dry_run:
                log_transaction(
                    "Execution",
                    result.symbol,
                    "PAPER_PNL",
                    "Paper PnL snapshot.",
                    config,
                    tags=module_tags,
                    details={
                        "quantity": position.quantity,
                        "avg_price": position.avg_price,
                        "mark_price": args.paper_mark_price,
                        "unrealized_pnl": pnl,
                        "module": module_tag or None,
                        "correlation_id": args.exec_correlation_id or None,
                    },
                )
        if args.paper_post_trade_reports:
            write_paper_pnl_report(config, "reports")
            write_slippage_report(config, "reports")
            write_attribution_report(config, "reports")
        if args.paper_post_trade_monthly:
            month_key = args.report_month.strip() or None
            md_path, json_path = generate_monthly_report(
                ledger_path=config.log_file,
                output_dir="reports",
                month_key=month_key,
            )
            print("Monthly report generated:")
            print(f"- {md_path}")
            print(f"- {json_path}")
        return 0

    def maybe_send(message: str) -> None:
        if args.dry_run:
            if args.dry_run:
                print(message)
            print("Alert suppressed (dry-run).")
            return
        send_telegram_alert(message, config)

    def maybe_log(category: str, ticker: str, action: str, rationale: str) -> None:
        if args.no_log or args.dry_run:
            return
        log_transaction(category, ticker, action, rationale, config)

    mas_rate, mas_status = fetch_mas_tbill_yield(config)
    if mas_rate is None:
        print(f"MAS 6-month T-bill yield unavailable ({mas_status}). Aborting.")
        return 4
    risk_free_rate = mas_rate
    risk_free_source = mas_status
    if mas_status.startswith("cache_ok:"):
        try:
            age_hours = float(mas_status.split(":", 1)[1].replace("h", ""))
        except ValueError:
            age_hours = None
        warn_threshold = config.mas_cache_max_age_hours * 0.8
        if age_hours is None:
            print("MAS yield loaded from cache.")
        else:
            print(f"MAS yield loaded from cache ({age_hours:.2f}h old).")
        if not args.no_log and not args.dry_run:
            log_transaction(
                "Compliance",
                "N/A",
                "DATA_STALENESS",
                "Yield Enhancement: MAS yield loaded from cache.",
                config,
                tags=["yield_enhancement", "data", "cache"],
                details={
                    "cache_age_hours": age_hours,
                    "cache_max_age_hours": config.mas_cache_max_age_hours,
                    "near_expiry": age_hours is not None
                    and age_hours >= warn_threshold,
                },
            )
        if age_hours is not None and age_hours >= warn_threshold:
            print(
                "MAS cache is nearing max age; consider refreshing data source."
            )

    if not args.no_log and not args.dry_run:
        log_daily_heartbeat(
            config,
            {
                "alpha_spread_threshold": config.alpha_spread_threshold,
                "vvix_safe_threshold": config.vvix_safe_threshold,
                "reit_spread_threshold": config.reit_spread_threshold,
                "risk_free_rate": risk_free_rate,
                "risk_free_rate_source": risk_free_source,
                "drift_band": config.drift_band,
            },
        )

    drift_alerted = False
    if (
        config.alloc_fortress is not None
        and config.alloc_alpha is not None
        and config.alloc_shield is not None
    ):
        snapshot = AllocationSnapshot(
            fortress=config.alloc_fortress,
            alpha=config.alloc_alpha,
            shield=config.alloc_shield,
        )
        drift_message, drifts = check_allocation_drift(snapshot, config.drift_band)
        if drift_message:
            maybe_send(drift_message)
            drift_alerted = True
            if not args.no_log and not args.dry_run:
                log_transaction(
                    "Rebalance",
                    "PORTFOLIO",
                    "DRIFT_ALERT",
                    "Yield Enhancement: Allocation drift exceeds band.",
                    config,
                    tags=["yield_enhancement", "rebalance"],
                    details={
                        "allocations": {
                            "fortress": config.alloc_fortress,
                            "alpha": config.alloc_alpha,
                            "shield": config.alloc_shield,
                        },
                        "drifts": drifts,
                        "drift_band": config.drift_band,
                    },
                )
            print("Rebalance drift alert sent.")
        else:
            if not args.no_log and not args.dry_run:
                log_transaction(
                    "Rebalance",
                    "PORTFOLIO",
                    "DRIFT_CHECK",
                    "Yield Enhancement: Allocation drift within band.",
                    config,
                    tags=["yield_enhancement", "rebalance"],
                    details={
                        "allocations": {
                            "fortress": config.alloc_fortress,
                            "alpha": config.alloc_alpha,
                            "shield": config.alloc_shield,
                        },
                        "drifts": drifts,
                        "drift_band": config.drift_band,
                    },
                )
            print("No rebalance drift detected.")
    else:
        print("Rebalance drift check skipped (missing allocation inputs).")

    signals = get_market_signals(config)
    if signals is None:
        print("Market signals unavailable; check data sources.")
        return 2

    safe_regime = signals.vvix < config.vvix_safe_threshold
    alpha_signal_detected = False
    correlation_id = uuid.uuid4().hex
    if signals.spread > config.alpha_spread_threshold and safe_regime:
        message = (
            "ALPHA SIGNAL DETECTED\n"
            f"IV {signals.iv:.2f}% > RV {signals.rv:.2f}% "
            f"| Spread {signals.spread:.2f}%\n"
            f"VVIX {signals.vvix:.2f}\n"
            "Action: Consider selling 30-45 DTE put spread.\n"
            f"Correlation ID: {correlation_id}"
        )
        maybe_send(message)
        alpha_signal_detected = True
        maybe_log(
            "Alpha",
            config.ticker,
            "SIGNAL",
            "Yield Enhancement: IV-RV spread signal; manual execution required.",
        )
        if not args.no_log and not args.dry_run:
            log_transaction(
                "Alpha",
                config.ticker,
                "SIGNAL_DETAILS",
                "Yield Enhancement",
                config,
                tags=["yield_enhancement", "alpha", "signal"],
                details={
                    "iv": signals.iv,
                    "rv": signals.rv,
                    "spread": signals.spread,
                    "vvix": signals.vvix,
                    "vvix_safe_threshold": config.vvix_safe_threshold,
                    "alpha_spread_threshold": config.alpha_spread_threshold,
                    "safe_regime": safe_regime,
                    "correlation_id": correlation_id,
                },
            )
        print("Alpha signal detected.")
    else:
        print(
            f"No Alpha signal. Spread {signals.spread:.2f}% | "
            f"VVIX {signals.vvix:.2f}"
        )

    fortress_message, fortress_opps = check_fortress_rebalance(
        config,
        reit_spread_threshold=config.reit_spread_threshold,
        risk_free_rate=risk_free_rate,
    )
    fortress_signal_detected = False
    if fortress_message:
        fortress_correlation_id = uuid.uuid4().hex
        fortress_message = (
            f"{fortress_message}\nCorrelation ID: {fortress_correlation_id}"
        )
        maybe_send(fortress_message)
        fortress_signal_detected = True
        maybe_log(
            "Fortress",
            "S-REIT_BASKET",
            "SIGNAL",
            "Yield Enhancement: Yield spread above threshold; manual review required.",
        )
        if not args.no_log and not args.dry_run:
            log_transaction(
                "Fortress",
                "S-REIT_BASKET",
                "SIGNAL_DETAILS",
                "Yield Enhancement",
                config,
                tags=["yield_enhancement", "fortress", "signal"],
                details={
                    "reit_spread_threshold": config.reit_spread_threshold,
                    "risk_free_rate": risk_free_rate,
                    "risk_free_rate_source": risk_free_source,
                    "opportunities": fortress_opps.to_dict(orient="records"),
                    "correlation_id": fortress_correlation_id,
                },
            )
        print("Fortress signal detected.")
    else:
        print("No Fortress rebalance signal.")

    shield_strike = calculate_shield_strike(signals.spx, signals.iv, dte=args.dte)
    print(f"Shield 3-sigma strike (DTE {args.dte}): {shield_strike:.2f}")
    shield_roll_dte = 45
    shield_profit_take_min = 5
    shield_profit_take_max = 10
    shield_profit_take_sell_pct = 0.8
    shield_correlation_id = uuid.uuid4().hex
    shield_message = (
        "SHIELD STRIKE ESTIMATE\n"
        f"SPX {signals.spx:.2f} | VIX {signals.iv:.2f}% | DTE {args.dte}\n"
        f"3-sigma strike: {shield_strike:.2f}\n"
        f"Roll rule: {shield_roll_dte} DTE\n"
        f"Profit-take: {shield_profit_take_min}x-{shield_profit_take_max}x, "
        f"sell {int(shield_profit_take_sell_pct * 100)}%\n"
        f"Correlation ID: {shield_correlation_id}"
    )
    maybe_send(shield_message)
    if not args.no_log and not args.dry_run:
        log_transaction(
            "Shield",
            "SPX_PUT",
            "STRIKE_ESTIMATE",
            "Yield Enhancement: Tail-risk hedge strike estimate.",
            config,
            tags=["yield_enhancement", "shield"],
            details={
                "spx": signals.spx,
                "vix": signals.iv,
                "dte": args.dte,
                "strike": shield_strike,
                "roll_dte": shield_roll_dte,
                "profit_take_multiple_min": shield_profit_take_min,
                "profit_take_multiple_max": shield_profit_take_max,
                "profit_take_sell_pct": shield_profit_take_sell_pct,
                "correlation_id": shield_correlation_id,
            },
        )

    growth_signal = get_growth_signal(config)
    growth_signal_active = False
    if growth_signal is None:
        print("Growth signal unavailable; check data source.")
    else:
        if growth_signal.above:
            growth_correlation_id = uuid.uuid4().hex
            growth_message = (
                "GROWTH SIGNAL DETECTED\n"
                f"{growth_signal.ticker} {growth_signal.price:.2f} >= "
                f"{growth_signal.ma_days}D SMA {growth_signal.sma:.2f}\n"
                f"Correlation ID: {growth_correlation_id}"
            )
            maybe_send(growth_message)
            growth_signal_active = True
            maybe_log(
                "Growth",
                growth_signal.ticker,
                "SIGNAL",
                "Yield Enhancement: Growth trend filter above SMA; manual review required.",
            )
            if not args.no_log and not args.dry_run:
                log_transaction(
                    "Growth",
                    growth_signal.ticker,
                    "SIGNAL_DETAILS",
                    "Yield Enhancement",
                    config,
                    tags=["yield_enhancement", "growth", "signal"],
                    details={
                        "price": growth_signal.price,
                        "sma": growth_signal.sma,
                        "ma_days": growth_signal.ma_days,
                        "above_sma": growth_signal.above,
                        "correlation_id": growth_correlation_id,
                    },
                )
            print("Growth signal detected.")
        else:
            print(
                "No Growth signal. "
                f"{growth_signal.ticker} {growth_signal.price:.2f} < "
                f"{growth_signal.ma_days}D SMA {growth_signal.sma:.2f}"
            )

    if args.shield_dte_remaining is not None:
        if args.shield_dte_remaining <= shield_roll_dte:
            roll_message = (
                "SHIELD ROLL ALERT\n"
                f"Remaining DTE: {args.shield_dte_remaining} (roll at {shield_roll_dte})"
            )
            maybe_send(roll_message)
            if not args.no_log and not args.dry_run:
                log_transaction(
                    "Shield",
                    "SPX_PUT",
                    "ROLL_ALERT",
                    "Yield Enhancement: Shield roll condition met.",
                    config,
                    tags=["yield_enhancement", "shield"],
                    details={
                        "dte_remaining": args.shield_dte_remaining,
                        "roll_dte": shield_roll_dte,
                        "correlation_id": shield_correlation_id,
                    },
                )

    if args.shield_profit_multiple is not None:
        if args.shield_profit_multiple >= shield_profit_take_min:
            above_max = args.shield_profit_multiple > shield_profit_take_max
            profit_message = (
                "SHIELD PROFIT-TAKE ALERT\n"
                f"Multiple: {args.shield_profit_multiple:.2f}x "
                f"(target {shield_profit_take_min}x-{shield_profit_take_max}x)"
            )
            if above_max:
                profit_message += "\nAbove target range."
            maybe_send(profit_message)
            if not args.no_log and not args.dry_run:
                log_transaction(
                    "Shield",
                    "SPX_PUT",
                    "PROFIT_TAKE_ALERT",
                    "Yield Enhancement: Shield profit-take condition met.",
                    config,
                    tags=["yield_enhancement", "shield"],
                    details={
                        "profit_multiple": args.shield_profit_multiple,
                        "profit_take_multiple_min": shield_profit_take_min,
                        "profit_take_multiple_max": shield_profit_take_max,
                        "profit_take_sell_pct": shield_profit_take_sell_pct,
                        "above_target_range": above_max,
                        "correlation_id": shield_correlation_id,
                    },
                )

    if not args.no_log and not args.dry_run:
        data_quality = check_market_data_quality(config)
        if not args.no_log:
            log_transaction(
                "Compliance",
                "N/A",
                "DATA_QUALITY",
                "Yield Enhancement: Market data recency check.",
                config,
                tags=["yield_enhancement", "data"],
                details=data_quality,
            )
            stale = {
                key: value
                for key, value in data_quality.items()
                if isinstance(value, (int, float))
                and value > config.data_freshness_days
            }
            if stale:
                log_transaction(
                    "Compliance",
                    "N/A",
                    "DATA_STALE",
                    "Yield Enhancement: Market data is stale.",
                    config,
                    tags=["yield_enhancement", "data"],
                    details={
                        "stale": stale,
                        "threshold_days": config.data_freshness_days,
                    },
                )
        stale_keys = [
            key
            for key, value in data_quality.items()
            if isinstance(value, (int, float))
            and value > config.data_freshness_days
        ]
        if stale_keys:
            print(
                "Data freshness warning: "
                + ", ".join(stale_keys)
                + " exceeds threshold."
            )
        log_daily_summary(
            config,
            {
                "risk_free_rate": risk_free_rate,
                "risk_free_rate_source": risk_free_source,
                "alpha_signal": alpha_signal_detected,
                "fortress_signal": fortress_signal_detected,
                "safe_regime": safe_regime,
                "vvix": signals.vvix,
                "spread": signals.spread,
                "shield_strike": shield_strike,
                "drift_alerted": drift_alerted,
                "market_data": {
                    "spx": signals.spx,
                    "vix": signals.iv,
                    "rv": signals.rv,
                },
                "data_quality": data_quality,
                "data_freshness_days": config.data_freshness_days,
                "growth": {
                    "signal": growth_signal_active,
                    "ticker": growth_signal.ticker if growth_signal else None,
                    "price": growth_signal.price if growth_signal else None,
                    "sma": growth_signal.sma if growth_signal else None,
                    "ma_days": growth_signal.ma_days if growth_signal else None,
                },
            },
            force=args.refresh_daily_summary,
        )

    if not args.dry_run:
        consistency = check_ledger_consistency(config)
        missing = consistency.get("missing_signal_for_details", 0)
        if missing:
            print(f"Ledger consistency warning: {missing} missing SIGNAL entries.")
            if not args.no_log:
                log_transaction(
                    "Compliance",
                    "N/A",
                    "CONSISTENCY_CHECK",
                    "Yield Enhancement: Ledger consistency check.",
                    config,
                    tags=["yield_enhancement", "consistency"],
                    details=consistency,
                )

    initial_investment = 1.0
    projection_years = 40
    cagr_min = 0.12
    cagr_max = 0.14
    projection_low, projection_high = project_initial_investment(
        initial_investment,
        projection_years,
        cagr_min,
        cagr_max,
    )
    projection_message = (
        "Projection from initial investment "
        f"${initial_investment:.2f} over {projection_years} years: "
        f"${projection_low:.2f} to ${projection_high:.2f}"
    )
    print(projection_message)
    maybe_send(f"PROJECTION\n{projection_message}")

    if args.backtest:
        try:
            result = run_backtest(config)
        except RuntimeError as exc:
            print(str(exc))
            return 3
        print("Backtest summary:")
        for key, value in result.stats.items():
            print(f"- {key}: {value:.4f}")
        print(f"- geometric_mean: {result.geometric_mean:.4f}")
        if result.sample_trades:
            print("Backtest sample trades:")
            for trade in result.sample_trades:
                print(
                    f"- entry {trade['entry']} | exit {trade['exit']} | pnl {trade['pnl']}"
                )

    if args.monthly_report:
        raw_month_key = args.report_month.strip()
        if raw_month_key.lower() == "last":
            now = datetime.now()
            year = now.year
            month = now.month - 1
            if month <= 0:
                month = 12
                year -= 1
            month_key = f"{year:04d}-{month:02d}"
        else:
            month_key = raw_month_key or None
        md_path, json_path = generate_monthly_report(
            config.log_file,
            "reports",
            month_key=month_key,
        )
        print("Monthly report generated:")
        print(f"- {md_path}")
        print(f"- {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
