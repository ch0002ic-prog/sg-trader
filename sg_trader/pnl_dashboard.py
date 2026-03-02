from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
import csv
import json
import math
import statistics
from typing import Any, Iterable

from .config import AppConfig
from .logging_utils import load_ledger


@dataclass
class PnlDashboard:
    daily: list[dict[str, Any]]
    weekly: list[dict[str, Any]]
    monthly: list[dict[str, Any]]
    summary: dict[str, Any]


TRADING_DAYS = 252


def _parse_timestamp(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _in_range(day: date, start: date | None, end: date | None) -> bool:
    if start and day < start:
        return False
    if end and day > end:
        return False
    return True


def _compute_drawdown_series(
    items: Iterable[dict[str, Any]],
    label_key: str,
    equity_key: str,
) -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    peak = 0.0
    for entry in items:
        equity = float(entry.get(equity_key, 0.0))
        if equity > peak:
            peak = equity
        drawdown = equity - peak
        drawdown_pct = 0.0 if peak == 0 else drawdown / peak
        series.append(
            {
                label_key: entry.get(label_key),
                "equity": equity,
                "peak": peak,
                "drawdown": drawdown,
                "drawdown_pct": drawdown_pct,
            }
        )
    return series


def _aggregate_period_end(
    daily: list[dict[str, Any]],
    period_key: str,
) -> list[dict[str, Any]]:
    aggregated: list[dict[str, Any]] = []
    last_by_period: dict[str, dict[str, Any]] = {}
    for entry in daily:
        key = entry.get(period_key)
        if key is None:
            continue
        last_by_period[key] = entry
    for key in sorted(last_by_period.keys()):
        entry = last_by_period[key]
        aggregated.append(
            {
                period_key: key,
                "equity": entry.get("equity", 0.0),
                "cumulative_realized": entry.get("cumulative_realized", 0.0),
                "unrealized": entry.get("unrealized", 0.0),
            }
        )
    return aggregated


def _compute_performance_metrics(
    daily: list[dict[str, Any]],
    initial_capital: float,
    risk_free_rate: float,
    downside_min_days: int,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "initial_capital": initial_capital,
        "normalized_start": None,
        "normalized_end": None,
        "cumulative_return": None,
        "cagr": None,
        "annualized_volatility": None,
        "sharpe_ratio": None,
        "sortino_ratio": None,
        "downside_points": 0,
        "downside_deviation": None,
        "downside_sufficient": False,
        "downside_sample_days": 0,
        "downside_min_days": downside_min_days,
        "downside_days_needed": 0,
        "calmar_ratio": None,
        "max_drawdown_pct": None,
        "win_rate": None,
        "rolling_30d_return": None,
        "rolling_90d_return": None,
        "period_days": 0,
        "start_date": None,
        "end_date": None,
        "data_points": len(daily),
    }
    if initial_capital <= 0 or len(daily) < 2:
        return metrics
    normalized: list[float] = []
    dates: list[date] = []
    for entry in daily:
        equity = float(entry.get("equity", 0.0))
        normalized.append(1.0 + equity / initial_capital)
        day_key = entry.get("date")
        if isinstance(day_key, str):
            try:
                dates.append(_parse_date(day_key))
            except ValueError:
                continue
    if len(normalized) < 2 or len(dates) < 2:
        return metrics
    returns = [
        (normalized[idx] / normalized[idx - 1]) - 1.0
        for idx in range(1, len(normalized))
        if normalized[idx - 1] != 0
    ]
    if not returns:
        return metrics
    start_date = dates[0]
    end_date = dates[-1]
    period_days = (end_date - start_date).days
    years = period_days / 365.0 if period_days > 0 else 0.0
    normalized_end = normalized[-1]
    metrics.update(
        {
            "normalized_start": normalized[0],
            "normalized_end": normalized_end,
            "cumulative_return": normalized_end - 1.0,
            "period_days": period_days,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
        }
    )
    if years > 0:
        metrics["cagr"] = normalized_end ** (1.0 / years) - 1.0
    metrics["downside_sample_days"] = len(returns)
    if downside_min_days > len(returns):
        metrics["downside_days_needed"] = downside_min_days - len(returns)
    if len(returns) > 1:
        vol = statistics.stdev(returns) * math.sqrt(TRADING_DAYS)
        metrics["annualized_volatility"] = vol
        rf_annual = float(risk_free_rate) / 100.0
        rf_daily = (1.0 + rf_annual) ** (1.0 / TRADING_DAYS) - 1.0
        excess = [value - rf_daily for value in returns]
        if len(excess) > 1:
            excess_std = statistics.stdev(excess)
            if excess_std > 0:
                metrics["sharpe_ratio"] = (
                    statistics.mean(excess) / excess_std
                ) * math.sqrt(TRADING_DAYS)
            downside = [value for value in excess if value < 0]
            metrics["downside_points"] = len(downside)
            metrics["downside_sufficient"] = (
                len(downside) > 1 and len(returns) >= downside_min_days
            )
            if metrics["downside_sufficient"]:
                downside_std = statistics.stdev(downside)
                metrics["downside_deviation"] = downside_std
                if downside_std > 0:
                    metrics["sortino_ratio"] = (
                        statistics.mean(excess) / downside_std
                    ) * math.sqrt(TRADING_DAYS)
    peak = normalized[0]
    max_dd = 0.0
    for value in normalized:
        if value > peak:
            peak = value
        drawdown = value / peak - 1.0
        if drawdown < max_dd:
            max_dd = drawdown
    metrics["max_drawdown_pct"] = max_dd
    if metrics.get("cagr") is not None and max_dd < 0:
        metrics["calmar_ratio"] = metrics["cagr"] / abs(max_dd)
    metrics["win_rate"] = sum(1 for value in returns if value > 0) / len(returns)
    if len(normalized) > 30:
        metrics["rolling_30d_return"] = (
            normalized[-1] / normalized[-31]
        ) - 1.0
    if len(normalized) > 90:
        metrics["rolling_90d_return"] = (
            normalized[-1] / normalized[-91]
        ) - 1.0
    return metrics


def format_pnl_performance_message(summary: dict[str, Any]) -> str:
    performance = summary.get("performance")
    lines = ["Portfolio Performance (normalized to $1)"]
    if not isinstance(performance, dict) or performance.get("data_points", 0) < 2:
        lines.append("- No PnL data available.")
        return "\n".join(lines)

    def fmt_pct(value: float | None) -> str:
        return "N/A" if value is None else f"{value * 100:.2f}%"

    def fmt_num(value: float | None, digits: int = 2) -> str:
        return "N/A" if value is None else f"{value:.{digits}f}"

    start = performance.get("start_date") or "N/A"
    end = performance.get("end_date") or "N/A"
    days = performance.get("period_days", 0)
    lines.append(f"Period: {start} -> {end} ({days} days)")
    lines.append("Key Metrics")
    lines.append(f"- Final Equity: ${fmt_num(performance.get('normalized_end'), 4)}")
    lines.append(
        f"- Cumulative Return: {fmt_pct(performance.get('cumulative_return'))}"
    )
    lines.append(f"- CAGR: {fmt_pct(performance.get('cagr'))}")
    lines.append(
        "- Annualized Vol: "
        f"{fmt_pct(performance.get('annualized_volatility'))}"
    )
    lines.append(f"- Sharpe: {fmt_num(performance.get('sharpe_ratio'))}")
    sortino = fmt_num(performance.get('sortino_ratio'))
    downside_points = performance.get("downside_points", 0)
    sufficient = performance.get("downside_sufficient", False)
    sample_days = performance.get("downside_sample_days", 0)
    min_days = performance.get("downside_min_days", 0)
    lines.append(
        f"- Sortino: {sortino} (downside n={downside_points}, "
        f"days={sample_days}/{min_days}, ok={sufficient})"
    )
    if not sufficient:
        needed = performance.get("downside_days_needed", 0)
        lines.append(
            "- Warning: downside sample too small for Sortino. "
            f"Need {needed} more day(s)."
        )
    lines.append(f"- Calmar: {fmt_num(performance.get('calmar_ratio'))}")
    lines.append(f"- Max Drawdown: {fmt_pct(performance.get('max_drawdown_pct'))}")
    lines.append(f"- Win Rate (daily): {fmt_pct(performance.get('win_rate'))}")
    lines.append("Rolling Returns")
    lines.append(
        f"- 30D Return: {fmt_pct(performance.get('rolling_30d_return'))}"
    )
    lines.append(
        f"- 90D Return: {fmt_pct(performance.get('rolling_90d_return'))}"
    )
    return "\n".join(lines)


def _write_pnl_markdown(dashboard: PnlDashboard, path: Path) -> None:
    performance = dashboard.summary.get("performance", {})

    def fmt_pct(value: float | None) -> str:
        return "N/A" if value is None else f"{value * 100:.2f}%"

    def fmt_num(value: float | None, digits: int = 2) -> str:
        return "N/A" if value is None else f"{value:.{digits}f}"

    lines = ["# PnL Dashboard", ""]
    lines.append("## Performance (normalized to $1)")
    if not isinstance(performance, dict) or performance.get("data_points", 0) < 2:
        lines.append("- No PnL data available.")
    else:
        start = performance.get("start_date") or "N/A"
        end = performance.get("end_date") or "N/A"
        days = performance.get("period_days", 0)
        lines.append(f"- Period: {start} -> {end} ({days} days)")
        lines.append(
            f"- Final Equity: ${fmt_num(performance.get('normalized_end'), 4)}"
        )
        lines.append(
            f"- Cumulative Return: {fmt_pct(performance.get('cumulative_return'))}"
        )
        lines.append(f"- CAGR: {fmt_pct(performance.get('cagr'))}")
        lines.append(
            f"- Annualized Vol: {fmt_pct(performance.get('annualized_volatility'))}"
        )
        lines.append(f"- Sharpe: {fmt_num(performance.get('sharpe_ratio'))}")
        lines.append(f"- Sortino: {fmt_num(performance.get('sortino_ratio'))}")
        lines.append(
            f"- Downside Points: {performance.get('downside_points', 0)}"
        )
        lines.append(
            f"- Downside Sample Days: {performance.get('downside_sample_days', 0)}"
        )
        lines.append(
            f"- Downside Min Days: {performance.get('downside_min_days', 0)}"
        )
        lines.append(
            "- Downside Days Needed: "
            f"{performance.get('downside_days_needed', 0)}"
        )
        lines.append(
            f"- Downside Deviation: {fmt_num(performance.get('downside_deviation'))}"
        )
        lines.append(
            "- Downside Data Sufficient: "
            f"{performance.get('downside_sufficient', False)}"
        )
        if not performance.get("downside_sufficient", False):
            lines.append(
                "- Warning: downside sample too small for Sortino. "
                f"Need {performance.get('downside_days_needed', 0)} more day(s)."
            )
        lines.append(f"- Calmar: {fmt_num(performance.get('calmar_ratio'))}")
        lines.append(f"- Max Drawdown: {fmt_pct(performance.get('max_drawdown_pct'))}")
        lines.append(f"- Win Rate (daily): {fmt_pct(performance.get('win_rate'))}")
        lines.append(
            f"- 30D Return: {fmt_pct(performance.get('rolling_30d_return'))}"
        )
        lines.append(
            f"- 90D Return: {fmt_pct(performance.get('rolling_90d_return'))}"
        )

    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Total Realized: {fmt_num(dashboard.summary.get('total_realized'))}")
    lines.append(
        f"- Latest Unrealized: {fmt_num(dashboard.summary.get('latest_unrealized'))}"
    )
    lines.append(f"- Latest Equity: {fmt_num(dashboard.summary.get('latest_equity'))}")
    lines.append(
        f"- Max Drawdown (Daily): {fmt_num(dashboard.summary.get('max_drawdown_daily'))}"
    )
    lines.append(
        f"- Max Drawdown (Weekly): {fmt_num(dashboard.summary.get('max_drawdown_weekly'))}"
    )
    lines.append(
        f"- Max Drawdown (Monthly): {fmt_num(dashboard.summary.get('max_drawdown_monthly'))}"
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_pnl_performance_csv(performance: dict[str, Any], path: Path) -> None:
    rows = [
        ("metric", "value"),
        ("start_date", performance.get("start_date")),
        ("end_date", performance.get("end_date")),
        ("period_days", performance.get("period_days")),
        ("normalized_start", performance.get("normalized_start")),
        ("normalized_end", performance.get("normalized_end")),
        ("cumulative_return", performance.get("cumulative_return")),
        ("cagr", performance.get("cagr")),
        ("annualized_volatility", performance.get("annualized_volatility")),
        ("sharpe_ratio", performance.get("sharpe_ratio")),
        ("sortino_ratio", performance.get("sortino_ratio")),
        ("downside_points", performance.get("downside_points")),
        ("downside_sample_days", performance.get("downside_sample_days")),
        ("downside_min_days", performance.get("downside_min_days")),
        ("downside_days_needed", performance.get("downside_days_needed")),
        ("downside_deviation", performance.get("downside_deviation")),
        ("downside_sufficient", performance.get("downside_sufficient")),
        ("calmar_ratio", performance.get("calmar_ratio")),
        ("max_drawdown_pct", performance.get("max_drawdown_pct")),
        ("win_rate", performance.get("win_rate")),
        ("rolling_30d_return", performance.get("rolling_30d_return")),
        ("rolling_90d_return", performance.get("rolling_90d_return")),
        ("data_points", performance.get("data_points")),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def write_pnl_performance_json(performance: dict[str, Any], path: Path) -> Path:
    payload = {
        "performance": performance,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def build_pnl_dashboard(
    config: AppConfig,
    start_date: date | None = None,
    end_date: date | None = None,
) -> PnlDashboard:
    entries = load_ledger(config)
    daily_realized: dict[str, float] = {}
    daily_unrealized: dict[str, dict[str, float]] = {}
    daily_last_pnl_ts: dict[str, dict[str, datetime]] = {}

    for entry in entries:
        action = entry.get("action")
        if action not in {"PAPER_PNL", "PAPER_REALIZED"}:
            continue
        timestamp = entry.get("timestamp", "")
        parsed = _parse_timestamp(timestamp)
        if not parsed:
            continue
        day = parsed.date()
        if not _in_range(day, start_date, end_date):
            continue
        day_key = day.strftime("%Y-%m-%d")
        details = entry.get("details", {})
        symbol = entry.get("ticker", "Unknown")
        if action == "PAPER_REALIZED":
            pnl = details.get("realized_pnl")
            if isinstance(pnl, (int, float)):
                daily_realized[day_key] = daily_realized.get(day_key, 0.0) + float(pnl)
        else:
            pnl = details.get("unrealized_pnl")
            if not isinstance(pnl, (int, float)):
                continue
            daily_unrealized.setdefault(day_key, {})
            daily_last_pnl_ts.setdefault(day_key, {})
            prior_ts = daily_last_pnl_ts[day_key].get(symbol)
            if prior_ts is None or parsed >= prior_ts:
                daily_unrealized[day_key][symbol] = float(pnl)
                daily_last_pnl_ts[day_key][symbol] = parsed

    all_days = sorted(set(daily_realized.keys()) | set(daily_unrealized.keys()))
    daily: list[dict[str, Any]] = []
    cumulative_realized = 0.0
    for day_key in all_days:
        realized = daily_realized.get(day_key, 0.0)
        unrealized = sum(daily_unrealized.get(day_key, {}).values())
        cumulative_realized += realized
        day_dt = _parse_date(day_key)
        week_key = f"{day_dt.isocalendar().year}-W{day_dt.isocalendar().week:02d}"
        month_key = day_dt.strftime("%Y-%m")
        equity = cumulative_realized + unrealized
        daily.append(
            {
                "date": day_key,
                "week": week_key,
                "month": month_key,
                "realized": realized,
                "unrealized": unrealized,
                "cumulative_realized": cumulative_realized,
                "equity": equity,
            }
        )

    daily_drawdown = _compute_drawdown_series(daily, "date", "equity")

    weekly = _aggregate_period_end(daily, "week")
    weekly_drawdown = _compute_drawdown_series(weekly, "week", "equity")

    monthly = _aggregate_period_end(daily, "month")
    monthly_drawdown = _compute_drawdown_series(monthly, "month", "equity")

    summary = {
        "total_realized": cumulative_realized,
        "latest_unrealized": daily[-1]["unrealized"] if daily else 0.0,
        "latest_equity": daily[-1]["equity"] if daily else 0.0,
        "max_drawdown_daily": min((d["drawdown"] for d in daily_drawdown), default=0.0),
        "max_drawdown_weekly": min((d["drawdown"] for d in weekly_drawdown), default=0.0),
        "max_drawdown_monthly": min((d["drawdown"] for d in monthly_drawdown), default=0.0),
    }
    summary["performance"] = _compute_performance_metrics(
        daily,
        config.paper_initial_capital,
        config.risk_free_rate,
        config.pnl_downside_min_days,
    )

    return PnlDashboard(
        daily=daily_drawdown,
        weekly=weekly_drawdown,
        monthly=monthly_drawdown,
        summary=summary,
    )


def write_pnl_dashboard(
    config: AppConfig,
    output_dir: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
    dashboard: PnlDashboard | None = None,
) -> Path:
    start = _parse_date(start_date) if start_date else None
    end = _parse_date(end_date) if end_date else None
    if dashboard is None:
        dashboard = build_pnl_dashboard(config, start_date=start, end_date=end)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "all"
    if start_date or end_date:
        suffix = f"{start_date or 'start'}_{end_date or 'end'}"
    path = output_dir / f"pnl_dashboard_{suffix}.json"
    payload = {
        "summary": dashboard.summary,
        "daily": dashboard.daily,
        "weekly": dashboard.weekly,
        "monthly": dashboard.monthly,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_pnl_markdown(dashboard, output_dir / f"pnl_dashboard_{suffix}.md")
    performance = dashboard.summary.get("performance")
    if isinstance(performance, dict):
        _write_pnl_performance_csv(
            performance,
            output_dir / f"pnl_dashboard_{suffix}_performance.csv",
        )
    return path
