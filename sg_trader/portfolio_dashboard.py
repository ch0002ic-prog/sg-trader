from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
import json
import csv
from typing import Any

import numpy as np

from .config import AppConfig
from .logging_utils import load_ledger


@dataclass
class PortfolioDashboard:
    summary: dict[str, Any]
    modules: dict[str, dict[str, Any]]
    correlations: dict[str, dict[str, float]]
    daily: list[dict[str, Any]]


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


def _extract_module(entry: dict[str, Any]) -> str:
    details = entry.get("details") or {}
    module = details.get("module")
    if module:
        return str(module).lower()
    tags = entry.get("tags") or []
    for tag in tags:
        if isinstance(tag, str) and tag.lower() in {"fortress", "alpha", "shield", "growth"}:
            return tag.lower()
    return "unattributed"


def _correlation_matrix(series_by_module: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    modules = [m for m in sorted(series_by_module.keys()) if len(series_by_module[m]) >= 2]
    if len(modules) < 2:
        return {}
    values = np.array([series_by_module[m] for m in modules], dtype=float)
    diffs = np.diff(values, axis=1)
    if diffs.shape[1] < 2:
        return {}
    corr = np.corrcoef(diffs)
    result: dict[str, dict[str, float]] = {}
    for i, m1 in enumerate(modules):
        row: dict[str, float] = {}
        for j, m2 in enumerate(modules):
            row[m2] = float(corr[i, j])
        result[m1] = row
    return result


def build_portfolio_dashboard(
    config: AppConfig,
    start_date: date | None = None,
    end_date: date | None = None,
) -> PortfolioDashboard:
    entries = load_ledger(config)
    daily_realized: dict[str, dict[str, float]] = {}
    daily_unrealized: dict[str, dict[str, dict[str, float]]] = {}
    daily_last_pnl_ts: dict[str, dict[str, dict[str, datetime]]] = {}

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
        module = _extract_module(entry)
        details = entry.get("details", {})
        symbol = entry.get("ticker", "Unknown")

        if action == "PAPER_REALIZED":
            pnl = details.get("realized_pnl")
            if isinstance(pnl, (int, float)):
                daily_realized.setdefault(day_key, {})
                daily_realized[day_key][module] = (
                    daily_realized[day_key].get(module, 0.0) + float(pnl)
                )
        else:
            pnl = details.get("unrealized_pnl")
            if not isinstance(pnl, (int, float)):
                continue
            daily_unrealized.setdefault(day_key, {}).setdefault(module, {})
            daily_last_pnl_ts.setdefault(day_key, {}).setdefault(module, {})
            prior_ts = daily_last_pnl_ts[day_key][module].get(symbol)
            if prior_ts is None or parsed >= prior_ts:
                daily_unrealized[day_key][module][symbol] = float(pnl)
                daily_last_pnl_ts[day_key][module][symbol] = parsed

    all_days = sorted(set(daily_realized.keys()) | set(daily_unrealized.keys()))
    modules = sorted(
        {
            module
            for day_values in daily_realized.values()
            for module in day_values.keys()
        }
        | {
            module
            for day_values in daily_unrealized.values()
            for module in day_values.keys()
        }
    )

    module_cumulative: dict[str, float] = {module: 0.0 for module in modules}
    daily_rows: list[dict[str, Any]] = []
    equity_by_module: dict[str, list[float]] = {module: [] for module in modules}

    for day_key in all_days:
        row: dict[str, Any] = {"date": day_key}
        realized_by_module = daily_realized.get(day_key, {})
        unrealized_by_module = daily_unrealized.get(day_key, {})
        for module in modules:
            realized = realized_by_module.get(module, 0.0)
            module_cumulative[module] += realized
            unrealized = sum(unrealized_by_module.get(module, {}).values())
            equity = module_cumulative[module] + unrealized
            row[f"{module}_realized"] = realized
            row[f"{module}_unrealized"] = unrealized
            row[f"{module}_equity"] = equity
            equity_by_module[module].append(equity)
        daily_rows.append(row)

    module_totals: dict[str, dict[str, float]] = {}
    for module in modules:
        realized_total = module_cumulative[module]
        unrealized_latest = (
            sum(daily_unrealized.get(all_days[-1], {}).get(module, {}).values())
            if all_days
            else 0.0
        )
        module_totals[module] = {
            "realized_total": realized_total,
            "unrealized_latest": unrealized_latest,
            "equity_latest": realized_total + unrealized_latest,
        }

    correlations = _correlation_matrix(equity_by_module)
    total_realized = sum(
        float(totals.get("realized_total", 0.0)) for totals in module_totals.values()
    )
    total_unrealized = sum(
        float(totals.get("unrealized_latest", 0.0)) for totals in module_totals.values()
    )
    summary = {
        "date": all_days[-1] if all_days else None,
        "module_count": len(modules),
        "modules": modules,
        "total_realized": total_realized,
        "total_unrealized": total_unrealized,
        "total_equity": total_realized + total_unrealized,
    }

    return PortfolioDashboard(
        summary=summary,
        modules=module_totals,
        correlations=correlations,
        daily=daily_rows,
    )


def write_portfolio_dashboard(
    config: AppConfig,
    output_dir: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
    include_recent: bool = True,
) -> Path:
    def _sparkline(values: list[float]) -> str:
        if not values:
            return ""
        levels = " .:-=+*#%@"
        low = min(values)
        high = max(values)
        if high == low:
            idx = len(levels) // 2
            return levels[idx] * len(values)
        scale = (len(levels) - 1) / (high - low)
        return "".join(levels[int((value - low) * scale)] for value in values)

    start = _parse_date(start_date) if start_date else None
    end = _parse_date(end_date) if end_date else None
    dashboard = build_portfolio_dashboard(config, start_date=start, end_date=end)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "all"
    if start_date or end_date:
        suffix = f"{start_date or 'start'}_{end_date or 'end'}"
    path = output_dir / f"portfolio_dashboard_{suffix}.json"
    payload = {
        "summary": dashboard.summary,
        "modules": dashboard.modules,
        "correlations": dashboard.correlations,
        "daily": dashboard.daily,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path = output_dir / f"portfolio_dashboard_{suffix}.md"
    csv_path = output_dir / f"portfolio_dashboard_{suffix}.csv"
    corr_path = output_dir / f"portfolio_dashboard_{suffix}_correlations.csv"

    md_lines = [
        "# Portfolio Dashboard",
        "",
        f"Date: {dashboard.summary.get('date')}",
        f"Modules: {', '.join(dashboard.summary.get('modules', []))}",
        "",
        "## Module Totals",
        "module | realized_total | unrealized_latest | equity_latest",
        "--- | --- | --- | ---",
    ]
    total_realized = float(dashboard.summary.get("total_realized", 0.0))
    total_unrealized = float(dashboard.summary.get("total_unrealized", 0.0))
    total_equity = float(dashboard.summary.get("total_equity", 0.0))
    md_lines.extend(
        [
            "",
            "## Summary",
            f"Total equity (latest): {total_equity:.4f}",
            f"Total realized (latest): {total_realized:.4f}",
            f"Total unrealized (latest): {total_unrealized:.4f}",
        ]
    )

    for module, totals in dashboard.modules.items():
        md_lines.append(
            f"{module} | {totals.get('realized_total', 0.0):.4f} | "
            f"{totals.get('unrealized_latest', 0.0):.4f} | "
            f"{totals.get('equity_latest', 0.0):.4f}"
        )
    if include_recent and dashboard.daily and dashboard.summary.get("modules"):
        md_lines.extend([
            "",
            "## Last 7 Days",
            "module | start_equity | end_equity | change | change_pct | avg_daily | trend",
            "--- | --- | --- | --- | --- | --- | ---",
        ])
        recent_rows = dashboard.daily[-7:]
        mover_rows: list[tuple[str, float, float]] = []
        total_series: list[float] = []
        for row in recent_rows:
            total_series.append(
                sum(
                    float(row.get(f"{module}_equity", 0.0))
                    for module in dashboard.summary.get("modules", [])
                )
            )
        for module in dashboard.summary.get("modules", []):
            series = [row.get(f"{module}_equity", 0.0) for row in recent_rows]
            if not series:
                continue
            start_equity = float(series[0])
            end_equity = float(series[-1])
            change = end_equity - start_equity
            avg_daily = change / max(1, len(series) - 1)
            if start_equity == 0:
                change_pct = "-"
            else:
                change_pct = f"{(change / start_equity) * 100:.2f}%"
            md_lines.append(
                f"{module} | {start_equity:.4f} | {end_equity:.4f} | "
                f"{change:.4f} | {change_pct} | {avg_daily:.4f} | "
                f"{_sparkline(series)}"
            )
            mover_rows.append((module, change, avg_daily))
        if total_series:
            total_change = total_series[-1] - total_series[0]
            total_avg = total_change / max(1, len(total_series) - 1)
            md_lines.extend(
                [
                    "",
                    "Total 7-day change: "
                    f"{total_change:.4f} | avg_daily {total_avg:.4f}",
                ]
            )
        if mover_rows:
            mover_rows.sort(key=lambda item: abs(item[1]), reverse=True)
            md_lines.extend([
                "",
                "## Top Movers (7-Day Change)",
                "module | change | avg_daily",
                "--- | --- | ---",
            ])
            for module, change, avg_daily in mover_rows[:3]:
                md_lines.append(
                    f"{module} | {change:.4f} | {avg_daily:.4f}"
                )
    if dashboard.correlations:
        md_lines.extend(["", "## Module Correlations (Equity Changes)"])
        modules = sorted(dashboard.correlations.keys())
        header = "module | " + " | ".join(modules)
        separator = "--- | " + " | ".join(["---"] * len(modules))
        md_lines.extend([header, separator])
        for module in modules:
            row = [module]
            for other in modules:
                value = dashboard.correlations.get(module, {}).get(other)
                row.append("" if value is None else f"{value:.3f}")
            md_lines.append(" | ".join(row))

    md_lines.extend(["", "## Daily", "(See CSV for full daily breakdown.)"])
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    if dashboard.daily:
        fieldnames = list(dashboard.daily[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(dashboard.daily)
    if dashboard.correlations:
        modules = sorted(dashboard.correlations.keys())
        with corr_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["module"] + modules)
            for module in modules:
                row = [module]
                for other in modules:
                    value = dashboard.correlations.get(module, {}).get(other)
                    row.append("") if value is None else row.append(f"{value:.3f}")
                writer.writerow(row)
    return path
