from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
import json
from typing import Any

from .config import AppConfig
from .logging_utils import load_ledger


MODULE_TAGS = ("alpha", "fortress", "shield", "growth")


def _infer_module_from_ticker(ticker: str, config: AppConfig) -> str | None:
    upper = ticker.upper()
    if upper == "SPX_PUT":
        return "shield"
    if upper == "S-REIT_BASKET":
        return "fortress"
    if upper in {"^SPX", "SPX", "SPY"}:
        return "alpha"
    if upper == config.growth_ticker.upper():
        return "growth"
    return None


@dataclass
class AttributionReport:
    summary: dict[str, Any]
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


def _module_from_entry(
    entry: dict[str, Any],
    config: AppConfig,
    correlation_map: dict[str, str],
) -> str:
    details = entry.get("details", {})
    module = details.get("module")
    if isinstance(module, str) and module.strip():
        return module.strip().lower()
    correlation_id = details.get("correlation_id")
    if isinstance(correlation_id, str):
        mapped = correlation_map.get(correlation_id)
        if mapped:
            return mapped
    tags = entry.get("tags", []) or []
    tag_set = {str(tag).strip().lower() for tag in tags}
    for tag in MODULE_TAGS:
        if tag in tag_set:
            return tag
    ticker = str(entry.get("ticker", ""))
    inferred = _infer_module_from_ticker(ticker, config)
    if inferred:
        return inferred
    return "unattributed"


def _build_correlation_map(entries: list[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for entry in entries:
        if entry.get("action") != "SIGNAL_DETAILS":
            continue
        category = str(entry.get("category", "")).lower()
        if category not in MODULE_TAGS:
            continue
        details = entry.get("details", {})
        correlation_id = details.get("correlation_id")
        if isinstance(correlation_id, str) and correlation_id:
            mapping[correlation_id] = category
    return mapping


def build_attribution_report(
    config: AppConfig,
    start_date: date | None = None,
    end_date: date | None = None,
) -> AttributionReport:
    entries = load_ledger(config)
    correlation_map = _build_correlation_map(entries)
    daily: dict[str, dict[str, float]] = {}
    daily_realized: dict[str, dict[str, float]] = {}
    manual_fills: dict[str, dict[str, int]] = {}

    for entry in entries:
        action = entry.get("action")
        if action in {"MANUAL_BUY", "MANUAL_SELL"}:
            timestamp = entry.get("timestamp", "")
            parsed = _parse_timestamp(timestamp)
            if not parsed:
                continue
            day = parsed.date()
            if not _in_range(day, start_date, end_date):
                continue
            day_key = day.strftime("%Y-%m-%d")
            module = _module_from_entry(entry, config, correlation_map)
            manual_fills.setdefault(day_key, {})
            manual_fills[day_key][module] = manual_fills[day_key].get(module, 0) + 1
            continue
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
        module = _module_from_entry(entry, config, correlation_map)
        details = entry.get("details", {})
        if entry.get("action") == "PAPER_PNL":
            pnl = details.get("unrealized_pnl")
            if isinstance(pnl, (int, float)):
                daily.setdefault(day_key, {})
                daily[day_key][module] = daily[day_key].get(module, 0.0) + float(pnl)
        else:
            pnl = details.get("realized_pnl")
            if isinstance(pnl, (int, float)):
                daily_realized.setdefault(day_key, {})
                daily_realized[day_key][module] = (
                    daily_realized[day_key].get(module, 0.0) + float(pnl)
                )

    daily_rows: list[dict[str, Any]] = []
    summary_unrealized: dict[str, float] = {}
    summary_realized: dict[str, float] = {}

    for day_key in sorted(set(daily.keys()) | set(daily_realized.keys())):
        row = {"date": day_key, "unrealized": {}, "realized": {}}
        for module, value in daily.get(day_key, {}).items():
            row["unrealized"][module] = value
            summary_unrealized[module] = summary_unrealized.get(module, 0.0) + value
        for module, value in daily_realized.get(day_key, {}).items():
            row["realized"][module] = value
            summary_realized[module] = summary_realized.get(module, 0.0) + value
        daily_rows.append(row)

    manual_summary: dict[str, int] = {}
    for day in manual_fills.values():
        for module, count in day.items():
            manual_summary[module] = manual_summary.get(module, 0) + count

    summary = {
        "total_unrealized": sum(summary_unrealized.values()),
        "total_realized": sum(summary_realized.values()),
        "unrealized_by_module": summary_unrealized,
        "realized_by_module": summary_realized,
        "manual_fills_by_module": manual_summary,
        "manual_fills_total": sum(manual_summary.values()),
    }

    return AttributionReport(summary=summary, daily=daily_rows)


def write_attribution_report(
    config: AppConfig,
    output_dir: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Path:
    start = _parse_date(start_date) if start_date else None
    end = _parse_date(end_date) if end_date else None
    report = build_attribution_report(config, start_date=start, end_date=end)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "all"
    if start_date or end_date:
        suffix = f"{start_date or 'start'}_{end_date or 'end'}"
    path = output_dir / f"attribution_report_{suffix}.json"
    md_path = output_dir / f"attribution_report_{suffix}.md"
    payload = {
        "summary": report.summary,
        "daily": report.daily,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_lines = [
        "# Attribution Report",
        "",
        f"Total unrealized: {report.summary['total_unrealized']:.4f}",
        f"Total realized: {report.summary['total_realized']:.4f}",
        "",
        "## Manual Fills",
    ]
    manual = report.summary.get("manual_fills_by_module", {})
    md_lines.append(f"- Total: {report.summary.get('manual_fills_total', 0)}")
    if manual:
        for module, count in manual.items():
            md_lines.append(f"- {module}: {count}")
    else:
        md_lines.append("- None")

    md_lines.append("")
    md_lines.append("## Daily")
    for row in report.daily:
        md_lines.append(f"- {row['date']}: {row}")

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return path
