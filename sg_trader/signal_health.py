from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
import json
from typing import Any, Iterable

from .config import AppConfig
from .logging_utils import load_ledger


@dataclass
class SignalHealthReport:
    daily: list[dict[str, Any]]
    summary: dict[str, Any]


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


def _sum_counts(rows: Iterable[dict[str, Any]], key: str) -> int:
    total = 0
    for row in rows:
        value = row.get(key)
        if isinstance(value, int):
            total += value
    return total


def build_signal_health_report(
    config: AppConfig,
    start_date: date | None = None,
    end_date: date | None = None,
) -> SignalHealthReport:
    entries = load_ledger(config)
    signals_by_day: dict[str, set[tuple[str, str]]] = {}
    details_rows: list[dict[str, Any]] = []
    daily_counts: dict[str, dict[str, int]] = {}

    for entry in entries:
        timestamp = entry.get("timestamp", "")
        parsed = _parse_timestamp(timestamp)
        if not parsed:
            continue
        day = parsed.date()
        if not _in_range(day, start_date, end_date):
            continue
        day_key = day.strftime("%Y-%m-%d")
        action = entry.get("action", "")
        category = entry.get("category", "")
        ticker = entry.get("ticker", "")

        daily_counts.setdefault(day_key, {})
        daily_counts[day_key]["ledger_entries"] = daily_counts[day_key].get(
            "ledger_entries", 0
        ) + 1

        if action == "SIGNAL":
            signals_by_day.setdefault(day_key, set()).add((category, ticker))
            if category == "Alpha":
                daily_counts[day_key]["alpha_signals"] = (
                    daily_counts[day_key].get("alpha_signals", 0) + 1
                )
            elif category == "Fortress":
                daily_counts[day_key]["fortress_signals"] = (
                    daily_counts[day_key].get("fortress_signals", 0) + 1
                )

        if action == "SIGNAL_DETAILS":
            details_rows.append(
                {"day": day_key, "category": category, "ticker": ticker}
            )

        if action == "STRIKE_ESTIMATE":
            daily_counts[day_key]["shield_strike_estimates"] = (
                daily_counts[day_key].get("shield_strike_estimates", 0) + 1
            )
        if action == "ROLL_ALERT":
            daily_counts[day_key]["shield_roll_alerts"] = (
                daily_counts[day_key].get("shield_roll_alerts", 0) + 1
            )
        if action == "PROFIT_TAKE_ALERT":
            daily_counts[day_key]["shield_profit_alerts"] = (
                daily_counts[day_key].get("shield_profit_alerts", 0) + 1
            )
        if action == "DATA_STALE":
            daily_counts[day_key]["data_stale"] = (
                daily_counts[day_key].get("data_stale", 0) + 1
            )
        if action == "DATA_QUALITY":
            daily_counts[day_key]["data_quality"] = (
                daily_counts[day_key].get("data_quality", 0) + 1
            )

    for detail in details_rows:
        day_key = detail["day"]
        key = (detail["category"], detail["ticker"])
        if key not in signals_by_day.get(day_key, set()):
            daily_counts.setdefault(day_key, {})
            daily_counts[day_key]["missing_signal_entries"] = (
                daily_counts[day_key].get("missing_signal_entries", 0) + 1
            )

    daily = []
    for day_key in sorted(daily_counts.keys()):
        row = {"date": day_key}
        row.update(daily_counts[day_key])
        daily.append(row)

    summary = {
        "total_days": len(daily),
        "total_entries": _sum_counts(daily, "ledger_entries"),
        "alpha_signals": _sum_counts(daily, "alpha_signals"),
        "fortress_signals": _sum_counts(daily, "fortress_signals"),
        "shield_strike_estimates": _sum_counts(daily, "shield_strike_estimates"),
        "shield_roll_alerts": _sum_counts(daily, "shield_roll_alerts"),
        "shield_profit_alerts": _sum_counts(daily, "shield_profit_alerts"),
        "data_stale": _sum_counts(daily, "data_stale"),
        "data_quality": _sum_counts(daily, "data_quality"),
        "missing_signal_entries": _sum_counts(daily, "missing_signal_entries"),
    }

    return SignalHealthReport(daily=daily, summary=summary)


def write_signal_health_report(
    config: AppConfig,
    output_dir: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Path:
    start = _parse_date(start_date) if start_date else None
    end = _parse_date(end_date) if end_date else None
    report = build_signal_health_report(config, start_date=start, end_date=end)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "all"
    if start_date or end_date:
        suffix = f"{start_date or 'start'}_{end_date or 'end'}"
    path = output_dir / f"signal_health_{suffix}.json"
    payload = {
        "summary": report.summary,
        "daily": report.daily,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
