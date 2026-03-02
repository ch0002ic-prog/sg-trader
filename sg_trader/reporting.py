from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class MonthlyReport:
    summary: dict[str, Any]
    entries: list[dict[str, Any]]


def _parse_timestamp(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None


def _month_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m")


def _build_summary(entries: list[dict[str, Any]], month_key: str) -> dict[str, Any]:
    category_counts = Counter(entry.get("category", "Unknown") for entry in entries)
    action_counts = Counter(entry.get("action", "Unknown") for entry in entries)
    entry_type_counts = Counter(
        entry.get("entry_type", "unknown") for entry in entries
    )
    tag_counts: Counter[str] = Counter()
    tickers = set()
    heartbeat_present = False
    manual_fills: dict[str, int] = {}

    for entry in entries:
        tickers.add(entry.get("ticker", "Unknown"))
        for tag in entry.get("tags", []) or []:
            tag_counts[tag] += 1
        if entry.get("action") == "HEARTBEAT":
            heartbeat_present = True
        if entry.get("action") in {"MANUAL_BUY", "MANUAL_SELL"}:
            for tag in entry.get("tags", []) or []:
                tag_value = str(tag).strip().lower()
                if tag_value in {"alpha", "fortress", "shield", "growth"}:
                    manual_fills[tag_value] = manual_fills.get(tag_value, 0) + 1
                    break

    return {
        "month": month_key,
        "total_entries": len(entries),
        "categories": dict(category_counts),
        "actions": dict(action_counts),
        "entry_types": dict(entry_type_counts),
        "tags": dict(tag_counts),
        "tickers": sorted(tickers),
        "heartbeat_present": heartbeat_present,
        "manual_fills_by_module": manual_fills,
        "recommended_validation_band": {
            "vvix_quantile": 0.95,
            "alpha_spread": [5.2, 5.25],
            "vvix_safe": [177.5, 180.0],
        },
    }


def generate_monthly_report(
    ledger_path: str | Path,
    output_dir: str | Path,
    month_key: str | None = None,
) -> tuple[Path, Path]:
    ledger_path = Path(ledger_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if month_key is None:
        month_key = datetime.now().strftime("%Y-%m")

    if ledger_path.exists():
        data = json.loads(ledger_path.read_text(encoding="utf-8"))
    else:
        data = []

    filtered = []
    for entry in data:
        timestamp = entry.get("timestamp", "")
        parsed = _parse_timestamp(timestamp)
        if not parsed:
            continue
        if _month_key(parsed) == month_key:
            filtered.append(entry)

    summary = _build_summary(filtered, month_key)
    recap_path = Path(output_dir) / "backtest_top_unique.csv"
    if recap_path.exists():
        try:
            import csv

            with recap_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                top = next(reader, None)
        except Exception:
            top = None
        if top:
            summary["validation_recap"] = {
                "q": top.get("q"),
                "alpha": top.get("alpha"),
                "vvix": top.get("vvix"),
                "trades": top.get("trades"),
                "score": top.get("score"),
                "robust": top.get("robust"),
            }
    report = MonthlyReport(summary=summary, entries=filtered)

    md_path = output_dir / f"iras_report_{month_key}.md"
    json_path = output_dir / f"iras_report_{month_key}.json"

    md_lines = [
        "# IRAS Monthly Report",
        "",
        f"Month: {month_key}",
        "",
        f"Total entries: {summary['total_entries']}",
        f"Heartbeat present: {summary['heartbeat_present']}",
        "",
        "## Categories",
    ]
    for key, value in summary["categories"].items():
        md_lines.append(f"- {key}: {value}")

    md_lines.append("")
    md_lines.append("## Actions")
    for key, value in summary["actions"].items():
        md_lines.append(f"- {key}: {value}")

    md_lines.append("")
    md_lines.append("## Entry Types")
    for key, value in summary["entry_types"].items():
        md_lines.append(f"- {key}: {value}")

    md_lines.append("")
    md_lines.append("## Tags")
    if summary["tags"]:
        for key, value in summary["tags"].items():
            md_lines.append(f"- {key}: {value}")
    else:
        md_lines.append("- None")

    md_lines.append("")
    md_lines.append("## Tickers")
    for ticker in summary["tickers"]:
        md_lines.append(f"- {ticker}")

    md_lines.append("")
    md_lines.append("## Recommended Validation Band")
    band = summary["recommended_validation_band"]
    md_lines.append(f"- vvix quantile: {band['vvix_quantile']}")
    md_lines.append(f"- alpha spread: {band['alpha_spread'][0]}-{band['alpha_spread'][1]}")
    md_lines.append(f"- vvix safe threshold: {band['vvix_safe'][0]}-{band['vvix_safe'][1]}")

    md_lines.append("")
    md_lines.append("## Validation Recap")
    md_lines.append(
        "- See reports/backtest_top_unique.md for the latest deduped table."
    )
    recap = summary.get("validation_recap")
    if recap:
        md_lines.append(
            f"- Top row: q={recap.get('q')}, alpha={recap.get('alpha')}, "
            f"vvix={recap.get('vvix')}, trades={recap.get('trades')}, "
            f"score={recap.get('score')}, robust={recap.get('robust')}"
        )

    md_lines.append("")
    md_lines.append("## Manual Fills")
    if summary.get("manual_fills_by_module"):
        for key, value in summary["manual_fills_by_module"].items():
            md_lines.append(f"- {key}: {value}")
    else:
        md_lines.append("- None")

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "summary": report.summary,
                "entries": report.entries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return md_path, json_path
