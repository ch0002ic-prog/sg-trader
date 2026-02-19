from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
import json
from typing import Any

from .config import AppConfig
from .logging_utils import load_ledger


@dataclass
class SlippageReport:
    summary: dict[str, Any]
    trades: list[dict[str, Any]]


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


def build_slippage_report(
    config: AppConfig,
    start_date: date | None = None,
    end_date: date | None = None,
) -> SlippageReport:
    entries = load_ledger(config)
    trades: list[dict[str, Any]] = []
    total_slippage = 0.0
    total_bps = 0.0
    count = 0
    manual_count = 0
    total_commission = 0.0

    for entry in entries:
        if entry.get("action") not in {"PAPER_BUY", "PAPER_SELL", "MANUAL_BUY", "MANUAL_SELL"}:
            continue
        timestamp = entry.get("timestamp", "")
        parsed = _parse_timestamp(timestamp)
        if not parsed:
            continue
        if not _in_range(parsed.date(), start_date, end_date):
            continue
        details = entry.get("details", {})
        reference = details.get("reference_price")
        fill = details.get("fill_price")
        quantity = details.get("quantity")
        side = entry.get("action")
        if not all(isinstance(value, (int, float)) for value in [reference, fill, quantity]):
            continue
        reference = float(reference)
        fill = float(fill)
        quantity = float(quantity)
        signed = 1.0 if "BUY" in side else -1.0
        slippage = (fill - reference) * signed * quantity
        slippage_bps = 0.0 if reference == 0 else (fill - reference) / reference * 10000

        if side in {"MANUAL_BUY", "MANUAL_SELL"}:
            manual_count += 1
        commission = details.get("commission")
        if isinstance(commission, (int, float)):
            total_commission += float(commission)

        trades.append(
            {
                "timestamp": timestamp,
                "symbol": entry.get("ticker", "Unknown"),
                "side": side,
                "quantity": quantity,
                "reference_price": reference,
                "fill_price": fill,
                "slippage": slippage,
                "slippage_bps": slippage_bps,
                "commission": details.get("commission"),
                "venue": details.get("venue"),
            }
        )
        total_slippage += slippage
        total_bps += slippage_bps
        count += 1

    avg_slippage = total_slippage / count if count else 0.0
    avg_bps = total_bps / count if count else 0.0

    avg_commission = total_commission / manual_count if manual_count else 0.0
    summary = {
        "total_trades": count,
        "manual_trades": manual_count,
        "total_slippage": total_slippage,
        "average_slippage": avg_slippage,
        "average_slippage_bps": avg_bps,
        "total_commission": total_commission,
        "average_commission": avg_commission,
    }
    return SlippageReport(summary=summary, trades=trades)


def write_slippage_report(
    config: AppConfig,
    output_dir: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Path:
    start = _parse_date(start_date) if start_date else None
    end = _parse_date(end_date) if end_date else None
    report = build_slippage_report(config, start_date=start, end_date=end)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "all"
    if start_date or end_date:
        suffix = f"{start_date or 'start'}_{end_date or 'end'}"
    path = output_dir / f"slippage_report_{suffix}.json"
    md_path = output_dir / f"slippage_report_{suffix}.md"
    payload = {
        "summary": report.summary,
        "trades": report.trades,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_lines = [
        "# Slippage Report",
        "",
        f"Total trades: {report.summary['total_trades']}",
        f"Manual trades: {report.summary['manual_trades']}",
        f"Total slippage: {report.summary['total_slippage']:.4f}",
        f"Average slippage: {report.summary['average_slippage']:.4f}",
        f"Average slippage (bps): {report.summary['average_slippage_bps']:.2f}",
        f"Total commission: {report.summary['total_commission']:.2f}",
        f"Average commission: {report.summary['average_commission']:.2f}",
        "",
        "## Trades",
        "",
        "timestamp | symbol | side | qty | reference | fill | slippage | bps | commission | venue",
        "--- | --- | --- | --- | --- | --- | --- | --- | --- | ---",
    ]
    for trade in report.trades:
        md_lines.append(
            f"{trade['timestamp']} | {trade['symbol']} | {trade['side']} | "
            f"{trade['quantity']:.0f} | {trade['reference_price']:.4f} | "
            f"{trade['fill_price']:.4f} | {trade['slippage']:.4f} | "
            f"{trade['slippage_bps']:.2f} | {trade.get('commission')} | "
            f"{trade.get('venue')}"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return path
