from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json

from .logging_utils import load_ledger
from .config import AppConfig


@dataclass
class PnlSummary:
    by_symbol: dict[str, float]
    total: float
    realized_by_symbol: dict[str, float]
    realized_total: float


def _parse_timestamp(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None


def build_paper_pnl_summary(config: AppConfig, date_key: str | None = None) -> PnlSummary:
    entries = load_ledger(config)
    by_symbol: dict[str, float] = {}
    realized_by_symbol: dict[str, float] = {}
    if date_key is None:
        date_key = datetime.now().strftime("%Y-%m-%d")
    for entry in entries:
        if entry.get("action") not in {"PAPER_PNL", "PAPER_REALIZED"}:
            continue
        timestamp = entry.get("timestamp", "")
        parsed = _parse_timestamp(timestamp)
        if not parsed:
            continue
        if parsed.strftime("%Y-%m-%d") != date_key:
            continue
        details = entry.get("details", {})
        symbol = entry.get("ticker", "Unknown")
        if entry.get("action") == "PAPER_PNL":
            pnl = details.get("unrealized_pnl")
            if isinstance(pnl, (int, float)):
                by_symbol[symbol] = by_symbol.get(symbol, 0.0) + float(pnl)
        else:
            pnl = details.get("realized_pnl")
            if isinstance(pnl, (int, float)):
                realized_by_symbol[symbol] = realized_by_symbol.get(symbol, 0.0) + float(pnl)
    total = sum(by_symbol.values())
    realized_total = sum(realized_by_symbol.values())
    return PnlSummary(
        by_symbol=by_symbol,
        total=total,
        realized_by_symbol=realized_by_symbol,
        realized_total=realized_total,
    )


def write_paper_pnl_report(
    config: AppConfig, output_dir: str | Path, date_key: str | None = None
) -> Path:
    summary = build_paper_pnl_summary(config, date_key=date_key)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if date_key is None:
        date_key = datetime.now().strftime("%Y-%m-%d")
    path = output_dir / f"paper_pnl_{date_key}.json"
    payload = {
        "date": date_key,
        "total_unrealized_pnl": summary.total,
        "by_symbol": summary.by_symbol,
        "total_realized_pnl": summary.realized_total,
        "realized_by_symbol": summary.realized_by_symbol,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
