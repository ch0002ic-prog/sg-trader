import json
from datetime import datetime
from pathlib import Path
from typing import Any

import time

import requests

from .config import AppConfig


LEDGER_SCHEMA_VERSION = 2


def send_telegram_alert(message: str, config: AppConfig) -> bool:
    if not config.telegram_token or not config.telegram_chat_id:
        print("Telegram credentials not set; skipping alert.")
        return False
    url = (
        f"https://api.telegram.org/bot{config.telegram_token}/sendMessage?"
        f"chat_id={config.telegram_chat_id}&text={requests.utils.quote(message)}"
    )
    retries = max(0, config.telegram_retries)
    timeout = config.telegram_timeout_seconds
    backoff = max(0.0, config.telegram_backoff_seconds)
    last_error = None
    for attempt in range(1, retries + 2):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return True
        except requests.RequestException as exc:
            last_error = exc
            if attempt <= retries:
                delay = backoff * (2 ** (attempt - 1))
                print(
                    "Telegram alert failed (attempt "
                    f"{attempt}/{retries + 1}): {exc}"
                )
                if delay > 0:
                    time.sleep(delay)
            else:
                print(f"Telegram alert failed: {exc}")
    return False


def _load_ledger(config: AppConfig) -> list[dict[str, Any]]:
    try:
        with open(config.log_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        path = Path(config.log_file)
        if path.exists():
            backup = path.with_suffix(
                path.suffix + ".corrupt-" + datetime.now().strftime("%Y%m%d%H%M%S")
            )
            path.rename(backup)
            print(f"Ledger was corrupt; moved to {backup}")
        return []


def load_ledger(config: AppConfig) -> list[dict[str, Any]]:
    return _load_ledger(config)


def _write_ledger(config: AppConfig, data: list[dict[str, Any]]) -> None:
    with open(config.log_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _is_duplicate_entry(
    data: list[dict[str, Any]], entry: dict[str, Any], window_seconds: int
) -> bool:
    if not data:
        return False
    last = data[-1]
    last_ts = last.get("timestamp")
    entry_ts = entry.get("timestamp")
    if not last_ts or not entry_ts:
        return False
    try:
        last_dt = datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S")
        entry_dt = datetime.strptime(entry_ts, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return False
    if abs((entry_dt - last_dt).total_seconds()) > window_seconds:
        return False
    keys = (
        "category",
        "ticker",
        "action",
        "rationale",
        "tags",
        "details",
        "entry_type",
    )
    return all(entry.get(key) == last.get(key) for key in keys)


def _is_daily_duplicate(data: list[dict[str, Any]], entry: dict[str, Any]) -> bool:
    entry_ts = entry.get("timestamp")
    if not entry_ts:
        return False
    try:
        entry_dt = datetime.strptime(entry_ts, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return False
    entry_date = entry_dt.date()
    keys = (
        "category",
        "ticker",
        "action",
        "rationale",
        "tags",
        "details",
        "entry_type",
    )
    for existing in reversed(data):
        existing_ts = existing.get("timestamp")
        if not existing_ts:
            continue
        try:
            existing_dt = datetime.strptime(existing_ts, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        if existing_dt.date() != entry_date:
            if existing_dt.date() < entry_date:
                break
            continue
        if all(entry.get(key) == existing.get(key) for key in keys):
            return True
    return False


def _dedup_key(entry: dict[str, Any]) -> tuple[Any, ...]:
    timestamp = entry.get("timestamp", "")
    day = timestamp.split(" ", 1)[0] if timestamp else ""
    tags = json.dumps(entry.get("tags", []), sort_keys=True)
    details = json.dumps(entry.get("details", {}), sort_keys=True)
    return (
        day,
        entry.get("category"),
        entry.get("ticker"),
        entry.get("action"),
        entry.get("rationale"),
        tags,
        details,
        entry.get("entry_type"),
    )


def _classify_entry_type(category: str) -> str:
    if category.lower() == "execution":
        return "execution"
    return "decision"


def backfill_entry_types(config: AppConfig) -> tuple[int, Path]:
    data = _load_ledger(config)
    if not data:
        return 0, Path(config.log_file)
    updated = 0
    for entry in data:
        if entry.get("entry_type"):
            continue
        category = str(entry.get("category", ""))
        entry["entry_type"] = _classify_entry_type(category)
        updated += 1
    path = Path(config.log_file)
    backup = path.with_suffix(
        path.suffix + ".bak-" + datetime.now().strftime("%Y%m%d%H%M%S")
    )
    backup.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _write_ledger(config, data)
    return updated, backup


def backfill_schema_version(config: AppConfig) -> tuple[int, Path]:
    data = _load_ledger(config)
    if not data:
        return 0, Path(config.log_file)
    updated = 0
    for entry in data:
        version = entry.get("schema_version")
        if isinstance(version, int) and version >= LEDGER_SCHEMA_VERSION:
            continue
        entry["schema_version"] = LEDGER_SCHEMA_VERSION
        updated += 1
    path = Path(config.log_file)
    backup = path.with_suffix(
        path.suffix + ".bak-" + datetime.now().strftime("%Y%m%d%H%M%S")
    )
    backup.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _write_ledger(config, data)
    return updated, backup


def backfill_execution_modules(config: AppConfig) -> tuple[int, Path]:
    data = _load_ledger(config)
    if not data:
        return 0, Path(config.log_file)
    updated = 0
    for entry in data:
        if entry.get("category") != "Execution":
            continue
        details = entry.get("details")
        if not isinstance(details, dict):
            details = {}
            entry["details"] = details
        module = details.get("module")
        if module and str(module).lower() != "unattributed":
            continue
        ticker = str(entry.get("ticker", ""))
        upper = ticker.upper()
        if upper == "SPX_PUT":
            details["module"] = "shield"
        elif upper == "S-REIT_BASKET":
            details["module"] = "fortress"
        elif upper in {"^SPX", "SPX", "SPY"}:
            details["module"] = "alpha"
        elif upper == config.growth_ticker.upper():
            details["module"] = "growth"
        elif upper:
            details["module"] = "unattributed"
        updated += 1
    path = Path(config.log_file)
    backup = path.with_suffix(
        path.suffix + ".bak-" + datetime.now().strftime("%Y%m%d%H%M%S")
    )
    backup.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _write_ledger(config, data)
    return updated, backup


def migrate_ledger(config: AppConfig) -> tuple[dict[str, int], Path]:
    data = _load_ledger(config)
    if not data:
        return {"entry_type": 0, "modules": 0, "schema_version": 0}, Path(
            config.log_file
        )
    updated_entry_type = 0
    updated_modules = 0
    updated_schema = 0
    for entry in data:
        if not entry.get("entry_type"):
            category = str(entry.get("category", ""))
            entry["entry_type"] = _classify_entry_type(category)
            updated_entry_type += 1
        version = entry.get("schema_version")
        if not isinstance(version, int) or version < LEDGER_SCHEMA_VERSION:
            entry["schema_version"] = LEDGER_SCHEMA_VERSION
            updated_schema += 1
        if entry.get("category") == "Execution":
            details = entry.get("details")
            if not isinstance(details, dict):
                details = {}
                entry["details"] = details
            module = details.get("module")
            if not module or str(module).lower() == "unattributed":
                ticker = str(entry.get("ticker", ""))
                upper = ticker.upper()
                if upper == "SPX_PUT":
                    details["module"] = "shield"
                elif upper == "S-REIT_BASKET":
                    details["module"] = "fortress"
                elif upper in {"^SPX", "SPX", "SPY"}:
                    details["module"] = "alpha"
                elif upper == config.growth_ticker.upper():
                    details["module"] = "growth"
                elif upper:
                    details["module"] = "unattributed"
                updated_modules += 1
    path = Path(config.log_file)
    backup = path.with_suffix(
        path.suffix + ".bak-" + datetime.now().strftime("%Y%m%d%H%M%S")
    )
    backup.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _write_ledger(config, data)
    return (
        {
            "entry_type": updated_entry_type,
            "modules": updated_modules,
            "schema_version": updated_schema,
        },
        backup,
    )


def dedup_ledger(config: AppConfig) -> tuple[int, Path]:
    data = _load_ledger(config)
    if not data:
        return 0, Path(config.log_file)
    seen = set()
    deduped = []
    for entry in data:
        key = _dedup_key(entry)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    path = Path(config.log_file)
    backup = path.with_suffix(
        path.suffix + ".bak-" + datetime.now().strftime("%Y%m%d%H%M%S")
    )
    backup.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _write_ledger(config, deduped)
    removed = len(data) - len(deduped)
    return removed, backup


def log_transaction(
    category: str,
    ticker: str,
    action: str,
    rationale: str,
    config: AppConfig,
    tags: list[str] | None = None,
    details: dict[str, Any] | None = None,
    entry_type: str | None = None,
) -> None:
    resolved_entry_type = entry_type or _classify_entry_type(category)
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "category": category,
        "ticker": ticker,
        "action": action,
        "rationale": rationale,
        "tags": tags or [],
        "details": details or {},
        "entry_type": resolved_entry_type,
        "schema_version": LEDGER_SCHEMA_VERSION,
    }
    data = _load_ledger(config)
    if _is_duplicate_entry(data, entry, config.dedup_window_seconds):
        return
    if _is_daily_duplicate(data, entry):
        return
    data.append(entry)
    _write_ledger(config, data)


def log_daily_heartbeat(config: AppConfig, details: dict[str, Any]) -> None:
    today = datetime.now().strftime("%Y-%m-%d")
    data = _load_ledger(config)
    for entry in data:
        if (
            entry.get("action") == "HEARTBEAT"
            and entry.get("timestamp", "").startswith(today)
        ):
            return
    data.append(
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "category": "Compliance",
            "ticker": "N/A",
            "action": "HEARTBEAT",
            "rationale": "Yield Enhancement",
            "tags": ["yield_enhancement", "heartbeat"],
            "details": details,
            "entry_type": "decision",
            "schema_version": LEDGER_SCHEMA_VERSION,
        }
    )
    _write_ledger(config, data)


def log_daily_summary(
    config: AppConfig, details: dict[str, Any], force: bool = False
) -> None:
    today = datetime.now().strftime("%Y-%m-%d")
    data = _load_ledger(config)
    if force:
        data = [
            entry
            for entry in data
            if not (
                entry.get("action") == "DAILY_SUMMARY"
                and entry.get("timestamp", "").startswith(today)
            )
        ]
    else:
        for entry in data:
            if (
                entry.get("action") == "DAILY_SUMMARY"
                and entry.get("timestamp", "").startswith(today)
            ):
                return
    action_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    last_action_timestamp: dict[str, str] = {}
    for entry in data:
        timestamp = entry.get("timestamp", "")
        if not timestamp.startswith(today):
            continue
        if entry.get("action") == "DAILY_SUMMARY":
            continue
        action = entry.get("action", "Unknown")
        category = entry.get("category", "Unknown")
        action_counts[action] = action_counts.get(action, 0) + 1
        category_counts[category] = category_counts.get(category, 0) + 1
        if action != "Unknown":
            last_action_timestamp[action] = timestamp
    merged_details = {
        **details,
        "action_counts": action_counts,
        "category_counts": category_counts,
        "last_action_timestamp": last_action_timestamp,
    }
    data.append(
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "category": "Compliance",
            "ticker": "N/A",
            "action": "DAILY_SUMMARY",
            "rationale": "Yield Enhancement",
            "tags": ["yield_enhancement", "summary"],
            "details": merged_details,
            "entry_type": "decision",
            "schema_version": LEDGER_SCHEMA_VERSION,
        }
    )
    _write_ledger(config, data)


def check_ledger_consistency(config: AppConfig) -> dict[str, Any]:
    data = _load_ledger(config)
    signals = set()
    for entry in data:
        if entry.get("action") != "SIGNAL":
            continue
        timestamp = entry.get("timestamp", "")
        day = timestamp.split(" ", 1)[0] if timestamp else ""
        signals.add((day, entry.get("category"), entry.get("ticker")))

    missing = 0
    samples: list[dict[str, Any]] = []
    for entry in data:
        if entry.get("action") != "SIGNAL_DETAILS":
            continue
        timestamp = entry.get("timestamp", "")
        day = timestamp.split(" ", 1)[0] if timestamp else ""
        key = (day, entry.get("category"), entry.get("ticker"))
        if key in signals:
            continue
        missing += 1
        if len(samples) < 5:
            samples.append(
                {
                    "timestamp": timestamp,
                    "category": entry.get("category"),
                    "ticker": entry.get("ticker"),
                }
            )
    return {
        "missing_signal_for_details": missing,
        "sample_missing": samples,
    }
