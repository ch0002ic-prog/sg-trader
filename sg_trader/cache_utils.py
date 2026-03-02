from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json

import pandas as pd


@dataclass
class CacheEntry:
    timestamp: str
    value: float


def _cache_path(cache_dir: Path, name: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{name}.json"


def read_cache(cache_dir: Path, name: str) -> CacheEntry | None:
    path = _cache_path(cache_dir, name)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    timestamp = payload.get("timestamp")
    value = payload.get("value")
    if not timestamp or value is None:
        return None
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    return CacheEntry(timestamp=timestamp, value=value_float)


def write_cache(cache_dir: Path, name: str, value: float) -> None:
    path = _cache_path(cache_dir, name)
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "value": value,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def cache_age_hours(entry: CacheEntry) -> float | None:
    try:
        ts = datetime.strptime(entry.timestamp, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    delta = datetime.now() - ts
    return delta.total_seconds() / 3600.0


def _series_cache_path(cache_dir: Path, name: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{name}_series.json"


def write_series_cache(cache_dir: Path, name: str, series: pd.Series) -> None:
    path = _series_cache_path(cache_dir, name)
    payload = [
        {"timestamp": str(idx), "value": float(val)}
        for idx, val in series.items()
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_series_cache(
    cache_dir: Path, name: str, max_age_hours: int
) -> pd.Series | None:
    path = _series_cache_path(cache_dir, name)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, list) or not payload:
        return None
    timestamps = []
    values = []
    for item in payload:
        timestamp = item.get("timestamp")
        value = item.get("value")
        if timestamp is None or value is None:
            continue
        timestamps.append(timestamp)
        values.append(value)
    if not timestamps:
        return None
    try:
        last_dt = pd.to_datetime(timestamps[-1]).to_pydatetime()
    except Exception:
        return None
    if getattr(last_dt, "tzinfo", None) is not None:
        last_dt = last_dt.replace(tzinfo=None)
    age_hours = (datetime.now() - last_dt).total_seconds() / 3600.0
    if age_hours > max_age_hours:
        return None
    try:
        idx = pd.to_datetime(timestamps, utc=True).tz_convert(None)
        series = pd.Series(values, index=idx)
    except Exception:
        return None
    return series


def series_cache_age_hours(cache_dir: Path, name: str) -> float | None:
    path = _series_cache_path(cache_dir, name)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, list) or not payload:
        return None
    last_item = payload[-1]
    timestamp = last_item.get("timestamp") if isinstance(last_item, dict) else None
    if not timestamp:
        return None
    try:
        last_dt = pd.to_datetime(timestamp).to_pydatetime()
    except Exception:
        return None
    if getattr(last_dt, "tzinfo", None) is not None:
        last_dt = last_dt.replace(tzinfo=None)
    age_hours = (datetime.now() - last_dt).total_seconds() / 3600.0
    return age_hours
