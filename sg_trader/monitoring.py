from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, date
from pathlib import Path
import json
import re
from typing import Any

import numpy as np

from .config import AppConfig
from .cache_utils import (
    cache_age_hours,
    read_cache,
    read_series_cache,
    series_cache_age_hours,
)
from .logging_utils import load_ledger


@dataclass
class MonitoringReport:
    summary: dict[str, Any]
    alerts: list[str]
    checks: dict[str, Any]


def _parse_timestamp(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None


def _day_key(day: date) -> str:
    return day.strftime("%Y-%m-%d")


def _date_range(end: date, days: int) -> list[date]:
    return [end - timedelta(days=offset) for offset in range(days - 1, -1, -1)]


def _series_cache_key(ticker: str) -> str:
    return "market_" + re.sub(r"[^a-zA-Z0-9_-]", "_", ticker)


def _load_series(cache_dir: Path, ticker: str, max_age_hours: int) -> Any:
    return read_series_cache(cache_dir, _series_cache_key(ticker), max_age_hours)


def _zscore_latest(
    series: Any,
    window_days: int,
    min_points: int,
) -> dict[str, float] | None:
    if series is None or getattr(series, "empty", True):
        return None
    series = series.dropna()
    if series.empty:
        return None
    window = series.tail(window_days + 1)
    if len(window) <= 1:
        return None
    latest = float(window.iloc[-1])
    prior = window.iloc[:-1]
    if len(prior) < min_points:
        return None
    mean = float(prior.mean())
    std = float(prior.std(ddof=0))
    if std <= 0:
        return None
    zscore = (latest - mean) / std
    return {
        "latest": latest,
        "mean": mean,
        "std": std,
        "zscore": float(zscore),
        "points": float(len(prior)),
    }


def _alert_state_path(config: AppConfig) -> Path:
    return Path(config.cache_dir) / "monitoring_alert_state.json"


def _load_alert_state(config: AppConfig) -> dict[str, str]:
    path = _alert_state_path(config)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(k): str(v) for k, v in payload.items()}


def _save_alert_state(config: AppConfig, state: dict[str, str]) -> None:
    path = _alert_state_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def build_monitoring_report(config: AppConfig) -> MonitoringReport:
    entries = load_ledger(config)
    now = datetime.now()
    today = now.date()
    window_days = config.monitoring_window_days

    latest_summary = None
    latest_heartbeat = None
    signal_counts: dict[str, int] = {}

    for entry in entries:
        timestamp = entry.get("timestamp", "")
        parsed = _parse_timestamp(timestamp)
        if not parsed:
            continue
        day = parsed.date()
        day_key = _day_key(day)
        action = entry.get("action")

        if action == "DAILY_SUMMARY" and day == today:
            if latest_summary is None or parsed > latest_summary[0]:
                latest_summary = (parsed, entry)
        if action == "HEARTBEAT" and day == today:
            if latest_heartbeat is None or parsed > latest_heartbeat[0]:
                latest_heartbeat = (parsed, entry)
        if action == "SIGNAL":
            signal_counts[day_key] = signal_counts.get(day_key, 0) + 1

    alerts_detail: list[dict[str, str]] = []
    checks: dict[str, Any] = {}

    def record_alert(code: str, severity: str, message: str) -> None:
        alerts_detail.append({"code": code, "severity": severity, "message": message})

    checks["heartbeat_today"] = latest_heartbeat is not None
    checks["daily_summary_today"] = latest_summary is not None

    if not checks["heartbeat_today"]:
        record_alert("heartbeat_missing", "critical", "Missing daily heartbeat.")
    if not checks["daily_summary_today"]:
        record_alert("summary_missing", "critical", "Missing daily summary.")

    stale_keys: list[str] = []
    if latest_summary is not None:
        details = latest_summary[1].get("details", {})
        data_quality = details.get("data_quality") or {}
        threshold = details.get("data_freshness_days")
        if not isinstance(threshold, (int, float)):
            threshold = config.data_freshness_days
        for key, value in data_quality.items():
            if isinstance(value, (int, float)) and value > float(threshold):
                stale_keys.append(key)
    checks["stale_data_keys"] = stale_keys
    if stale_keys:
        record_alert(
            "stale_data",
            "warn",
            "Stale market data: " + ", ".join(sorted(stale_keys)),
        )

    data_health: dict[str, Any] = {}
    missing_sources: list[str] = []
    if latest_summary is not None:
        details = latest_summary[1].get("details", {})
        data_quality = details.get("data_quality") or {}
        for key in ("spx", "vix", "vvix"):
            source = data_quality.get(f"{key}_source")
            if source:
                data_health[f"{key}_source"] = source
                if source == "missing":
                    missing_sources.append(key)

    cache_dir = Path(config.cache_dir)
    data_health["spx_cache_age_hours"] = series_cache_age_hours(
        cache_dir, "market_^SPX"
    )
    data_health["vix_cache_age_hours"] = series_cache_age_hours(
        cache_dir, "market_^VIX"
    )
    data_health["vvix_cache_age_hours"] = series_cache_age_hours(
        cache_dir, "market_^VVIX"
    )

    mas_entry = read_cache(cache_dir, "mas_tbill_6m")
    data_health["mas_cache_age_hours"] = (
        cache_age_hours(mas_entry) if mas_entry else None
    )

    sources = {
        key: data_health.get(f"{key}_source")
        for key in ("spx", "vix", "vvix")
        if data_health.get(f"{key}_source")
    }
    cache_sources = [key for key, value in sources.items() if value == "cache"]
    live_sources = [key for key, value in sources.items() if value == "live"]
    mas_age = data_health.get("mas_cache_age_hours")
    mas_cache_stale = (
        isinstance(mas_age, (int, float))
        and mas_age > config.mas_cache_max_age_hours
    )
    mas_cache_warn = (
        isinstance(mas_age, (int, float))
        and mas_age >= config.mas_cache_max_age_hours * config.mas_cache_warn_pct
    )
    cache_warn_hours = float(config.market_cache_max_age_hours) * config.market_cache_warn_pct
    cache_stale: list[str] = []
    cache_warn: list[str] = []
    for key in ("spx", "vix", "vvix"):
        age = data_health.get(f"{key}_cache_age_hours")
        if not isinstance(age, (int, float)):
            continue
        if age > float(config.market_cache_max_age_hours):
            cache_stale.append(key)
        elif age >= cache_warn_hours:
            cache_warn.append(key)

    all_missing = len(missing_sources) == 3
    all_cache = len(cache_sources) == 3
    checks["data_health_summary"] = {
        "sources": sources,
        "missing_sources": missing_sources,
        "cache_sources": cache_sources,
        "live_sources": live_sources,
        "mas_cache_stale": mas_cache_stale,
        "mas_cache_warn": mas_cache_warn,
        "cache_stale": cache_stale,
        "cache_warn": cache_warn,
        "all_sources_missing": all_missing,
        "all_sources_cache": all_cache,
    }

    checks["data_health"] = data_health
    if missing_sources:
        record_alert(
            "missing_sources",
            "critical",
            "Missing market data source: " + ", ".join(missing_sources),
        )
    if all_missing:
        record_alert(
            "all_sources_missing",
            "critical",
            "Market data unavailable: all Yahoo sources missing.",
        )
    if all_cache:
        record_alert(
            "all_sources_cache",
            "warn",
            "Market data using cache only; live sources unavailable.",
        )
    if mas_cache_stale:
        record_alert("mas_cache_stale", "warn", "MAS cache stale; refresh data source.")
    elif mas_cache_warn:
        record_alert(
            "mas_cache_warn",
            "info",
            "MAS cache nearing max age; refresh recommended.",
        )
    if cache_stale:
        record_alert(
            "market_cache_stale",
            "warn",
            "Market data cache stale: " + ", ".join(sorted(cache_stale)),
        )
    if cache_warn:
        record_alert(
            "market_cache_warn",
            "info",
            "Market data cache nearing max age: " + ", ".join(sorted(cache_warn)),
        )

    days = _date_range(today, window_days)
    day_keys = [_day_key(day) for day in days]
    counts = [signal_counts.get(key, 0) for key in day_keys]
    checks["signal_counts"] = dict(zip(day_keys, counts))

    if len(counts) > 1:
        prior = counts[:-1]
        prior_days = len(prior)
        if prior_days >= config.monitoring_min_days:
            avg_prior = sum(prior) / prior_days
            spike_threshold = max(2.0, avg_prior * config.monitoring_spike_mult)
            checks["signal_avg_prior"] = avg_prior
            checks["signal_spike_threshold"] = spike_threshold
            if counts[-1] > spike_threshold:
                record_alert(
                    "signal_spike",
                    "warn",
                    "Signal spike detected: "
                    f"today {counts[-1]} > threshold {spike_threshold:.2f}",
                )

    drift_days = config.monitoring_signal_drift_window_days
    if len(counts) >= drift_days + config.monitoring_min_days:
        recent = counts[-drift_days:]
        prior = counts[:-drift_days]
        if prior:
            avg_recent = sum(recent) / len(recent)
            avg_prior = sum(prior) / len(prior)
            checks["signal_avg_recent"] = avg_recent
            checks["signal_drift_baseline"] = avg_prior
            checks["signal_drift_low_threshold"] = (
                avg_prior * config.monitoring_signal_drift_low_mult
            )
            if avg_prior > 0 and avg_recent < avg_prior * config.monitoring_signal_drift_low_mult:
                record_alert(
                    "signal_drift_low",
                    "warn",
                    "Signal drift low: "
                    f"recent {avg_recent:.2f} < "
                    f"{config.monitoring_signal_drift_low_mult:.2f}x prior {avg_prior:.2f}",
                )

    cache_dir = Path(config.cache_dir)
    vix_series = _load_series(cache_dir, config.vix_ticker, config.market_cache_max_age_hours)
    vvix_series = _load_series(cache_dir, config.vvix_ticker, config.market_cache_max_age_hours)
    spx_series = _load_series(cache_dir, config.ticker, config.market_cache_max_age_hours)

    regime_checks: dict[str, Any] = {
        "window_days": config.monitoring_regime_window_days,
        "zscore_threshold": config.monitoring_regime_zscore,
        "min_points": config.monitoring_regime_min_points,
    }

    vix_z = _zscore_latest(
        vix_series,
        config.monitoring_regime_window_days,
        config.monitoring_regime_min_points,
    )
    vvix_z = _zscore_latest(
        vvix_series,
        config.monitoring_regime_window_days,
        config.monitoring_regime_min_points,
    )

    if vix_z:
        vix_shift = abs(vix_z["zscore"]) >= config.monitoring_regime_zscore
        regime_checks["vix"] = {**vix_z, "shift": vix_shift}
        if vix_shift:
            severity = (
                "critical"
                if abs(vix_z["zscore"]) >= config.monitoring_regime_critical_zscore
                else "warn"
            )
            record_alert(
                "regime_vix",
                severity,
                "Regime shift (VIX): "
                f"z={vix_z['zscore']:.2f} latest={vix_z['latest']:.2f}",
            )
    else:
        regime_checks["vix"] = {"status": "missing"}

    if vvix_z:
        vvix_shift = abs(vvix_z["zscore"]) >= config.monitoring_regime_zscore
        regime_checks["vvix"] = {**vvix_z, "shift": vvix_shift}
        if vvix_shift:
            severity = (
                "critical"
                if abs(vvix_z["zscore"]) >= config.monitoring_regime_critical_zscore
                else "warn"
            )
            record_alert(
                "regime_vvix",
                severity,
                "Regime shift (VVIX): "
                f"z={vvix_z['zscore']:.2f} latest={vvix_z['latest']:.2f}",
            )
    else:
        regime_checks["vvix"] = {"status": "missing"}

    spread_z = None
    if (
        spx_series is not None
        and vix_series is not None
        and not spx_series.empty
        and not vix_series.empty
    ):
        joined = np.log(spx_series / spx_series.shift(1))
        rv = joined.rolling(window=30).std() * np.sqrt(252) * 100
        rv = rv.dropna()
        if not rv.empty:
            aligned = rv.to_frame("rv").join(vix_series.rename("vix"), how="inner")
            spread = aligned["vix"] - aligned["rv"]
            spread_z = _zscore_latest(
                spread,
                config.monitoring_regime_window_days,
                config.monitoring_regime_min_points,
            )

    if spread_z:
        spread_shift = abs(spread_z["zscore"]) >= config.monitoring_regime_zscore
        regime_checks["spread"] = {**spread_z, "shift": spread_shift}
        if spread_shift:
            severity = (
                "critical"
                if abs(spread_z["zscore"]) >= config.monitoring_regime_critical_zscore
                else "warn"
            )
            record_alert(
                "regime_spread",
                severity,
                "Regime shift (IV-RV spread): "
                f"z={spread_z['zscore']:.2f} latest={spread_z['latest']:.2f}",
            )
    else:
        regime_checks["spread"] = {"status": "missing"}

    checks["regime_shift"] = regime_checks

    throttle_hours = max(0.0, config.monitoring_alert_throttle_hours)
    state = _load_alert_state(config)
    active_alerts: list[dict[str, str]] = []
    suppressed: list[dict[str, str]] = []
    now_ts = now.strftime("%Y-%m-%d %H:%M:%S")
    for alert in alerts_detail:
        key = f"{alert['severity']}:{alert['code']}"
        last_ts = state.get(key)
        if last_ts:
            try:
                last_dt = datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                last_dt = None
            if last_dt is not None:
                age_hours = (now - last_dt).total_seconds() / 3600.0
                if age_hours < throttle_hours:
                    suppressed.append({**alert, "last_sent": last_ts})
                    continue
        active_alerts.append(alert)
        state[key] = now_ts
    if active_alerts:
        _save_alert_state(config, state)

    alerts = [
        f"[{a['severity'].upper()}] {a['message']}" for a in active_alerts
    ]
    checks["alerts_detail"] = active_alerts
    checks["alerts_suppressed"] = suppressed
    checks["alerts_throttle_hours"] = throttle_hours
    suppressed_count = len(suppressed)
    active_count = len(active_alerts)
    total_alerts = active_count + suppressed_count
    severity_counts: dict[str, int] = {}
    for alert in active_alerts:
        severity_counts[alert["severity"]] = severity_counts.get(alert["severity"], 0) + 1

    summary = {
        "date": _day_key(today),
        "window_days": window_days,
        "alert_count": len(alerts),
        "alert_severity_counts": severity_counts,
        "alerts_throttle_hours": throttle_hours,
        "alerts_suppressed_count": suppressed_count,
        "alerts_total_count": total_alerts,
    }

    return MonitoringReport(summary=summary, alerts=alerts, checks=checks)


def write_monitoring_report(config: AppConfig, output_dir: str | Path) -> Path:
    report = build_monitoring_report(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"monitoring_report_{report.summary['date']}.json"
    md_path = output_dir / f"monitoring_report_{report.summary['date']}.md"
    payload = {
        "summary": report.summary,
        "alerts": report.alerts,
        "checks": report.checks,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_lines = [
        "# Monitoring Report",
        "",
        f"Date: {report.summary['date']}",
        f"Window days: {report.summary['window_days']}",
        f"Alert count: {report.summary['alert_count']}",
        "",
        "## Alerts",
    ]
    if report.alerts:
        for alert in report.alerts:
            md_lines.append(f"- {alert}")
    else:
        md_lines.append("- None")
    suppressed = report.checks.get("alerts_suppressed", [])
    if report.checks.get("alerts_throttle_hours", 0) > 0 or suppressed:
        md_lines.append("")
        md_lines.append("## Alert Throttle")
        md_lines.append(
            f"- Throttle hours: {report.checks.get('alerts_throttle_hours')}"
        )
        md_lines.append(f"- Suppressed alerts: {len(suppressed)}")

    md_lines.append("")
    md_lines.append("## Data Health Summary")
    summary = report.checks.get("data_health_summary", {})
    sources = summary.get("sources", {})
    md_lines.append(f"- Sources: {sources}")
    md_lines.append(f"- Missing sources: {summary.get('missing_sources', [])}")
    md_lines.append(f"- Cache sources: {summary.get('cache_sources', [])}")
    md_lines.append(f"- Live sources: {summary.get('live_sources', [])}")
    md_lines.append(f"- MAS cache stale: {summary.get('mas_cache_stale')}")

    md_lines.append("")
    md_lines.append("## Checks")
    md_lines.append(f"- Heartbeat today: {report.checks.get('heartbeat_today')}")
    md_lines.append(
        f"- Daily summary today: {report.checks.get('daily_summary_today')}"
    )
    md_lines.append(
        f"- Stale data keys: {report.checks.get('stale_data_keys', [])}"
    )

    md_lines.append("")
    md_lines.append("## Signal Counts")
    for key, value in report.checks.get("signal_counts", {}).items():
        md_lines.append(f"- {key}: {value}")
    md_lines.append("")
    md_lines.append(
        f"Signal avg prior: {report.checks.get('signal_avg_prior')}"
    )
    md_lines.append(
        f"Signal spike threshold: {report.checks.get('signal_spike_threshold')}"
    )
    md_lines.append(
        f"Signal avg recent: {report.checks.get('signal_avg_recent')}"
    )
    md_lines.append(
        f"Signal drift baseline: {report.checks.get('signal_drift_baseline')}"
    )
    md_lines.append(
        f"Signal drift low threshold: {report.checks.get('signal_drift_low_threshold')}"
    )

    md_lines.append("")
    md_lines.append("## Regime Shift")
    regime = report.checks.get("regime_shift", {})
    md_lines.append(f"- Window days: {regime.get('window_days')}")
    md_lines.append(f"- Z-score threshold: {regime.get('zscore_threshold')}")
    md_lines.append(f"- Min points: {regime.get('min_points')}")
    for key in ("vix", "vvix", "spread"):
        details = regime.get(key, {})
        if details.get("status") == "missing":
            md_lines.append(f"- {key}: missing")
            continue
        if details:
            md_lines.append(
                f"- {key}: latest={details.get('latest'):.2f} "
                f"z={details.get('zscore'):.2f} shift={details.get('shift')}"
            )

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return path
