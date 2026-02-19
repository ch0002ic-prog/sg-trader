import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from sg_trader.config import AppConfig
from sg_trader.monitoring import build_monitoring_report


def _write_ledger(path: Path, entries: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def _now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class MonitoringReportTests(unittest.TestCase):
    def test_missing_heartbeat_and_summary_alerts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ledger_path = tmp_path / "ledger.json"
            _write_ledger(ledger_path, [])
            config = AppConfig(
                telegram_token="",
                telegram_chat_id="",
                log_file=str(ledger_path),
                cache_dir=str(tmp_path / "cache"),
            )

            report = build_monitoring_report(config)
            self.assertEqual(report.summary.get("alert_count"), 2)
            severities = report.summary.get("alert_severity_counts", {})
            self.assertEqual(severities.get("critical"), 2)
            alerts = "\n".join(report.alerts)
            self.assertIn("Missing daily heartbeat", alerts)
            self.assertIn("Missing daily summary", alerts)

    def test_stale_data_and_missing_source_alerts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ledger_path = tmp_path / "ledger.json"
            timestamp = _now_ts()
            entries = [
                {
                    "timestamp": timestamp,
                    "action": "HEARTBEAT",
                    "details": {},
                },
                {
                    "timestamp": timestamp,
                    "action": "DAILY_SUMMARY",
                    "details": {
                        "data_freshness_days": 2,
                        "data_quality": {
                            "spx": 3,
                            "spx_source": "missing",
                        },
                    },
                },
            ]
            _write_ledger(ledger_path, entries)
            config = AppConfig(
                telegram_token="",
                telegram_chat_id="",
                log_file=str(ledger_path),
                cache_dir=str(tmp_path / "cache"),
            )

            report = build_monitoring_report(config)
            alerts = "\n".join(report.alerts)
            self.assertIn("Stale market data", alerts)
            self.assertIn("Missing market data source", alerts)
            severities = report.summary.get("alert_severity_counts", {})
            self.assertEqual(severities.get("critical"), 1)
            self.assertEqual(severities.get("warn"), 1)

    def test_summary_fields_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ledger_path = tmp_path / "ledger.json"
            timestamp = _now_ts()
            entries = [
                {
                    "timestamp": timestamp,
                    "action": "HEARTBEAT",
                    "details": {},
                },
                {
                    "timestamp": timestamp,
                    "action": "DAILY_SUMMARY",
                    "details": {
                        "data_quality": {},
                    },
                },
            ]
            _write_ledger(ledger_path, entries)
            config = AppConfig(
                telegram_token="",
                telegram_chat_id="",
                log_file=str(ledger_path),
                cache_dir=str(tmp_path / "cache"),
            )

            report = build_monitoring_report(config)
            summary = report.summary
            self.assertIn("date", summary)
            self.assertIn("window_days", summary)
            self.assertIn("alert_count", summary)
            self.assertIn("alert_severity_counts", summary)


if __name__ == "__main__":
    unittest.main()
