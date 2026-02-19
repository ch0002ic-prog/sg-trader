import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from sg_trader.config import AppConfig
from sg_trader.monitoring import build_monitoring_report


class MonitoringThrottleTests(unittest.TestCase):
    def test_summary_includes_throttle_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            log_path = tmp_path / "ledger.json"
            log_path.write_text("[]", encoding="utf-8")
            cache_dir = tmp_path / "cache"
            state_path = cache_dir / "monitoring_alert_state.json"
            cache_dir.mkdir(parents=True, exist_ok=True)
            state_payload = {
                "critical:heartbeat_missing": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            }
            state_path.write_text(json.dumps(state_payload, indent=2), encoding="utf-8")

            config = AppConfig(
                telegram_token="",
                telegram_chat_id="",
                cache_dir=str(cache_dir),
                log_file=str(log_path),
                monitoring_alert_throttle_hours=6.0,
            )

            report = build_monitoring_report(config)
            summary = report.summary
            self.assertIn("alerts_throttle_hours", summary)
            self.assertIn("alerts_suppressed_count", summary)
            self.assertGreaterEqual(summary["alerts_suppressed_count"], 1)
            self.assertIn("alerts_total_count", summary)


if __name__ == "__main__":
    unittest.main()
