import json
import tempfile
import unittest
from pathlib import Path

from sg_trader.config import AppConfig
from sg_trader.portfolio_dashboard import build_portfolio_dashboard, write_portfolio_dashboard


class PortfolioDashboardEmptyTests(unittest.TestCase):
    def test_empty_ledger_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            log_path = tmp_path / "ledger.json"
            log_path.write_text("[]", encoding="utf-8")
            config = AppConfig(
                telegram_token="",
                telegram_chat_id="",
                log_file=str(log_path),
                cache_dir=str(tmp_path / "cache"),
            )
            dashboard = build_portfolio_dashboard(config)
            self.assertEqual(dashboard.summary.get("module_count"), 0)
            self.assertEqual(dashboard.modules, {})
            self.assertEqual(dashboard.daily, [])

            path = write_portfolio_dashboard(config, tmp_path)
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertIn("summary", payload)
            self.assertIn("modules", payload)
            self.assertIn("daily", payload)


if __name__ == "__main__":
    unittest.main()
