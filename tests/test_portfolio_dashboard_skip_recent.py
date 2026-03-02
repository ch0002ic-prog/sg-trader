import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from sg_trader.config import AppConfig
from sg_trader.portfolio_dashboard import write_portfolio_dashboard


def _entry(ts: str, module: str, pnl: float) -> dict[str, object]:
    return {
        "timestamp": ts,
        "category": "Execution",
        "ticker": "SPX_PUT",
        "action": "PAPER_PNL",
        "rationale": "test",
        "tags": [module],
        "details": {"unrealized_pnl": pnl, "module": module},
        "entry_type": "execution",
    }


class PortfolioDashboardSkipRecentTests(unittest.TestCase):
    def test_skip_recent_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            log_path = tmp_path / "ledger.json"
            base = datetime(2026, 2, 1, 10, 0, 0)
            entries = [
                _entry(base.strftime("%Y-%m-%d %H:%M:%S"), "alpha", 1.0),
                _entry(base.replace(day=2).strftime("%Y-%m-%d %H:%M:%S"), "alpha", 1.7),
                _entry(base.replace(day=3).strftime("%Y-%m-%d %H:%M:%S"), "alpha", 2.1),
                _entry(base.strftime("%Y-%m-%d %H:%M:%S"), "shield", 0.8),
                _entry(base.replace(day=2).strftime("%Y-%m-%d %H:%M:%S"), "shield", 0.6),
                _entry(base.replace(day=3).strftime("%Y-%m-%d %H:%M:%S"), "shield", 0.9),
            ]
            log_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
            config = AppConfig(
                telegram_token="",
                telegram_chat_id="",
                log_file=str(log_path),
                cache_dir=str(tmp_path / "cache"),
            )

            path = write_portfolio_dashboard(config, tmp_path, include_recent=False)
            md_path = tmp_path / "portfolio_dashboard_all.md"
            self.assertTrue(path.exists())
            self.assertTrue(md_path.exists())
            content = md_path.read_text(encoding="utf-8")
            self.assertNotIn("## Last 7 Days", content)
            self.assertNotIn("## Top Movers", content)


if __name__ == "__main__":
    unittest.main()
