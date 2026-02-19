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


class PortfolioDashboardSummaryTests(unittest.TestCase):
    def test_summary_contains_totals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            log_path = tmp_path / "ledger.json"
            base = datetime(2026, 2, 1, 10, 0, 0)
            entries = [
                _entry(base.strftime("%Y-%m-%d %H:%M:%S"), "alpha", 1.25),
                _entry(base.strftime("%Y-%m-%d %H:%M:%S"), "shield", 0.75),
            ]
            log_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
            config = AppConfig(
                telegram_token="",
                telegram_chat_id="",
                log_file=str(log_path),
                cache_dir=str(tmp_path / "cache"),
            )

            write_portfolio_dashboard(config, tmp_path)
            md_path = tmp_path / "portfolio_dashboard_all.md"
            json_path = tmp_path / "portfolio_dashboard_all.json"
            content = md_path.read_text(encoding="utf-8")
            self.assertIn("Total equity (latest):", content)
            self.assertIn("Total realized (latest):", content)
            self.assertIn("Total unrealized (latest):", content)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            summary = payload.get("summary", {})
            self.assertIn("total_realized", summary)
            self.assertIn("total_unrealized", summary)
            self.assertIn("total_equity", summary)


if __name__ == "__main__":
    unittest.main()
