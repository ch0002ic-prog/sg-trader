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


class PortfolioDashboardCorrelationTests(unittest.TestCase):
    def test_correlations_csv_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            log_path = tmp_path / "ledger.json"
            base = datetime(2026, 2, 1, 10, 0, 0)
            entries = [
                _entry(base.strftime("%Y-%m-%d %H:%M:%S"), "alpha", 1.0),
                _entry(base.strftime("%Y-%m-%d %H:%M:%S"), "shield", 0.5),
                _entry(base.replace(day=2).strftime("%Y-%m-%d %H:%M:%S"), "alpha", 1.7),
                _entry(base.replace(day=2).strftime("%Y-%m-%d %H:%M:%S"), "shield", 0.6),
                _entry(base.replace(day=3).strftime("%Y-%m-%d %H:%M:%S"), "alpha", 2.6),
                _entry(base.replace(day=3).strftime("%Y-%m-%d %H:%M:%S"), "shield", 0.4),
            ]
            log_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
            config = AppConfig(
                telegram_token="",
                telegram_chat_id="",
                log_file=str(log_path),
                cache_dir=str(tmp_path / "cache"),
            )

            path = write_portfolio_dashboard(config, tmp_path)
            corr_path = tmp_path / "portfolio_dashboard_all_correlations.csv"
            self.assertTrue(path.exists())
            self.assertTrue(corr_path.exists())
            content = corr_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertGreaterEqual(len(content), 2)
            self.assertEqual(content[0], "module,alpha,shield")
            self.assertTrue(content[1].startswith("alpha,"))
            self.assertTrue(content[2].startswith("shield,"))

    def test_correlations_csv_not_written_two_days(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            log_path = tmp_path / "ledger.json"
            base = datetime(2026, 2, 1, 10, 0, 0)
            entries = [
                _entry(base.strftime("%Y-%m-%d %H:%M:%S"), "alpha", 1.0),
                _entry(base.strftime("%Y-%m-%d %H:%M:%S"), "shield", 0.5),
                _entry(base.replace(day=2).strftime("%Y-%m-%d %H:%M:%S"), "alpha", 1.4),
                _entry(base.replace(day=2).strftime("%Y-%m-%d %H:%M:%S"), "shield", 0.8),
            ]
            log_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
            config = AppConfig(
                telegram_token="",
                telegram_chat_id="",
                log_file=str(log_path),
                cache_dir=str(tmp_path / "cache"),
            )

            path = write_portfolio_dashboard(config, tmp_path)
            corr_path = tmp_path / "portfolio_dashboard_all_correlations.csv"
            self.assertTrue(path.exists())
            self.assertFalse(corr_path.exists())

    def test_correlations_csv_not_written_single_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            log_path = tmp_path / "ledger.json"
            base = datetime(2026, 2, 1, 10, 0, 0)
            entries = [
                _entry(base.strftime("%Y-%m-%d %H:%M:%S"), "alpha", 1.0),
                _entry(base.replace(day=2).strftime("%Y-%m-%d %H:%M:%S"), "alpha", 1.5),
                _entry(base.replace(day=3).strftime("%Y-%m-%d %H:%M:%S"), "alpha", 1.8),
            ]
            log_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
            config = AppConfig(
                telegram_token="",
                telegram_chat_id="",
                log_file=str(log_path),
                cache_dir=str(tmp_path / "cache"),
            )

            path = write_portfolio_dashboard(config, tmp_path)
            corr_path = tmp_path / "portfolio_dashboard_all_correlations.csv"
            self.assertTrue(path.exists())
            self.assertFalse(corr_path.exists())


if __name__ == "__main__":
    unittest.main()
