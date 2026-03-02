import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from sg_trader.config import AppConfig
from sg_trader.pnl_dashboard import build_pnl_dashboard, format_pnl_performance_message


class PnlDashboardPerformanceTests(unittest.TestCase):
    def _entry(self, timestamp: str, action: str, ticker: str, details: dict) -> dict:
        return {
            "timestamp": timestamp,
            "category": "Execution",
            "ticker": ticker,
            "action": action,
            "rationale": "test",
            "details": details,
            "entry_type": "execution",
            "schema_version": 2,
        }

    def _config(self, ledger_path: Path) -> AppConfig:
        config = AppConfig(telegram_token="", telegram_chat_id="")
        config.log_file = str(ledger_path)
        config.paper_initial_capital = 1000.0
        config.risk_free_rate = 3.65
        return config

    def test_performance_metrics_from_pnl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ledger_path = tmp_path / "ledger.json"
            entries = [
                self._entry(
                    "2026-02-01 10:00:00",
                    "PAPER_PNL",
                    "SPX_PUT",
                    {"unrealized_pnl": 100.0},
                ),
                self._entry(
                    "2026-02-02 10:00:00",
                    "PAPER_REALIZED",
                    "SPX_PUT",
                    {"realized_pnl": 50.0},
                ),
                self._entry(
                    "2026-02-02 15:00:00",
                    "PAPER_PNL",
                    "SPX_PUT",
                    {"unrealized_pnl": 150.0},
                ),
                self._entry(
                    "2026-02-03 10:00:00",
                    "PAPER_PNL",
                    "SPX_PUT",
                    {"unrealized_pnl": 50.0},
                ),
            ]
            ledger_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

            config = self._config(ledger_path)

            dashboard = build_pnl_dashboard(config)
            performance = dashboard.summary.get("performance", {})

            self.assertEqual(performance.get("data_points"), 3)
            self.assertEqual(performance.get("period_days"), 2)
            self.assertAlmostEqual(performance.get("normalized_end", 0.0), 1.1, places=6)
            self.assertAlmostEqual(
                performance.get("cumulative_return", 0.0), 0.1, places=6
            )
            self.assertAlmostEqual(
                performance.get("max_drawdown_pct", 0.0), -1.0 / 12.0, places=6
            )
            self.assertIsNotNone(performance.get("cagr"))
            self.assertIsNotNone(performance.get("sharpe_ratio"))
            self.assertIn("sortino_ratio", performance)
            self.assertIsNotNone(performance.get("calmar_ratio"))
            self.assertIsNone(performance.get("rolling_30d_return"))
            self.assertIsNone(performance.get("rolling_90d_return"))
            self.assertGreaterEqual(performance.get("downside_points", 0), 1)
            self.assertIn("downside_sufficient", performance)
            self.assertIn("downside_days_needed", performance)

            message = format_pnl_performance_message(dashboard.summary)
            self.assertIn("Portfolio Performance", message)
            self.assertIn("Final Equity", message)

    def test_performance_metrics_with_empty_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ledger_path = tmp_path / "ledger.json"
            ledger_path.write_text("[]", encoding="utf-8")
            config = self._config(ledger_path)

            dashboard = build_pnl_dashboard(config)
            performance = dashboard.summary.get("performance", {})

            self.assertEqual(performance.get("data_points"), 0)
            self.assertIsNone(performance.get("normalized_end"))

            message = format_pnl_performance_message(dashboard.summary)
            self.assertIn("No PnL data available", message)

    def test_performance_metrics_with_single_day(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ledger_path = tmp_path / "ledger.json"
            entries = [
                self._entry(
                    "2026-02-01 10:00:00",
                    "PAPER_PNL",
                    "SPX_PUT",
                    {"unrealized_pnl": 100.0},
                )
            ]
            ledger_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
            config = self._config(ledger_path)

            dashboard = build_pnl_dashboard(config)
            performance = dashboard.summary.get("performance", {})

            self.assertEqual(performance.get("data_points"), 1)
            self.assertIsNone(performance.get("normalized_end"))

            message = format_pnl_performance_message(dashboard.summary)
            self.assertIn("No PnL data available", message)

    def test_performance_metrics_with_zero_initial_capital(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ledger_path = tmp_path / "ledger.json"
            entries = [
                self._entry(
                    "2026-02-01 10:00:00",
                    "PAPER_PNL",
                    "SPX_PUT",
                    {"unrealized_pnl": 100.0},
                ),
                self._entry(
                    "2026-02-02 10:00:00",
                    "PAPER_PNL",
                    "SPX_PUT",
                    {"unrealized_pnl": 110.0},
                ),
            ]
            ledger_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
            config = self._config(ledger_path)
            config.paper_initial_capital = 0.0

            dashboard = build_pnl_dashboard(config)
            performance = dashboard.summary.get("performance", {})

            self.assertEqual(performance.get("data_points"), 2)
            self.assertIsNone(performance.get("normalized_end"))

    def test_performance_metrics_with_negative_pnl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ledger_path = tmp_path / "ledger.json"
            entries = [
                self._entry(
                    "2026-02-01 10:00:00",
                    "PAPER_PNL",
                    "SPX_PUT",
                    {"unrealized_pnl": -50.0},
                ),
                self._entry(
                    "2026-02-02 10:00:00",
                    "PAPER_PNL",
                    "SPX_PUT",
                    {"unrealized_pnl": -100.0},
                ),
            ]
            ledger_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
            config = self._config(ledger_path)

            dashboard = build_pnl_dashboard(config)
            performance = dashboard.summary.get("performance", {})

            self.assertEqual(performance.get("data_points"), 2)
            self.assertLess(performance.get("normalized_end", 1.0), 1.0)
            self.assertLess(performance.get("cumulative_return", 0.0), 0.0)

    def test_performance_metrics_with_zero_risk_free_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ledger_path = tmp_path / "ledger.json"
            entries = [
                self._entry(
                    "2026-02-01 10:00:00",
                    "PAPER_PNL",
                    "SPX_PUT",
                    {"unrealized_pnl": 100.0},
                ),
                self._entry(
                    "2026-02-02 10:00:00",
                    "PAPER_PNL",
                    "SPX_PUT",
                    {"unrealized_pnl": 120.0},
                ),
                self._entry(
                    "2026-02-03 10:00:00",
                    "PAPER_PNL",
                    "SPX_PUT",
                    {"unrealized_pnl": 130.0},
                ),
            ]
            ledger_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
            config = self._config(ledger_path)
            config.risk_free_rate = 0.0

            dashboard = build_pnl_dashboard(config)
            performance = dashboard.summary.get("performance", {})

            self.assertIsNotNone(performance.get("sharpe_ratio"))

    def test_performance_metrics_with_flat_equity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ledger_path = tmp_path / "ledger.json"
            entries = [
                self._entry(
                    "2026-02-01 10:00:00",
                    "PAPER_PNL",
                    "SPX_PUT",
                    {"unrealized_pnl": 100.0},
                ),
                self._entry(
                    "2026-02-02 10:00:00",
                    "PAPER_PNL",
                    "SPX_PUT",
                    {"unrealized_pnl": 100.0},
                ),
                self._entry(
                    "2026-02-03 10:00:00",
                    "PAPER_PNL",
                    "SPX_PUT",
                    {"unrealized_pnl": 100.0},
                ),
            ]
            ledger_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
            config = self._config(ledger_path)

            dashboard = build_pnl_dashboard(config)
            performance = dashboard.summary.get("performance", {})

            self.assertEqual(performance.get("annualized_volatility"), 0.0)
            self.assertIsNone(performance.get("sharpe_ratio"))


if __name__ == "__main__":
    unittest.main()
