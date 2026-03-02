import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path


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


class PortfolioDashboardSkipRecentCliTests(unittest.TestCase):
    def test_skip_recent_cli(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ledger_path = tmp_path / "fortress_alpha_ledger.json"
            base = datetime(2026, 2, 1, 10, 0, 0)
            entries = [
                _entry(base.strftime("%Y-%m-%d %H:%M:%S"), "alpha", 1.0),
                _entry(base.replace(day=2).strftime("%Y-%m-%d %H:%M:%S"), "alpha", 1.4),
                _entry(base.replace(day=3).strftime("%Y-%m-%d %H:%M:%S"), "alpha", 1.8),
                _entry(base.strftime("%Y-%m-%d %H:%M:%S"), "shield", 0.8),
                _entry(base.replace(day=2).strftime("%Y-%m-%d %H:%M:%S"), "shield", 0.6),
                _entry(base.replace(day=3).strftime("%Y-%m-%d %H:%M:%S"), "shield", 0.9),
            ]
            ledger_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

            env = os.environ.copy()
            env["PYTHONPATH"] = str(repo_root)

            result = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--portfolio-dashboard",
                    "--portfolio-skip-recent",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)

            md_path = tmp_path / "reports" / "portfolio_dashboard_all.md"
            self.assertTrue(md_path.exists())
            content = md_path.read_text(encoding="utf-8")
            self.assertNotIn("## Last 7 Days", content)
            self.assertNotIn("## Top Movers", content)


if __name__ == "__main__":
    unittest.main()
