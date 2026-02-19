import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class HealthcheckTests(unittest.TestCase):
    def test_healthcheck_json_success(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ledger_path = tmp_path / "fortress_alpha_ledger.json"
            ledger_path.write_text("[]", encoding="utf-8")

            env = os.environ.copy()
            env["PYTHONPATH"] = str(repo_root)
            env["CACHE_DIR"] = str(tmp_path / ".cache")

            result = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--ledger-path",
                    str(ledger_path),
                    "--healthcheck-json",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            payload = json.loads(result.stdout.strip())
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["checks"]["ledger_readable"]["ok"])
            self.assertTrue(payload["checks"]["reports_writable"]["ok"])
            self.assertTrue(payload["checks"]["execution_plans_writable"]["ok"])
            self.assertTrue(payload["checks"]["cache_writable"]["ok"])

    def test_healthcheck_json_failure_missing_ledger(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            missing_ledger = tmp_path / "missing_ledger.json"

            env = os.environ.copy()
            env["PYTHONPATH"] = str(repo_root)
            env["CACHE_DIR"] = str(tmp_path / ".cache")

            result = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--ledger-path",
                    str(missing_ledger),
                    "--healthcheck-json",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 9, msg=result.stdout + result.stderr)
            payload = json.loads(result.stdout.strip())
            self.assertFalse(payload["ok"])
            self.assertFalse(payload["checks"]["ledger_readable"]["ok"])
            self.assertIn("ledger not found", payload["checks"]["ledger_readable"].get("error", ""))


if __name__ == "__main__":
    unittest.main()
