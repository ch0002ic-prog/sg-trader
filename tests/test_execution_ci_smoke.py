import os
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ExecutionCiSmokeTests(unittest.TestCase):
    def test_execution_ci_smoke_creates_plan_and_approval(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(repo_root)
            env["CACHE_DIR"] = str(tmp_path / ".cache")

            result = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--execution-ci-smoke",
                    "--execution-broker",
                    "paper",
                    "--paper-symbol",
                    "SPX_PUT",
                    "--paper-side",
                    "SELL",
                    "--paper-qty",
                    "1",
                    "--paper-reference-price",
                    "1.25",
                    "--paper-seed",
                    "1",
                    "--no-log",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            self.assertIn("Execution CI smoke plan id:", result.stdout)
            self.assertIn("Execution replay:", result.stdout)

            plan_id_match = re.search(r"Execution CI smoke plan id:\s*([a-f0-9]+)", result.stdout)
            self.assertIsNotNone(plan_id_match, msg=result.stdout)
            plan_id = plan_id_match.group(1)

            plan_path = (
                tmp_path
                / "reports"
                / "execution_plans"
                / f"execution_plan_{plan_id}.json"
            )
            approval_path = (
                tmp_path
                / "reports"
                / "execution_plans"
                / f"execution_approval_{plan_id}.json"
            )
            self.assertTrue(plan_path.exists(), msg=f"Missing plan: {plan_path}")
            self.assertTrue(approval_path.exists(), msg=f"Missing approval: {approval_path}")

    def test_execution_ci_smoke_json_output(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(repo_root)
            env["CACHE_DIR"] = str(tmp_path / ".cache")

            result = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--execution-ci-smoke",
                    "--execution-ci-smoke-json",
                    "--execution-broker",
                    "paper",
                    "--paper-symbol",
                    "SPX_PUT",
                    "--paper-side",
                    "SELL",
                    "--paper-qty",
                    "1",
                    "--paper-reference-price",
                    "1.25",
                    "--paper-seed",
                    "1",
                    "--no-log",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            payload = json.loads(result.stdout.strip())

            self.assertRegex(payload["plan_id"], r"^[a-f0-9]{32}$")
            self.assertEqual(payload["broker"], "paper")
            self.assertIn("result", payload)
            self.assertEqual(payload["result"]["symbol"], "SPX_PUT")
            self.assertEqual(payload["result"]["side"], "SELL")

            self.assertTrue((tmp_path / payload["plan_path"]).exists())
            self.assertTrue((tmp_path / payload["approval_path"]).exists())


if __name__ == "__main__":
    unittest.main()
