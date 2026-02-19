import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ExecutionIdOnlyTests(unittest.TestCase):
    def test_plan_id_only_output_and_artifact(self) -> None:
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
                    "--execution-plan",
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
                    "--execution-plan-id-only",
                    "--no-log",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            plan_id = result.stdout.strip()
            self.assertRegex(plan_id, r"^[a-f0-9]{32}$", msg=result.stdout)

            plan_path = (
                tmp_path
                / "reports"
                / "execution_plans"
                / f"execution_plan_{plan_id}.json"
            )
            self.assertTrue(plan_path.exists(), msg=f"Missing plan file: {plan_path}")

    def test_approve_id_only_output_and_artifact(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(repo_root)
            env["CACHE_DIR"] = str(tmp_path / ".cache")

            plan = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--execution-plan",
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
                    "--execution-plan-id-only",
                    "--no-log",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(plan.returncode, 0, msg=plan.stdout + plan.stderr)
            plan_id = plan.stdout.strip()
            self.assertTrue(re.fullmatch(r"[a-f0-9]{32}", plan_id))

            approve = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--execution-approve",
                    plan_id,
                    "--execution-approve-reason",
                    "ci-approval",
                    "--execution-approve-id-only",
                    "--no-log",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(approve.returncode, 0, msg=approve.stdout + approve.stderr)
            approved_id = approve.stdout.strip()
            self.assertEqual(approved_id, plan_id)

            approval_path = (
                tmp_path
                / "reports"
                / "execution_plans"
                / f"execution_approval_{approved_id}.json"
            )
            self.assertTrue(approval_path.exists(), msg=f"Missing approval file: {approval_path}")


if __name__ == "__main__":
    unittest.main()
