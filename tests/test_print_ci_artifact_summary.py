import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PrintCiArtifactSummaryTests(unittest.TestCase):
    def test_pass_when_both_payloads_ok(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "print_ci_artifact_summary.py"

        smoke_payload = {
            "ok": True,
            "healthcheck": {"rc": 0},
            "execution_ci_smoke": {"rc": 0},
        }
        local_payload = {
            "ok": True,
            "strict": True,
            "overall_rc": 0,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            smoke_path = tmp / "ci_smoke_summary.json"
            local_path = tmp / "local_ci_result.json"
            smoke_path.write_text(json.dumps(smoke_payload), encoding="utf-8")
            local_path.write_text(json.dumps(local_payload), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--smoke-path",
                    str(smoke_path),
                    "--local-ci-path",
                    str(local_path),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("status=PASS", result.stdout)

    def test_fail_on_missing_file(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "print_ci_artifact_summary.py"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            smoke_path = tmp / "ci_smoke_summary.json"
            smoke_path.write_text(
                json.dumps({"ok": True, "healthcheck": {"rc": 0}, "execution_ci_smoke": {"rc": 0}}),
                encoding="utf-8",
            )
            missing_local = tmp / "missing_local_ci.json"

            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--smoke-path",
                    str(smoke_path),
                    "--local-ci-path",
                    str(missing_local),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 1, msg=result.stdout + result.stderr)
        self.assertIn("status=FAIL", result.stdout)
        self.assertIn("missing:", result.stdout)


if __name__ == "__main__":
    unittest.main()
