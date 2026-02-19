import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


class CliCapabilitiesTests(unittest.TestCase):
    def test_version_flag(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root)

        result = subprocess.run(
            [sys.executable, str(main_path), "--version"],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertRegex(result.stdout.strip(), r"^\d{4}\.\d{2}$")

    def test_cli_capabilities_json_flag(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root)

        result = subprocess.run(
            [sys.executable, str(main_path), "--cli-capabilities-json"],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        payload = json.loads(result.stdout.strip())

        self.assertEqual(payload.get("name"), "sg-trader")
        self.assertIn("version", payload)
        self.assertIn("commands", payload)
        self.assertIn("exit_codes", payload)

        commands = payload["commands"]
        self.assertIn("execution_plan", commands)
        self.assertIn("execution_replay", commands)
        self.assertIn("execution_ci_smoke", commands)

        self.assertIn("--execution-plan", commands["execution_plan"]["flags"])
        self.assertIn("--execution-replay-json", commands["execution_replay"]["flags"])
        self.assertIn("--execution-ci-smoke-json", commands["execution_ci_smoke"]["flags"])


if __name__ == "__main__":
    unittest.main()
