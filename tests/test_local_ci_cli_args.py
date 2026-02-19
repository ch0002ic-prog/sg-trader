import subprocess
import unittest
from pathlib import Path


class LocalCiCliArgsTests(unittest.TestCase):
    def test_help_shows_supported_flags(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "local_ci.sh"

        result = subprocess.run(
            ["bash", str(script_path), "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("Usage:", result.stdout)
        self.assertIn("--strict", result.stdout)
        self.assertIn("--json", result.stdout)

    def test_unknown_argument_returns_code_2(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "local_ci.sh"

        result = subprocess.run(
            ["bash", str(script_path), "--not-a-real-flag"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 2, msg=result.stdout + result.stderr)
        self.assertIn("Unknown argument", result.stderr)


if __name__ == "__main__":
    unittest.main()
