import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class LocalCiParseTests(unittest.TestCase):
    def test_parse_from_stdin_success(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        parser_path = repo_root / "scripts" / "local_ci_parse.py"
        payload = {
            "ok": True,
            "strict": True,
            "smoke_rc": 0,
            "strict_validation_rc": 0,
            "unit_gates_rc": 0,
            "overall_rc": 0,
        }

        result = subprocess.run(
            [sys.executable, str(parser_path)],
            cwd=repo_root,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("local-ci status=PASS", result.stdout)
        self.assertIn("strict=yes", result.stdout)

    def test_parse_from_file_preserves_nonzero_overall_rc(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        parser_path = repo_root / "scripts" / "local_ci_parse.py"
        payload = {
            "ok": False,
            "strict": False,
            "smoke_rc": 0,
            "strict_validation_rc": 0,
            "unit_gates_rc": 6,
            "overall_rc": 6,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            payload_path = Path(tmp_dir) / "payload.json"
            payload_path.write_text(json.dumps(payload), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(parser_path), "--input", str(payload_path)],
                cwd=repo_root,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 6, msg=result.stdout + result.stderr)
        self.assertIn("local-ci status=FAIL", result.stdout)
        self.assertIn("strict=no", result.stdout)

    def test_parse_malformed_json_returns_2(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        parser_path = repo_root / "scripts" / "local_ci_parse.py"

        result = subprocess.run(
            [sys.executable, str(parser_path)],
            cwd=repo_root,
            input="{not-json}",
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 2, msg=result.stdout + result.stderr)
        self.assertIn("local-ci parse error:", result.stderr)

    def test_parse_missing_keys_returns_2(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        parser_path = repo_root / "scripts" / "local_ci_parse.py"

        result = subprocess.run(
            [sys.executable, str(parser_path)],
            cwd=repo_root,
            input=json.dumps({"ok": True}),
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 2, msg=result.stdout + result.stderr)
        self.assertIn("missing keys", result.stderr)


if __name__ == "__main__":
    unittest.main()
