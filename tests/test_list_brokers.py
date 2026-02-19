import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ListBrokersTests(unittest.TestCase):
    def test_list_brokers_includes_new_adapters(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(repo_root)

            result = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--list-brokers",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            output = result.stdout
            self.assertIn("paper", output)
            self.assertIn("manual", output)
            self.assertIn("dry-run", output)
            self.assertIn("external", output)


if __name__ == "__main__":
    unittest.main()
