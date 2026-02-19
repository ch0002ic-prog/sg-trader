import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path


class ExecutionReplayJsonTests(unittest.TestCase):
    def _write_plan(self, root: Path) -> str:
        plan_id = "test_plan"
        payload = {
            "plan_id": plan_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "broker": "paper",
            "symbol": "SPX_PUT",
            "side": "SELL",
            "quantity": 1.0,
            "reference_price": 1.25,
            "slippage_bps": 5.0,
            "latency_ms": 0,
            "expected_fill_range": {"min": 1.24375, "max": 1.25},
            "risk": {"blocked": False, "reasons": []},
            "correlation_id": "test-correlation",
        }
        encoded = json.dumps(payload, sort_keys=True)
        payload["plan_hash"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        plan_path = root / "reports" / "execution_plans" / f"execution_plan_{plan_id}.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return plan_id

    def _write_approval(self, root: Path, plan_id: str, plan_hash: str) -> None:
        approval = {
            "plan_id": plan_id,
            "plan_hash": plan_hash,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "reason": "test",
        }
        approval_path = (
            root / "reports" / "execution_plans" / f"execution_approval_{plan_id}.json"
        )
        approval_path.parent.mkdir(parents=True, exist_ok=True)
        approval_path.write_text(json.dumps(approval, indent=2), encoding="utf-8")

    def test_execution_replay_json_output(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            plan_id = self._write_plan(tmp_path)
            plan_path = (
                tmp_path
                / "reports"
                / "execution_plans"
                / f"execution_plan_{plan_id}.json"
            )
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self._write_approval(tmp_path, plan_id, str(plan["plan_hash"]))

            env = os.environ.copy()
            env["PYTHONPATH"] = str(repo_root)
            env["CACHE_DIR"] = str(tmp_path / ".cache")

            result = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--execution-replay",
                    plan_id,
                    "--execution-replay-json",
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
            self.assertEqual(payload["plan_id"], plan_id)
            self.assertEqual(payload["broker"], "paper")
            self.assertIn("result", payload)
            self.assertEqual(payload["result"]["symbol"], "SPX_PUT")
            self.assertEqual(payload["result"]["side"], "SELL")


if __name__ == "__main__":
    unittest.main()
