import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path


class ExecutionReplayTests(unittest.TestCase):
    def _write_plan(self, root: Path, timestamp: str | None = None) -> str:
        plan_id = "test_plan"
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        plan = {
            "plan_id": plan_id,
            "timestamp": timestamp,
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
        plan_payload = json.dumps(plan, sort_keys=True)
        plan["plan_hash"] = hashlib.sha256(plan_payload.encode("utf-8")).hexdigest()
        plan_path = root / "reports" / "execution_plans" / f"execution_plan_{plan_id}.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        return plan_id

    def _env_for(self, repo_root: Path, tmp_path: Path) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root)
        env["CACHE_DIR"] = str(tmp_path / ".cache")
        return env

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

    def _replay_args(
        self,
        main_path: Path,
        plan_id: str,
        extra_args: list[str] | None = None,
        include_seed: bool = True,
    ) -> list[str]:
        args = [sys.executable, str(main_path), "--execution-replay", plan_id]
        if include_seed:
            args.extend(["--paper-seed", "1"])
        if extra_args:
            args.extend(extra_args)
        args.append("--no-log")
        return args

    def test_execution_approve_and_replay(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            plan_id = self._write_plan(tmp_path)
            env = self._env_for(repo_root, tmp_path)

            approve = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--execution-approve",
                    plan_id,
                    "--execution-approve-reason",
                    "test approval",
                    "--no-log",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                approve.returncode,
                0,
                msg=approve.stdout + approve.stderr,
            )
            approval_path = (
                tmp_path
                / "reports"
                / "execution_plans"
                / f"execution_approval_{plan_id}.json"
            )
            self.assertTrue(approval_path.exists())

            replay = subprocess.run(
                self._replay_args(
                    main_path,
                    plan_id,
                    extra_args=["--execution-plan-max-age-hours", "48"],
                ),
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                replay.returncode,
                0,
                msg=replay.stdout + replay.stderr,
            )
            self.assertIn("Execution replay:", replay.stdout)

    def test_execution_replay_blocks_hash_mismatch(self) -> None:
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
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            payload["plan_hash"] = "bad_hash"
            plan_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            env = self._env_for(repo_root, tmp_path)

            replay = subprocess.run(
                self._replay_args(main_path, plan_id),
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(replay.returncode, 6, msg=replay.stdout + replay.stderr)
            self.assertIn("hash mismatch", replay.stdout.lower())

    def test_execution_replay_blocks_expired_plan(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            old_ts = "2000-01-01 00:00:00"
            plan_id = self._write_plan(tmp_path, timestamp=old_ts)
            env = self._env_for(repo_root, tmp_path)

            replay = subprocess.run(
                self._replay_args(
                    main_path,
                    plan_id,
                    extra_args=["--execution-plan-max-age-hours", "0.01"],
                ),
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(replay.returncode, 6, msg=replay.stdout + replay.stderr)
            self.assertIn("plan expired", replay.stdout.lower())

    def test_execution_replay_blocks_missing_approval(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            plan_id = self._write_plan(tmp_path)
            env = self._env_for(repo_root, tmp_path)

            replay = subprocess.run(
                self._replay_args(main_path, plan_id),
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(replay.returncode, 6, msg=replay.stdout + replay.stderr)
            self.assertIn("approval not found", replay.stdout.lower())

    def test_execution_replay_blocks_expired_approval(self) -> None:
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
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            approval_path = (
                tmp_path
                / "reports"
                / "execution_plans"
                / f"execution_approval_{plan_id}.json"
            )
            approval_path.parent.mkdir(parents=True, exist_ok=True)
            approval = {
                "plan_id": plan_id,
                "plan_hash": payload["plan_hash"],
                "timestamp": "2000-01-01 00:00:00",
                "reason": "test",
            }
            approval_path.write_text(json.dumps(approval, indent=2), encoding="utf-8")
            env = self._env_for(repo_root, tmp_path)

            replay = subprocess.run(
                self._replay_args(
                    main_path,
                    plan_id,
                    extra_args=["--execution-approval-max-age-hours", "0.01"],
                ),
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(replay.returncode, 6, msg=replay.stdout + replay.stderr)
            self.assertIn("approval expired", replay.stdout.lower())

    def test_execution_replay_blocks_invalid_approval_timestamp(self) -> None:
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
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            approval_path = (
                tmp_path
                / "reports"
                / "execution_plans"
                / f"execution_approval_{plan_id}.json"
            )
            approval_path.parent.mkdir(parents=True, exist_ok=True)
            approval = {
                "plan_id": plan_id,
                "plan_hash": payload["plan_hash"],
                "timestamp": "invalid",
                "reason": "test",
            }
            approval_path.write_text(json.dumps(approval, indent=2), encoding="utf-8")
            env = self._env_for(repo_root, tmp_path)

            replay = subprocess.run(
                self._replay_args(
                    main_path,
                    plan_id,
                    extra_args=["--execution-approval-max-age-hours", "0.01"],
                ),
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(replay.returncode, 6, msg=replay.stdout + replay.stderr)
            self.assertIn("invalid approval timestamp", replay.stdout.lower())

    def test_execution_replay_blocks_missing_plan_id(self) -> None:
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
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            payload.pop("plan_id", None)
            plan_payload = json.dumps(
                {k: v for k, v in payload.items() if k not in {"plan_hash"}},
                sort_keys=True,
            )
            payload["plan_hash"] = hashlib.sha256(
                plan_payload.encode("utf-8")
            ).hexdigest()
            plan_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            env = self._env_for(repo_root, tmp_path)

            replay = subprocess.run(
                self._replay_args(main_path, plan_id),
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(replay.returncode, 6, msg=replay.stdout + replay.stderr)
            self.assertIn("missing plan id", replay.stdout.lower())

    def test_execution_replay_blocks_approval_hash_mismatch(self) -> None:
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
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            self._write_approval(tmp_path, plan_id, "bad_hash")
            env = self._env_for(repo_root, tmp_path)

            replay = subprocess.run(
                self._replay_args(main_path, plan_id),
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(replay.returncode, 6, msg=replay.stdout + replay.stderr)
            self.assertIn("approval hash mismatch", replay.stdout.lower())

    def test_execution_replay_blocks_invalid_inputs(self) -> None:
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
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            payload["side"] = "HOLD"
            payload["quantity"] = 0.0
            plan_payload = json.dumps(
                {k: v for k, v in payload.items() if k not in {"plan_hash"}},
                sort_keys=True,
            )
            payload["plan_hash"] = hashlib.sha256(
                plan_payload.encode("utf-8")
            ).hexdigest()
            plan_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self._write_approval(tmp_path, plan_id, payload["plan_hash"])
            env = self._env_for(repo_root, tmp_path)

            replay = subprocess.run(
                self._replay_args(main_path, plan_id),
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(replay.returncode, 6, msg=replay.stdout + replay.stderr)
            self.assertIn("invalid plan inputs", replay.stdout.lower())

    def test_execution_replay_blocks_kill_switch(self) -> None:
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
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            self._write_approval(tmp_path, plan_id, payload["plan_hash"])
            env = self._env_for(repo_root, tmp_path)
            env["PAPER_KILL_SWITCH"] = "1"

            replay = subprocess.run(
                self._replay_args(main_path, plan_id),
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(replay.returncode, 8, msg=replay.stdout + replay.stderr)
            self.assertIn("execution replay blocked", replay.stdout.lower())

    def test_execution_replay_skips_risk_checks(self) -> None:
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
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            self._write_approval(tmp_path, plan_id, payload["plan_hash"])
            env = self._env_for(repo_root, tmp_path)
            env["PAPER_KILL_SWITCH"] = "1"

            replay = subprocess.run(
                self._replay_args(
                    main_path,
                    plan_id,
                    extra_args=["--execution-replay-skip-risk"],
                ),
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(replay.returncode, 0, msg=replay.stdout + replay.stderr)
            self.assertIn("Execution replay:", replay.stdout)

    def test_execution_replay_requires_seed(self) -> None:
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
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            self._write_approval(tmp_path, plan_id, payload["plan_hash"])
            env = self._env_for(repo_root, tmp_path)

            replay = subprocess.run(
                self._replay_args(main_path, plan_id, include_seed=False),
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(replay.returncode, 6, msg=replay.stdout + replay.stderr)
            self.assertIn("missing --paper-seed", replay.stdout.lower())

    def test_execution_replay_allows_random_seed(self) -> None:
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
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            self._write_approval(tmp_path, plan_id, payload["plan_hash"])
            env = self._env_for(repo_root, tmp_path)

            replay = subprocess.run(
                self._replay_args(
                    main_path,
                    plan_id,
                    extra_args=["--execution-replay-allow-random"],
                    include_seed=False,
                ),
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(replay.returncode, 0, msg=replay.stdout + replay.stderr)
            self.assertIn("Execution replay:", replay.stdout)

    def test_execution_replay_blocks_invalid_json(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            plan_id = "invalid_json"
            plan_path = (
                tmp_path
                / "reports"
                / "execution_plans"
                / f"execution_plan_{plan_id}.json"
            )
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_path.write_text("{", encoding="utf-8")
            env = self._env_for(repo_root, tmp_path)

            replay = subprocess.run(
                self._replay_args(main_path, plan_id),
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(replay.returncode, 6, msg=replay.stdout + replay.stderr)
            self.assertIn("invalid execution plan json", replay.stdout.lower())

    def test_execution_replay_blocks_missing_plan(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            env = self._env_for(repo_root, tmp_path)

            replay = subprocess.run(
                self._replay_args(main_path, "missing_plan"),
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(replay.returncode, 6, msg=replay.stdout + replay.stderr)
            self.assertIn("execution plan not found", replay.stdout.lower())

    def test_execution_replay_blocks_invalid_timestamp(self) -> None:
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
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            payload["timestamp"] = "invalid"
            plan_payload = json.dumps(
                {k: v for k, v in payload.items() if k not in {"plan_hash"}},
                sort_keys=True,
            )
            payload["plan_hash"] = hashlib.sha256(
                plan_payload.encode("utf-8")
            ).hexdigest()
            plan_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self._write_approval(tmp_path, plan_id, payload["plan_hash"])
            env = self._env_for(repo_root, tmp_path)

            replay = subprocess.run(
                self._replay_args(main_path, plan_id),
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(replay.returncode, 6, msg=replay.stdout + replay.stderr)
            self.assertIn("invalid plan timestamp", replay.stdout.lower())

    def test_execution_replay_blocks_invalid_approval_json(self) -> None:
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
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            approval_path = (
                tmp_path
                / "reports"
                / "execution_plans"
                / f"execution_approval_{plan_id}.json"
            )
            approval_path.parent.mkdir(parents=True, exist_ok=True)
            approval_path.write_text("{", encoding="utf-8")
            env = self._env_for(repo_root, tmp_path)

            replay = subprocess.run(
                self._replay_args(main_path, plan_id),
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(replay.returncode, 6, msg=replay.stdout + replay.stderr)
            self.assertIn("approval not found", replay.stdout.lower())

    def test_execution_replay_blocks_invalid_approval_payload(self) -> None:
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
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            approval_path = (
                tmp_path
                / "reports"
                / "execution_plans"
                / f"execution_approval_{plan_id}.json"
            )
            approval_path.parent.mkdir(parents=True, exist_ok=True)
            approval_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
            env = self._env_for(repo_root, tmp_path)

            replay = subprocess.run(
                self._replay_args(main_path, plan_id),
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(replay.returncode, 6, msg=replay.stdout + replay.stderr)
            self.assertIn("approval not found", replay.stdout.lower())

    def test_execution_replay_blocks_missing_approval_hash(self) -> None:
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
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            approval_path = (
                tmp_path
                / "reports"
                / "execution_plans"
                / f"execution_approval_{plan_id}.json"
            )
            approval_path.parent.mkdir(parents=True, exist_ok=True)
            approval = {
                "plan_id": plan_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "reason": "test",
            }
            approval_path.write_text(json.dumps(approval, indent=2), encoding="utf-8")
            env = self._env_for(repo_root, tmp_path)

            replay = subprocess.run(
                self._replay_args(main_path, plan_id),
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(replay.returncode, 6, msg=replay.stdout + replay.stderr)
            self.assertIn("approval hash mismatch", replay.stdout.lower())

    def test_execution_replay_blocks_invalid_plan_payload(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            plan_id = "invalid_payload"
            plan_path = (
                tmp_path
                / "reports"
                / "execution_plans"
                / f"execution_plan_{plan_id}.json"
            )
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
            env = self._env_for(repo_root, tmp_path)

            replay = subprocess.run(
                self._replay_args(main_path, plan_id),
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(replay.returncode, 6, msg=replay.stdout + replay.stderr)
            self.assertIn("invalid execution plan payload", replay.stdout.lower())

    def test_execution_replay_blocks_missing_plan_hash(self) -> None:
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
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            payload.pop("plan_hash", None)
            plan_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            env = self._env_for(repo_root, tmp_path)

            replay = subprocess.run(
                self._replay_args(main_path, plan_id),
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(replay.returncode, 6, msg=replay.stdout + replay.stderr)
            self.assertIn("missing plan_hash", replay.stdout.lower())

    def test_execution_replay_blocks_unknown_broker(self) -> None:
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
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            payload["broker"] = "unknown"
            plan_payload = json.dumps(
                {k: v for k, v in payload.items() if k not in {"plan_hash"}},
                sort_keys=True,
            )
            payload["plan_hash"] = hashlib.sha256(
                plan_payload.encode("utf-8")
            ).hexdigest()
            plan_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self._write_approval(tmp_path, plan_id, payload["plan_hash"])
            env = self._env_for(repo_root, tmp_path)

            replay = subprocess.run(
                self._replay_args(main_path, plan_id),
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(replay.returncode, 6, msg=replay.stdout + replay.stderr)
            self.assertIn("unknown broker", replay.stdout.lower())

    def test_execution_replay_allows_broker_override(self) -> None:
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
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
            payload["broker"] = "unknown"
            plan_payload = json.dumps(
                {k: v for k, v in payload.items() if k not in {"plan_hash"}},
                sort_keys=True,
            )
            payload["plan_hash"] = hashlib.sha256(
                plan_payload.encode("utf-8")
            ).hexdigest()
            plan_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self._write_approval(tmp_path, plan_id, payload["plan_hash"])
            env = self._env_for(repo_root, tmp_path)

            replay = subprocess.run(
                self._replay_args(
                    main_path,
                    plan_id,
                    extra_args=["--execution-replay-broker", "paper"],
                ),
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(replay.returncode, 0, msg=replay.stdout + replay.stderr)
            self.assertIn("Execution replay:", replay.stdout)




if __name__ == "__main__":
    unittest.main()
