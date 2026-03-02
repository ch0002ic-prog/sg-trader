import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path


def _write_plan(root: Path, plan_id: str, payload: dict[str, object]) -> Path:
    plan_payload = json.dumps(payload, sort_keys=True)
    payload["plan_hash"] = hashlib.sha256(plan_payload.encode("utf-8")).hexdigest()
    plan_path = root / "reports" / "execution_plans" / f"execution_plan_{plan_id}.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return plan_path


def _write_approval(root: Path, plan_id: str, plan_hash: str) -> None:
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


def _env_for(repo_root: Path, tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    env["CACHE_DIR"] = str(tmp_path / ".cache")
    env["PAPER_VOL_KILL_THRESHOLD"] = "0"
    return env


class ExecutionFlowTests(unittest.TestCase):
    def test_execution_approve_blocks_missing_hash(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            plan_id = "missing_hash"
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
            }
            plan_path = _write_plan(tmp_path, plan_id, payload)
            plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
            plan_payload.pop("plan_hash", None)
            plan_path.write_text(json.dumps(plan_payload, indent=2), encoding="utf-8")
            env = _env_for(repo_root, tmp_path)

            approve = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--execution-approve",
                    plan_id,
                    "--no-log",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(approve.returncode, 5, msg=approve.stdout + approve.stderr)
            self.assertIn("missing plan_hash", approve.stdout.lower())

    def test_execution_approve_blocks_invalid_json(self) -> None:
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
            env = _env_for(repo_root, tmp_path)

            approve = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--execution-approve",
                    plan_id,
                    "--no-log",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(approve.returncode, 5, msg=approve.stdout + approve.stderr)
            self.assertIn("invalid execution plan json", approve.stdout.lower())

    def test_execution_approve_missing_plan(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            env = _env_for(repo_root, tmp_path)

            approve = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--execution-approve",
                    "missing_plan",
                    "--no-log",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(approve.returncode, 5, msg=approve.stdout + approve.stderr)
            self.assertIn("execution plan not found", approve.stdout.lower())

    def test_execution_approve_blocks_invalid_payload(self) -> None:
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
            env = _env_for(repo_root, tmp_path)

            approve = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--execution-approve",
                    plan_id,
                    "--no-log",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(approve.returncode, 5, msg=approve.stdout + approve.stderr)
            self.assertIn("invalid execution plan payload", approve.stdout.lower())

    def test_execution_approve_blocks_missing_hash_field(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            plan_id = "missing_hash_field"
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
            }
            plan_path = _write_plan(tmp_path, plan_id, payload)
            plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
            plan_payload.pop("plan_hash", None)
            plan_path.write_text(json.dumps(plan_payload, indent=2), encoding="utf-8")
            env = _env_for(repo_root, tmp_path)

            approve = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--execution-approve",
                    plan_id,
                    "--no-log",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(approve.returncode, 5, msg=approve.stdout + approve.stderr)
            self.assertIn("missing plan_hash", approve.stdout.lower())
    def test_execution_approve_rejects_bad_hash(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            plan_id = "bad_hash_plan"
            payload = {
                "plan_id": plan_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "broker": "paper",
                "symbol": "SPX_PUT",
                "side": "SELL",
                "quantity": 1.0,
                "reference_price": 1.25,
                "slippage_bps": 5.0,
                "latency_ms": 150,
                "expected_fill_range": {"min": 1.24375, "max": 1.25},
                "risk": {"blocked": False, "reasons": []},
            }
            plan_path = _write_plan(tmp_path, plan_id, payload)
            plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
            plan_payload["plan_hash"] = "bad_hash"
            plan_path.write_text(json.dumps(plan_payload, indent=2), encoding="utf-8")
            env = _env_for(repo_root, tmp_path)

            approve = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--execution-approve",
                    plan_id,
                    "--no-log",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(approve.returncode, 5, msg=approve.stdout + approve.stderr)
            self.assertIn("verification failed", approve.stdout.lower())

    def test_paper_execution_blocks_plan_mismatch(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            plan_id = "plan_mismatch"
            payload = {
                "plan_id": plan_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "broker": "paper",
                "symbol": "SPX_PUT",
                "side": "SELL",
                "quantity": 1.0,
                "reference_price": 1.25,
                "slippage_bps": 5.0,
                "latency_ms": 150,
                "expected_fill_range": {"min": 1.24375, "max": 1.25},
                "risk": {"blocked": False, "reasons": []},
            }
            plan_path = _write_plan(tmp_path, plan_id, payload)
            plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
            _write_approval(tmp_path, plan_id, plan_payload["plan_hash"])
            env = _env_for(repo_root, tmp_path)

            execute = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--paper-execution",
                    "--execution-plan-id",
                    plan_id,
                    "--paper-symbol",
                    "OTHER_SYMBOL",
                    "--paper-side",
                    "SELL",
                    "--paper-qty",
                    "1",
                    "--paper-reference-price",
                    "1.25",
                    "--paper-latency-ms",
                    "150",
                    "--no-log",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(execute.returncode, 6, msg=execute.stdout + execute.stderr)
            self.assertIn("plan mismatch", execute.stdout.lower())

    def test_paper_execution_missing_plan_id(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            env = _env_for(repo_root, tmp_path)

            execute = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--paper-execution",
                    "--no-log",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(execute.returncode, 6, msg=execute.stdout + execute.stderr)
            self.assertIn("missing --execution-plan-id", execute.stdout.lower())

    def test_paper_execution_blocks_invalid_plan_timestamp(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            plan_id = "plan_bad_timestamp"
            payload = {
                "plan_id": plan_id,
                "timestamp": "invalid",
                "broker": "paper",
                "symbol": "SPX_PUT",
                "side": "SELL",
                "quantity": 1.0,
                "reference_price": 1.25,
                "slippage_bps": 5.0,
                "latency_ms": 0,
                "expected_fill_range": {"min": 1.24375, "max": 1.25},
                "risk": {"blocked": False, "reasons": []},
            }
            plan_path = _write_plan(tmp_path, plan_id, payload)
            plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
            _write_approval(tmp_path, plan_id, plan_payload["plan_hash"])
            env = _env_for(repo_root, tmp_path)

            execute = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--paper-execution",
                    "--execution-plan-id",
                    plan_id,
                    "--paper-symbol",
                    "SPX_PUT",
                    "--paper-side",
                    "SELL",
                    "--paper-qty",
                    "1",
                    "--paper-reference-price",
                    "1.25",
                    "--paper-latency-ms",
                    "0",
                    "--no-log",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(execute.returncode, 6, msg=execute.stdout + execute.stderr)
            self.assertIn("invalid plan timestamp", execute.stdout.lower())

    def test_paper_execution_blocks_invalid_approval_json(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            plan_id = "approval_invalid_json"
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
            }
            plan_path = _write_plan(tmp_path, plan_id, payload)
            plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
            approval_path = (
                tmp_path
                / "reports"
                / "execution_plans"
                / f"execution_approval_{plan_id}.json"
            )
            approval_path.parent.mkdir(parents=True, exist_ok=True)
            approval_path.write_text("{", encoding="utf-8")
            env = _env_for(repo_root, tmp_path)

            execute = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--paper-execution",
                    "--execution-plan-id",
                    plan_id,
                    "--paper-symbol",
                    "SPX_PUT",
                    "--paper-side",
                    "SELL",
                    "--paper-qty",
                    "1",
                    "--paper-reference-price",
                    "1.25",
                    "--paper-latency-ms",
                    "0",
                    "--no-log",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(execute.returncode, 6, msg=execute.stdout + execute.stderr)
            self.assertIn("approval not found", execute.stdout.lower())

    def test_paper_execution_blocks_missing_approval(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            plan_id = "missing_approval"
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
            }
            _write_plan(tmp_path, plan_id, payload)
            env = _env_for(repo_root, tmp_path)

            execute = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--paper-execution",
                    "--execution-plan-id",
                    plan_id,
                    "--paper-symbol",
                    "SPX_PUT",
                    "--paper-side",
                    "SELL",
                    "--paper-qty",
                    "1",
                    "--paper-reference-price",
                    "1.25",
                    "--paper-latency-ms",
                    "0",
                    "--no-log",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(execute.returncode, 6, msg=execute.stdout + execute.stderr)
            self.assertIn("approval not found", execute.stdout.lower())

    def test_paper_execution_blocks_invalid_plan_json(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            plan_id = "invalid_plan_json"
            plan_path = (
                tmp_path
                / "reports"
                / "execution_plans"
                / f"execution_plan_{plan_id}.json"
            )
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_path.write_text("{", encoding="utf-8")
            env = _env_for(repo_root, tmp_path)

            execute = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--paper-execution",
                    "--execution-plan-id",
                    plan_id,
                    "--paper-symbol",
                    "SPX_PUT",
                    "--paper-side",
                    "SELL",
                    "--paper-qty",
                    "1",
                    "--paper-reference-price",
                    "1.25",
                    "--paper-latency-ms",
                    "0",
                    "--no-log",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(execute.returncode, 6, msg=execute.stdout + execute.stderr)
            self.assertIn("invalid execution plan json", execute.stdout.lower())

    def test_paper_execution_blocks_invalid_plan_payload(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            plan_id = "invalid_plan_payload"
            plan_path = (
                tmp_path
                / "reports"
                / "execution_plans"
                / f"execution_plan_{plan_id}.json"
            )
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
            env = _env_for(repo_root, tmp_path)

            execute = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--paper-execution",
                    "--execution-plan-id",
                    plan_id,
                    "--paper-symbol",
                    "SPX_PUT",
                    "--paper-side",
                    "SELL",
                    "--paper-qty",
                    "1",
                    "--paper-reference-price",
                    "1.25",
                    "--paper-latency-ms",
                    "0",
                    "--no-log",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(execute.returncode, 6, msg=execute.stdout + execute.stderr)
            self.assertIn("invalid execution plan payload", execute.stdout.lower())

    def test_paper_execution_blocks_missing_plan(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            env = _env_for(repo_root, tmp_path)

            execute = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--paper-execution",
                    "--execution-plan-id",
                    "missing_plan",
                    "--paper-symbol",
                    "SPX_PUT",
                    "--paper-side",
                    "SELL",
                    "--paper-qty",
                    "1",
                    "--paper-reference-price",
                    "1.25",
                    "--paper-latency-ms",
                    "0",
                    "--no-log",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(execute.returncode, 6, msg=execute.stdout + execute.stderr)
            self.assertIn("execution plan not found", execute.stdout.lower())

    def test_paper_execution_blocks_invalid_approval_payload(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            plan_id = "approval_invalid_payload"
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
            }
            plan_path = _write_plan(tmp_path, plan_id, payload)
            plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
            approval_path = (
                tmp_path
                / "reports"
                / "execution_plans"
                / f"execution_approval_{plan_id}.json"
            )
            approval_path.parent.mkdir(parents=True, exist_ok=True)
            approval_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
            env = _env_for(repo_root, tmp_path)

            execute = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--paper-execution",
                    "--execution-plan-id",
                    plan_id,
                    "--paper-symbol",
                    "SPX_PUT",
                    "--paper-side",
                    "SELL",
                    "--paper-qty",
                    "1",
                    "--paper-reference-price",
                    "1.25",
                    "--paper-latency-ms",
                    "0",
                    "--no-log",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(execute.returncode, 6, msg=execute.stdout + execute.stderr)
            self.assertIn("approval not found", execute.stdout.lower())

    def test_paper_execution_blocks_expired_approval(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            plan_id = "approval_expired"
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
            }
            plan_path = _write_plan(tmp_path, plan_id, payload)
            plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
            approval_path = (
                tmp_path
                / "reports"
                / "execution_plans"
                / f"execution_approval_{plan_id}.json"
            )
            approval_path.parent.mkdir(parents=True, exist_ok=True)
            approval = {
                "plan_id": plan_id,
                "plan_hash": plan_payload["plan_hash"],
                "timestamp": "2000-01-01 00:00:00",
                "reason": "test",
            }
            approval_path.write_text(json.dumps(approval, indent=2), encoding="utf-8")
            env = _env_for(repo_root, tmp_path)

            execute = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--paper-execution",
                    "--execution-plan-id",
                    plan_id,
                    "--execution-approval-max-age-hours",
                    "0.01",
                    "--paper-symbol",
                    "SPX_PUT",
                    "--paper-side",
                    "SELL",
                    "--paper-qty",
                    "1",
                    "--paper-reference-price",
                    "1.25",
                    "--paper-latency-ms",
                    "0",
                    "--no-log",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(execute.returncode, 6, msg=execute.stdout + execute.stderr)
            self.assertIn("approval expired", execute.stdout.lower())

    def test_paper_execution_blocks_invalid_approval_timestamp(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            plan_id = "approval_bad_timestamp"
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
            }
            plan_path = _write_plan(tmp_path, plan_id, payload)
            plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
            approval_path = (
                tmp_path
                / "reports"
                / "execution_plans"
                / f"execution_approval_{plan_id}.json"
            )
            approval_path.parent.mkdir(parents=True, exist_ok=True)
            approval = {
                "plan_id": plan_id,
                "plan_hash": plan_payload["plan_hash"],
                "timestamp": "invalid",
                "reason": "test",
            }
            approval_path.write_text(json.dumps(approval, indent=2), encoding="utf-8")
            env = _env_for(repo_root, tmp_path)

            execute = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--paper-execution",
                    "--execution-plan-id",
                    plan_id,
                    "--execution-approval-max-age-hours",
                    "0.01",
                    "--paper-symbol",
                    "SPX_PUT",
                    "--paper-side",
                    "SELL",
                    "--paper-qty",
                    "1",
                    "--paper-reference-price",
                    "1.25",
                    "--paper-latency-ms",
                    "0",
                    "--no-log",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(execute.returncode, 6, msg=execute.stdout + execute.stderr)
            self.assertIn("invalid approval timestamp", execute.stdout.lower())

    def test_paper_execution_allows_tolerance(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            plan_id = "tolerance_ok"
            payload = {
                "plan_id": plan_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "broker": "paper",
                "symbol": "SPX_PUT",
                "side": "SELL",
                "quantity": 1.0,
                "reference_price": 1.25,
                "slippage_bps": 5.0,
                "latency_ms": 150,
                "expected_fill_range": {"min": 1.24375, "max": 1.25},
                "risk": {"blocked": False, "reasons": []},
            }
            plan_path = _write_plan(tmp_path, plan_id, payload)
            plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
            _write_approval(tmp_path, plan_id, plan_payload["plan_hash"])
            env = _env_for(repo_root, tmp_path)

            execute = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--paper-execution",
                    "--execution-plan-id",
                    plan_id,
                    "--execution-plan-slippage-tolerance-bps",
                    "2",
                    "--execution-plan-latency-tolerance-ms",
                    "5",
                    "--paper-symbol",
                    "SPX_PUT",
                    "--paper-side",
                    "SELL",
                    "--paper-qty",
                    "1",
                    "--paper-reference-price",
                    "1.25",
                    "--paper-slippage-bps",
                    "6",
                    "--paper-latency-ms",
                    "152",
                    "--paper-seed",
                    "1",
                    "--no-log",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(execute.returncode, 0, msg=execute.stdout + execute.stderr)

    def test_paper_execution_blocks_plan_hash_mismatch(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            plan_id = "hash_mismatch"
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
            }
            plan_path = _write_plan(tmp_path, plan_id, payload)
            plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
            plan_payload["plan_hash"] = "bad_hash"
            plan_path.write_text(json.dumps(plan_payload, indent=2), encoding="utf-8")
            env = _env_for(repo_root, tmp_path)

            execute = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--paper-execution",
                    "--execution-plan-id",
                    plan_id,
                    "--paper-symbol",
                    "SPX_PUT",
                    "--paper-side",
                    "SELL",
                    "--paper-qty",
                    "1",
                    "--paper-reference-price",
                    "1.25",
                    "--paper-latency-ms",
                    "0",
                    "--no-log",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(execute.returncode, 6, msg=execute.stdout + execute.stderr)
            self.assertIn("plan hash mismatch", execute.stdout.lower())

    def test_paper_execution_blocks_broker_mismatch(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            plan_id = "broker_mismatch"
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
            }
            plan_path = _write_plan(tmp_path, plan_id, payload)
            plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
            _write_approval(tmp_path, plan_id, plan_payload["plan_hash"])
            env = _env_for(repo_root, tmp_path)

            execute = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--paper-execution",
                    "--execution-plan-id",
                    plan_id,
                    "--execution-broker",
                    "dry-run",
                    "--paper-symbol",
                    "SPX_PUT",
                    "--paper-side",
                    "SELL",
                    "--paper-qty",
                    "1",
                    "--paper-reference-price",
                    "1.25",
                    "--paper-latency-ms",
                    "0",
                    "--no-log",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(execute.returncode, 6, msg=execute.stdout + execute.stderr)
            self.assertIn("plan mismatch", execute.stdout.lower())

    def test_paper_execution_blocks_slippage_mismatch(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            plan_id = "slippage_mismatch"
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
            }
            plan_path = _write_plan(tmp_path, plan_id, payload)
            plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
            _write_approval(tmp_path, plan_id, plan_payload["plan_hash"])
            env = _env_for(repo_root, tmp_path)

            execute = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--paper-execution",
                    "--execution-plan-id",
                    plan_id,
                    "--paper-symbol",
                    "SPX_PUT",
                    "--paper-side",
                    "SELL",
                    "--paper-qty",
                    "1",
                    "--paper-reference-price",
                    "1.25",
                    "--paper-slippage-bps",
                    "7",
                    "--paper-latency-ms",
                    "0",
                    "--no-log",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(execute.returncode, 6, msg=execute.stdout + execute.stderr)
            self.assertIn("plan mismatch", execute.stdout.lower())

    def test_paper_execution_blocks_reference_price_mismatch(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            plan_id = "reference_price_mismatch"
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
            }
            plan_path = _write_plan(tmp_path, plan_id, payload)
            plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
            _write_approval(tmp_path, plan_id, plan_payload["plan_hash"])
            env = _env_for(repo_root, tmp_path)

            execute = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--paper-execution",
                    "--execution-plan-id",
                    plan_id,
                    "--paper-symbol",
                    "SPX_PUT",
                    "--paper-side",
                    "SELL",
                    "--paper-qty",
                    "1",
                    "--paper-reference-price",
                    "1.3",
                    "--paper-latency-ms",
                    "0",
                    "--no-log",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(execute.returncode, 6, msg=execute.stdout + execute.stderr)
            self.assertIn("plan mismatch", execute.stdout.lower())

    def test_paper_execution_blocks_side_mismatch(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            plan_id = "side_mismatch"
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
            }
            plan_path = _write_plan(tmp_path, plan_id, payload)
            plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
            _write_approval(tmp_path, plan_id, plan_payload["plan_hash"])
            env = _env_for(repo_root, tmp_path)

            execute = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--paper-execution",
                    "--execution-plan-id",
                    plan_id,
                    "--paper-symbol",
                    "SPX_PUT",
                    "--paper-side",
                    "BUY",
                    "--paper-qty",
                    "1",
                    "--paper-reference-price",
                    "1.25",
                    "--paper-latency-ms",
                    "0",
                    "--no-log",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(execute.returncode, 6, msg=execute.stdout + execute.stderr)
            self.assertIn("plan mismatch", execute.stdout.lower())

    def test_paper_execution_blocks_quantity_mismatch(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            plan_id = "quantity_mismatch"
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
            }
            plan_path = _write_plan(tmp_path, plan_id, payload)
            plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
            _write_approval(tmp_path, plan_id, plan_payload["plan_hash"])
            env = _env_for(repo_root, tmp_path)

            execute = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--paper-execution",
                    "--execution-plan-id",
                    plan_id,
                    "--paper-symbol",
                    "SPX_PUT",
                    "--paper-side",
                    "SELL",
                    "--paper-qty",
                    "2",
                    "--paper-reference-price",
                    "1.25",
                    "--paper-latency-ms",
                    "0",
                    "--no-log",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(execute.returncode, 6, msg=execute.stdout + execute.stderr)
            self.assertIn("plan mismatch", execute.stdout.lower())

    def test_paper_execution_blocks_latency_mismatch(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            plan_id = "latency_mismatch"
            payload = {
                "plan_id": plan_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "broker": "paper",
                "symbol": "SPX_PUT",
                "side": "SELL",
                "quantity": 1.0,
                "reference_price": 1.25,
                "slippage_bps": 5.0,
                "latency_ms": 150,
                "expected_fill_range": {"min": 1.24375, "max": 1.25},
                "risk": {"blocked": False, "reasons": []},
            }
            plan_path = _write_plan(tmp_path, plan_id, payload)
            plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
            _write_approval(tmp_path, plan_id, plan_payload["plan_hash"])
            env = _env_for(repo_root, tmp_path)

            execute = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--paper-execution",
                    "--execution-plan-id",
                    plan_id,
                    "--paper-symbol",
                    "SPX_PUT",
                    "--paper-side",
                    "SELL",
                    "--paper-qty",
                    "1",
                    "--paper-reference-price",
                    "1.25",
                    "--paper-latency-ms",
                    "0",
                    "--no-log",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(execute.returncode, 6, msg=execute.stdout + execute.stderr)
            self.assertIn("plan mismatch", execute.stdout.lower())

    def test_paper_execution_blocks_expired_plan(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        main_path = repo_root / "main.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            plan_id = "plan_expired"
            payload = {
                "plan_id": plan_id,
                "timestamp": "2000-01-01 00:00:00",
                "broker": "paper",
                "symbol": "SPX_PUT",
                "side": "SELL",
                "quantity": 1.0,
                "reference_price": 1.25,
                "slippage_bps": 5.0,
                "latency_ms": 0,
                "expected_fill_range": {"min": 1.24375, "max": 1.25},
                "risk": {"blocked": False, "reasons": []},
            }
            plan_path = _write_plan(tmp_path, plan_id, payload)
            plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
            _write_approval(tmp_path, plan_id, plan_payload["plan_hash"])
            env = _env_for(repo_root, tmp_path)

            execute = subprocess.run(
                [
                    sys.executable,
                    str(main_path),
                    "--paper-execution",
                    "--execution-plan-id",
                    plan_id,
                    "--execution-plan-max-age-hours",
                    "0.01",
                    "--paper-symbol",
                    "SPX_PUT",
                    "--paper-side",
                    "SELL",
                    "--paper-qty",
                    "1",
                    "--paper-reference-price",
                    "1.25",
                    "--no-log",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(execute.returncode, 6, msg=execute.stdout + execute.stderr)
            self.assertIn("plan expired", execute.stdout.lower())


if __name__ == "__main__":
    unittest.main()
