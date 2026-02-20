import json
import os
import subprocess
import unittest
from pathlib import Path


class CiSmokeRobustnessTests(unittest.TestCase):
    def _extract_summary(self, stdout: str) -> dict:
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        self.assertTrue(lines, msg="ci_smoke produced no output")
        return json.loads(lines[-1])

    def test_ci_smoke_summary_reports_robustness_toggle(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "ci_smoke.sh"

        env = os.environ.copy()
        env["ENFORCE_ROBUSTNESS_GATE"] = "0"

        result = subprocess.run(
            ["bash", str(script_path)],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        payload = self._extract_summary(result.stdout)
        self.assertTrue(payload.get("ok"))
        self.assertFalse(payload.get("robustness_gate_enforced"))
        allocator = payload.get("allocator_robustness", {})
        self.assertEqual(allocator.get("rc"), 0)

    def test_ci_smoke_propagates_robustness_failure(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "ci_smoke.sh"

        env = os.environ.copy()
        env["ENFORCE_ROBUSTNESS_GATE"] = "1"
        env["ROBUSTNESS_STRATEGY_PROFILE"] = "aggressive"
        env["ROBUSTNESS_MAX_TOP3_CONCENTRATION"] = "1.0"
        env["ROBUSTNESS_MIN_EFFECTIVE_N"] = "1000.0"

        result = subprocess.run(
            ["bash", str(script_path)],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        payload = self._extract_summary(result.stdout)
        self.assertFalse(payload.get("ok"))
        self.assertTrue(payload.get("robustness_gate_enforced"))

        allocator = payload.get("allocator_robustness", {})
        self.assertEqual(allocator.get("rc"), 7)

        allocator_payload = allocator.get("payload") or {}
        gate = allocator_payload.get("robustness_gate") or {}
        self.assertTrue(gate.get("enabled"))
        self.assertFalse(gate.get("pass"))
        self.assertIn("effective_n", gate.get("breaches", []))

    def test_ci_smoke_regime_threshold_mode_emits_metadata(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script_path = repo_root / "scripts" / "ci_smoke.sh"

        env = os.environ.copy()
        env["ENFORCE_ROBUSTNESS_GATE"] = "1"
        env["ROBUSTNESS_THRESHOLD_MODE"] = "regime"
        env["ROBUSTNESS_MAX_TOP3_CONCENTRATION_NORMAL"] = "1.0"
        env["ROBUSTNESS_MIN_EFFECTIVE_N_NORMAL"] = "1.0"
        env["ROBUSTNESS_MAX_TOP3_CONCENTRATION_DEFENSIVE"] = "1.0"
        env["ROBUSTNESS_MIN_EFFECTIVE_N_DEFENSIVE"] = "1.0"

        result = subprocess.run(
            ["bash", str(script_path)],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        payload = self._extract_summary(result.stdout)
        self.assertEqual(payload.get("robustness_threshold_mode"), "regime")
        self.assertIn(payload.get("robustness_detected_regime"), {"normal", "defensive"})
        thresholds = payload.get("robustness_effective_thresholds") or {}
        self.assertEqual(float(thresholds.get("max_top3_concentration", 0.0)), 1.0)
        self.assertEqual(float(thresholds.get("min_effective_n", 0.0)), 1.0)


if __name__ == "__main__":
    unittest.main()
