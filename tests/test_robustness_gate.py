import unittest

from main import _cli_capabilities_payload, evaluate_robustness_gate


class RobustnessGateTests(unittest.TestCase):
    def test_gate_passes_when_enabled_and_within_thresholds(self) -> None:
        allocations = [
            {"ticker": "A", "weight": 0.30},
            {"ticker": "B", "weight": 0.25},
            {"ticker": "C", "weight": 0.20},
            {"ticker": "D", "weight": 0.15},
            {"ticker": "E", "weight": 0.10},
        ]

        result = evaluate_robustness_gate(
            allocations=allocations,
            enabled=True,
            max_top3_concentration=0.80,
            min_effective_n=4.0,
        )

        self.assertTrue(result["pass"])
        self.assertEqual(result["breaches"], [])

    def test_gate_fails_on_top3_concentration(self) -> None:
        allocations = [
            {"ticker": "A", "weight": 0.40},
            {"ticker": "B", "weight": 0.35},
            {"ticker": "C", "weight": 0.20},
            {"ticker": "D", "weight": 0.05},
        ]

        result = evaluate_robustness_gate(
            allocations=allocations,
            enabled=True,
            max_top3_concentration=0.90,
            min_effective_n=3.0,
        )

        self.assertFalse(result["pass"])
        self.assertIn("top3_concentration", result["breaches"])

    def test_gate_fails_on_effective_n(self) -> None:
        allocations = [
            {"ticker": "A", "weight": 0.60},
            {"ticker": "B", "weight": 0.20},
            {"ticker": "C", "weight": 0.20},
        ]

        result = evaluate_robustness_gate(
            allocations=allocations,
            enabled=True,
            max_top3_concentration=1.0,
            min_effective_n=3.5,
        )

        self.assertFalse(result["pass"])
        self.assertIn("effective_n", result["breaches"])

    def test_gate_disabled_never_blocks(self) -> None:
        allocations = [
            {"ticker": "A", "weight": 0.70},
            {"ticker": "B", "weight": 0.20},
            {"ticker": "C", "weight": 0.10},
        ]

        result = evaluate_robustness_gate(
            allocations=allocations,
            enabled=False,
            max_top3_concentration=0.50,
            min_effective_n=10.0,
        )

        self.assertTrue(result["pass"])
        self.assertEqual(result["breaches"], [])

    def test_cli_capabilities_include_robustness_flags_and_exit_code(self) -> None:
        payload = _cli_capabilities_payload()

        allocator_flags = payload["commands"]["allocator"]["flags"]
        self.assertIn("--robustness-gate", allocator_flags)
        self.assertIn("--robustness-max-top3-concentration", allocator_flags)
        self.assertIn("--robustness-min-effective-n", allocator_flags)
        self.assertEqual(payload["exit_codes"]["7"], "allocator robustness gate failed")


if __name__ == "__main__":
    unittest.main()
