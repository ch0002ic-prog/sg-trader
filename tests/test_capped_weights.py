import unittest

from main import capped_weights


class CappedWeightsTests(unittest.TestCase):
    def test_respects_cap_when_feasible(self) -> None:
        weights = capped_weights([3.0, 2.0, 1.0], 0.5)
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertLessEqual(max(weights), 0.5 + 1e-12)

    def test_relaxes_cap_when_infeasible(self) -> None:
        weights = capped_weights([1.0, 1.0, 1.0], 0.2)
        self.assertAlmostEqual(sum(weights), 1.0)
        self.assertAlmostEqual(max(weights), 1.0 / 3.0)
        self.assertTrue(all(abs(value - (1.0 / 3.0)) < 1e-12 for value in weights))

    def test_zero_or_negative_scores_allocate_zero(self) -> None:
        weights = capped_weights([-1.0, 0.0, -2.0], 0.4)
        self.assertEqual(weights, [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
