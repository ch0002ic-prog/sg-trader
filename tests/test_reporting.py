import unittest

from sg_trader import reporting


class ReportingSummaryTests(unittest.TestCase):
    def test_summary_includes_recommended_band(self) -> None:
        summary = reporting._build_summary([], "2026-02")
        band = summary.get("recommended_validation_band")
        self.assertIsNotNone(band)
        self.assertEqual(band["vvix_quantile"], 0.95)
        self.assertEqual(band["alpha_spread"], [5.2, 5.25])
        self.assertEqual(band["vvix_safe"], [177.5, 180.0])


if __name__ == "__main__":
    unittest.main()
