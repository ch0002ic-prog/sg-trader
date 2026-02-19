import unittest

from main import TickerMetrics, apply_regime_overlay


class StrategyRegimeOverlayTests(unittest.TestCase):
    def test_defensive_overlay_tightens_when_enabled(self) -> None:
        metrics = [
            TickerMetrics("AAA", 1.0, 0.1, 0.45, 0.10, 0.2),
            TickerMetrics("BBB", 1.0, 0.1, 0.40, 0.12, 0.1),
        ]

        effective_top_n, effective_max_weight, regime = apply_regime_overlay(
            metrics=metrics,
            base_top_n=10,
            base_max_weight=0.30,
            regime_aware_defaults=True,
            regime_volatility_threshold=0.30,
            regime_score_threshold=0.0,
            regime_defensive_top_n=6,
            regime_defensive_max_weight=0.20,
        )

        self.assertEqual(effective_top_n, 6)
        self.assertAlmostEqual(effective_max_weight, 0.20)
        self.assertEqual(regime["regime"], "defensive")

    def test_defensive_detected_but_not_applied_when_disabled(self) -> None:
        metrics = [
            TickerMetrics("AAA", 1.0, 0.1, 0.45, 0.10, 0.2),
            TickerMetrics("BBB", 1.0, 0.1, 0.40, 0.12, 0.1),
        ]

        effective_top_n, effective_max_weight, regime = apply_regime_overlay(
            metrics=metrics,
            base_top_n=10,
            base_max_weight=0.30,
            regime_aware_defaults=False,
            regime_volatility_threshold=0.30,
            regime_score_threshold=0.0,
            regime_defensive_top_n=6,
            regime_defensive_max_weight=0.20,
        )

        self.assertEqual(effective_top_n, 10)
        self.assertAlmostEqual(effective_max_weight, 0.30)
        self.assertEqual(regime["regime"], "defensive")

    def test_normal_regime_keeps_base_settings(self) -> None:
        metrics = [
            TickerMetrics("AAA", 1.0, 0.1, 0.15, 0.08, 0.4),
            TickerMetrics("BBB", 1.0, 0.1, 0.18, 0.10, 0.3),
        ]

        effective_top_n, effective_max_weight, regime = apply_regime_overlay(
            metrics=metrics,
            base_top_n=10,
            base_max_weight=0.30,
            regime_aware_defaults=True,
            regime_volatility_threshold=0.30,
            regime_score_threshold=0.0,
            regime_defensive_top_n=6,
            regime_defensive_max_weight=0.20,
        )

        self.assertEqual(effective_top_n, 10)
        self.assertAlmostEqual(effective_max_weight, 0.30)
        self.assertEqual(regime["regime"], "normal")


if __name__ == "__main__":
    unittest.main()
