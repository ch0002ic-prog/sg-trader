import unittest

from main import TickerMetrics, apply_strategy_filters


class StrategyFilterTests(unittest.TestCase):
    def test_filters_by_min_score(self) -> None:
        metrics = [
            TickerMetrics("AAA", 1.0, 0.1, 0.2, 0.5),
            TickerMetrics("BBB", 1.0, 0.1, 0.2, -0.1),
        ]

        filtered = apply_strategy_filters(
            metrics,
            min_score=0.0,
            max_annualized_volatility=None,
        )

        self.assertEqual([item.ticker for item in filtered], ["AAA"])

    def test_filters_by_max_annualized_volatility(self) -> None:
        metrics = [
            TickerMetrics("AAA", 1.0, 0.1, 0.15, 0.3),
            TickerMetrics("BBB", 1.0, 0.1, 0.45, 0.6),
        ]

        filtered = apply_strategy_filters(
            metrics,
            min_score=None,
            max_annualized_volatility=0.20,
        )

        self.assertEqual([item.ticker for item in filtered], ["AAA"])

    def test_filters_combined(self) -> None:
        metrics = [
            TickerMetrics("AAA", 1.0, 0.1, 0.10, 0.2),
            TickerMetrics("BBB", 1.0, 0.1, 0.25, 0.3),
            TickerMetrics("CCC", 1.0, 0.1, 0.10, -0.2),
        ]

        filtered = apply_strategy_filters(
            metrics,
            min_score=0.0,
            max_annualized_volatility=0.20,
        )

        self.assertEqual([item.ticker for item in filtered], ["AAA"])


if __name__ == "__main__":
    unittest.main()
