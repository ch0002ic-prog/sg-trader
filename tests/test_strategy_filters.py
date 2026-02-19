import unittest

from main import TickerMetrics, apply_strategy_filters, selection_score


class StrategyFilterTests(unittest.TestCase):
    def test_filters_by_min_score(self) -> None:
        metrics = [
            TickerMetrics("AAA", 1.0, 0.1, 0.2, 0.10, 0.5),
            TickerMetrics("BBB", 1.0, 0.1, 0.2, 0.10, -0.1),
        ]

        filtered = apply_strategy_filters(
            metrics,
            min_score=0.0,
            max_annualized_volatility=None,
            max_lookback_drawdown=None,
        )

        self.assertEqual([item.ticker for item in filtered], ["AAA"])

    def test_filters_by_max_annualized_volatility(self) -> None:
        metrics = [
            TickerMetrics("AAA", 1.0, 0.1, 0.15, 0.10, 0.3),
            TickerMetrics("BBB", 1.0, 0.1, 0.45, 0.10, 0.6),
        ]

        filtered = apply_strategy_filters(
            metrics,
            min_score=None,
            max_annualized_volatility=0.20,
            max_lookback_drawdown=None,
        )

        self.assertEqual([item.ticker for item in filtered], ["AAA"])

    def test_filters_combined(self) -> None:
        metrics = [
            TickerMetrics("AAA", 1.0, 0.1, 0.10, 0.10, 0.2),
            TickerMetrics("BBB", 1.0, 0.1, 0.25, 0.10, 0.3),
            TickerMetrics("CCC", 1.0, 0.1, 0.10, 0.10, -0.2),
        ]

        filtered = apply_strategy_filters(
            metrics,
            min_score=0.0,
            max_annualized_volatility=0.20,
            max_lookback_drawdown=None,
        )

        self.assertEqual([item.ticker for item in filtered], ["AAA"])

    def test_filters_by_max_lookback_drawdown(self) -> None:
        metrics = [
            TickerMetrics("AAA", 1.0, 0.1, 0.10, 0.12, 0.2),
            TickerMetrics("BBB", 1.0, 0.1, 0.10, 0.38, 0.3),
        ]

        filtered = apply_strategy_filters(
            metrics,
            min_score=None,
            max_annualized_volatility=None,
            max_lookback_drawdown=0.20,
        )

        self.assertEqual([item.ticker for item in filtered], ["AAA"])

    def test_selection_score_penalizes_volatility_and_drawdown(self) -> None:
        item = TickerMetrics("AAA", 1.0, 0.1, 0.20, 0.25, 1.0)
        value = selection_score(
            item,
            score_volatility_penalty=1.0,
            score_drawdown_penalty=2.0,
        )
        self.assertAlmostEqual(value, 0.30)

    def test_selection_score_can_change_ranking(self) -> None:
        metrics = [
            TickerMetrics("AAA", 1.0, 0.1, 0.50, 0.40, 1.00),
            TickerMetrics("BBB", 1.0, 0.1, 0.10, 0.05, 0.80),
        ]
        ranked = sorted(
            metrics,
            key=lambda item: selection_score(
                item,
                score_volatility_penalty=1.0,
                score_drawdown_penalty=0.5,
            ),
            reverse=True,
        )
        self.assertEqual([item.ticker for item in ranked], ["BBB", "AAA"])


if __name__ == "__main__":
    unittest.main()
