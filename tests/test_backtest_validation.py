import unittest

from sg_trader import backtest_validation


class BacktestValidationRankTests(unittest.TestCase):
    def test_rank_filters_min_trades(self) -> None:
        scenarios = [
            {
                "alpha_spread_threshold": 3.0,
                "vvix_safe_threshold": 110.0,
                "regime": "full",
                "stats": {
                    "return_to_vol": 1.2,
                    "return_to_vol_robust_penalized": 1.2,
                    "trade_count": 10,
                },
            },
            {
                "alpha_spread_threshold": 5.0,
                "vvix_safe_threshold": 130.0,
                "regime": "full",
                "stats": {
                    "return_to_vol": 1.1,
                    "return_to_vol_robust_penalized": 1.1,
                    "trade_count": 20,
                },
            },
        ]

        ranked = backtest_validation._rank_scenarios(scenarios, min_trades=15)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["alpha_spread_threshold"], 5.0)

    def test_rank_dedup_picks_best_score(self) -> None:
        scenarios = [
            {
                "alpha_spread_threshold": 3.0,
                "vvix_safe_threshold": 110.0,
                "regime": "full",
                "stats": {
                    "return_to_vol": 1.0,
                    "return_to_vol_robust_penalized": 1.0,
                    "trade_count": 20,
                },
            },
            {
                "alpha_spread_threshold": 3.0,
                "vvix_safe_threshold": 110.0,
                "regime": "high_vix",
                "stats": {
                    "return_to_vol": 1.3,
                    "return_to_vol_robust_penalized": 1.3,
                    "trade_count": 20,
                },
            },
        ]

        ranked = backtest_validation._rank_scenarios(
            scenarios, min_trades=15, dedupe=True
        )
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["stats"]["return_to_vol_robust_penalized"], 1.3)


if __name__ == "__main__":
    unittest.main()
