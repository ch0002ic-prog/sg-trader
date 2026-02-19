import argparse
import unittest

from main import apply_strategy_profile, resolve_strategy_profile


class StrategyProfileTests(unittest.TestCase):
    def _base_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            strategy_profile="none",
            top_n=10,
            max_weight=0.30,
            min_score=None,
            max_annualized_volatility=None,
            max_lookback_drawdown=None,
            score_volatility_penalty=0.0,
            score_drawdown_penalty=0.0,
            regime_aware_defaults=False,
            regime_volatility_threshold=0.30,
            regime_score_threshold=0.0,
            regime_defensive_top_n=6,
            regime_defensive_max_weight=0.20,
        )

    def test_normal_profile_applies_defaults(self) -> None:
        args = self._base_args()
        args.strategy_profile = "normal"

        info = apply_strategy_profile(args, provided_flags=set())

        self.assertEqual(info["profile"], "normal")
        self.assertTrue(info["applied_overrides"])
        self.assertTrue(args.regime_aware_defaults)
        self.assertEqual(args.min_score, 0.0)
        self.assertEqual(args.max_annualized_volatility, 0.35)

    def test_defensive_profile_respects_explicit_flag(self) -> None:
        args = self._base_args()
        args.strategy_profile = "defensive"
        args.max_weight = 0.40

        info = apply_strategy_profile(args, provided_flags={"--max-weight"})

        self.assertEqual(args.max_weight, 0.40)
        self.assertNotIn("max_weight", info["applied_overrides"])
        self.assertEqual(args.top_n, 8)

    def test_none_profile_no_changes(self) -> None:
        args = self._base_args()

        info = apply_strategy_profile(args, provided_flags=set())

        self.assertEqual(info["profile"], "none")
        self.assertEqual(info["applied_overrides"], {})
        self.assertFalse(args.regime_aware_defaults)

    def test_resolve_strategy_profile_uses_config_default(self) -> None:
        resolved = resolve_strategy_profile(
            cli_profile="none",
            config_profile_default="aggressive",
            provided_flags=set(),
        )
        self.assertEqual(resolved, "aggressive")

    def test_resolve_strategy_profile_cli_flag_wins(self) -> None:
        resolved = resolve_strategy_profile(
            cli_profile="defensive",
            config_profile_default="aggressive",
            provided_flags={"--strategy-profile"},
        )
        self.assertEqual(resolved, "defensive")


if __name__ == "__main__":
    unittest.main()
