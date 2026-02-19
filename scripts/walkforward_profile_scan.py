#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import (
    TickerMetrics,
    _strategy_profile_defaults,
    apply_regime_overlay,
    apply_strategy_filters,
    capped_weights,
    extract_ledger_tickers,
    load_ledger,
    selection_score,
)
from sg_trader.signals import fetch_close_series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward scan for strategy profiles and filter/penalty controls.",
    )
    parser.add_argument("--ledger-path", default="fortress_alpha_ledger.json")
    parser.add_argument("--cache-dir", default=".cache")
    parser.add_argument("--market-cache-max-age-hours", type=int, default=48)
    parser.add_argument("--history-period", default="2y")
    parser.add_argument("--lookback-days", type=int, default=63)
    parser.add_argument("--forward-days", type=int, default=21)
    parser.add_argument("--windows", type=int, default=6)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--max-weight", type=float, default=0.30)
    parser.add_argument("--profiles", default="normal,defensive,aggressive")
    parser.add_argument(
        "--out-csv",
        default="reports/walkforward_profile_scan.csv",
        help="Summary CSV output path.",
    )
    parser.add_argument(
        "--out-md",
        default="reports/walkforward_profile_scan.md",
        help="Summary markdown output path.",
    )
    parser.add_argument(
        "--out-detail-csv",
        default="reports/walkforward_profile_scan_detail.csv",
        help="Per-window detail CSV output path.",
    )
    return parser.parse_args()


def _compute_metric_and_forward(
    closes: pd.Series,
    *,
    ticker: str,
    window_end_idx: int,
    lookback_days: int,
    forward_days: int,
    risk_free_rate: float,
) -> tuple[TickerMetrics, float] | None:
    start_idx = window_end_idx - lookback_days
    forward_idx = window_end_idx + forward_days
    if start_idx < 0 or forward_idx >= len(closes):
        return None

    window = closes.iloc[start_idx : window_end_idx + 1]
    returns = np.log(window / window.shift(1)).dropna()
    if returns.empty:
        return None

    running_max = window.cummax()
    drawdown_series = (window / running_max) - 1.0
    max_lookback_drawdown = float(abs(drawdown_series.min()))

    lookback_return = float((window.iloc[-1] / window.iloc[0]) - 1.0)
    annualized_vol = float(returns.std(ddof=1) * np.sqrt(252))
    if not np.isfinite(annualized_vol):
        return None

    vol_floor = max(annualized_vol, 1e-8)
    annualized_rf_window = risk_free_rate * (lookback_days / 252.0)
    score = float((lookback_return - annualized_rf_window) / vol_floor)

    metric = TickerMetrics(
        ticker=ticker,
        price=float(window.iloc[-1]),
        lookback_return=lookback_return,
        annualized_volatility=annualized_vol,
        max_lookback_drawdown=max_lookback_drawdown,
        score=score,
    )
    forward_return = float((closes.iloc[forward_idx] / closes.iloc[window_end_idx]) - 1.0)
    return metric, forward_return


def _profile_config(profile: str, args: argparse.Namespace) -> dict[str, Any]:
    config = {
        "top_n": args.top_n,
        "max_weight": args.max_weight,
        "min_score": None,
        "max_annualized_volatility": None,
        "max_lookback_drawdown": None,
        "score_volatility_penalty": 0.0,
        "score_drawdown_penalty": 0.0,
        "regime_aware_defaults": False,
        "regime_volatility_threshold": 0.30,
        "regime_score_threshold": 0.0,
        "regime_defensive_top_n": 6,
        "regime_defensive_max_weight": 0.20,
    }
    config.update(_strategy_profile_defaults(profile))
    return config


def _evaluate_profile_window(
    metrics_with_forward: list[tuple[TickerMetrics, float]],
    *,
    config: dict[str, Any],
) -> tuple[float, int] | None:
    metrics = [item for item, _ in metrics_with_forward]
    forward_map = {item.ticker: fwd for item, fwd in metrics_with_forward}

    filtered = apply_strategy_filters(
        metrics,
        min_score=config["min_score"],
        max_annualized_volatility=config["max_annualized_volatility"],
        max_lookback_drawdown=config["max_lookback_drawdown"],
    )
    if not filtered:
        return None

    effective_top_n, effective_max_weight, _ = apply_regime_overlay(
        metrics=filtered,
        base_top_n=int(config["top_n"]),
        base_max_weight=float(config["max_weight"]),
        regime_aware_defaults=bool(config["regime_aware_defaults"]),
        regime_volatility_threshold=float(config["regime_volatility_threshold"]),
        regime_score_threshold=float(config["regime_score_threshold"]),
        regime_defensive_top_n=int(config["regime_defensive_top_n"]),
        regime_defensive_max_weight=float(config["regime_defensive_max_weight"]),
    )

    ranked = sorted(
        filtered,
        key=lambda item: selection_score(
            item,
            score_volatility_penalty=float(config["score_volatility_penalty"]),
            score_drawdown_penalty=float(config["score_drawdown_penalty"]),
        ),
        reverse=True,
    )
    selected = ranked[:effective_top_n]
    if not selected:
        return None

    scores = [
        selection_score(
            item,
            score_volatility_penalty=float(config["score_volatility_penalty"]),
            score_drawdown_penalty=float(config["score_drawdown_penalty"]),
        )
        for item in selected
    ]
    weights = capped_weights(scores, effective_max_weight)

    portfolio_forward = 0.0
    for item, weight in zip(selected, weights):
        portfolio_forward += weight * forward_map[item.ticker]
    return float(portfolio_forward), len(selected)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_md(summary_rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Walk-Forward Profile Scan",
        "",
        "profile | windows_used | avg_forward_return | median_forward_return | win_rate | avg_selected_count",
        "--- | --- | --- | --- | --- | ---",
    ]
    for row in summary_rows:
        lines.append(
            f"{row['profile']} | {row['windows_used']} | {row['avg_forward_return']:.4f} | "
            f"{row['median_forward_return']:.4f} | {row['win_rate']:.2%} | {row['avg_selected_count']:.2f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    if not profiles:
        raise SystemExit("No profiles provided.")

    entries = load_ledger(Path(args.ledger_path))
    tradable_tickers, _ = extract_ledger_tickers(entries)
    if not tradable_tickers:
        raise SystemExit("No tradable tickers found in ledger.")

    cache_dir = Path(args.cache_dir)
    close_map: dict[str, pd.Series] = {}
    for ticker in tradable_tickers:
        series = fetch_close_series(
            ticker,
            period=args.history_period,
            cache_dir=cache_dir,
            max_age_hours=args.market_cache_max_age_hours,
        )
        if series is None or series.empty:
            continue
        close_map[ticker] = pd.Series(series).dropna()

    if not close_map:
        raise SystemExit("No market series available for tradable tickers.")

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for profile in profiles:
        config = _profile_config(profile, args)
        profile_forwards: list[float] = []
        selected_counts: list[int] = []

        for w in range(args.windows):
            metrics_with_forward: list[tuple[TickerMetrics, float]] = []
            for ticker, closes in close_map.items():
                end_idx = len(closes) - 1 - args.forward_days * (w + 1)
                item = _compute_metric_and_forward(
                    closes,
                    ticker=ticker,
                    window_end_idx=end_idx,
                    lookback_days=args.lookback_days,
                    forward_days=args.forward_days,
                    risk_free_rate=args.risk_free_rate,
                )
                if item is None:
                    continue
                metrics_with_forward.append(item)

            if not metrics_with_forward:
                continue

            evaluated = _evaluate_profile_window(metrics_with_forward, config=config)
            if evaluated is None:
                continue
            portfolio_forward, selected_count = evaluated
            profile_forwards.append(portfolio_forward)
            selected_counts.append(selected_count)
            detail_rows.append(
                {
                    "profile": profile,
                    "window_index": w + 1,
                    "forward_return": portfolio_forward,
                    "selected_count": selected_count,
                }
            )

        if profile_forwards:
            summary_rows.append(
                {
                    "profile": profile,
                    "windows_used": len(profile_forwards),
                    "avg_forward_return": float(mean(profile_forwards)),
                    "median_forward_return": float(median(profile_forwards)),
                    "win_rate": float(
                        sum(1 for value in profile_forwards if value > 0) / len(profile_forwards)
                    ),
                    "avg_selected_count": float(mean(selected_counts)),
                }
            )

    summary_rows.sort(key=lambda row: row["avg_forward_return"], reverse=True)

    _write_csv(summary_rows, Path(args.out_csv))
    _write_md(summary_rows, Path(args.out_md))
    _write_csv(detail_rows, Path(args.out_detail_csv))

    print(json.dumps({"summary_rows": summary_rows, "out_csv": args.out_csv, "out_md": args.out_md}, sort_keys=True))


if __name__ == "__main__":
    main()
