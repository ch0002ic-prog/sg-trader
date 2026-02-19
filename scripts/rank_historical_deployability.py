from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analyze_historical_wealth import (
    compute_forward_outcomes,
    compute_portfolio_path,
    load_ticker_frames,
    parse_dvol_from_label,
    parse_version_from_label,
    summarize_outcomes,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def minmax(series: pd.Series) -> pd.Series:
    lo = float(series.min())
    hi = float(series.max())
    if hi <= lo:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - lo) / (hi - lo)


def build_deployability_scores(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()

    scored["n_median"] = scored.groupby("horizon_days")["median_multiple"].transform(minmax)
    scored["n_p10"] = scored.groupby("horizon_days")["p10_multiple"].transform(minmax)
    scored["n_hit"] = scored.groupby("horizon_days")["hit_rate_gt_1"].transform(minmax)
    scored["n_p90"] = scored.groupby("horizon_days")["p90_multiple"].transform(minmax)

    scored["deployability_score"] = (
        0.45 * scored["n_median"]
        + 0.30 * scored["n_p10"]
        + 0.20 * scored["n_hit"]
        + 0.05 * scored["n_p90"]
    )
    return scored


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank labels by practical deployability across horizons")
    parser.add_argument("--labels", nargs="+", required=True, help="Run labels to compare")
    parser.add_argument("--horizons", nargs="+", type=int, default=[21, 63, 126], help="Forward horizons")
    parser.add_argument("--rerisk", type=float, default=1.06)
    parser.add_argument("--hold", type=float, default=0.00)
    parser.add_argument("--cap", type=float, default=0.21)
    parser.add_argument("--exclude", default="^VIX,^VVIX")
    args = parser.parse_args()

    exclude = {x.strip() for x in args.exclude.split(",") if x.strip()}

    rows: list[dict] = []
    for label in args.labels:
        version = parse_version_from_label(label)
        derisk = parse_dvol_from_label(label)
        frames = load_ticker_frames(label)
        _, portfolio = compute_portfolio_path(
            frames,
            rerisk=args.rerisk,
            hold=args.hold,
            cap=args.cap,
            derisk=derisk,
            exclude_tickers=exclude,
        )

        for horizon in args.horizons:
            outcomes = compute_forward_outcomes(portfolio, horizon_days=horizon)
            summary = summarize_outcomes(outcomes)
            rows.append(
                {
                    "label": label,
                    "version": version,
                    "derisk": derisk,
                    "horizon_days": horizon,
                    **summary,
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        raise SystemExit("No results produced")

    scored = build_deployability_scores(result)

    agg = (
        scored.groupby(["label", "version", "derisk"], as_index=False)
        .agg(
            deployability_score=("deployability_score", "mean"),
            mean_multiple=("mean_multiple", "mean"),
            median_multiple=("median_multiple", "mean"),
            p10_multiple=("p10_multiple", "mean"),
            hit_rate_gt_1=("hit_rate_gt_1", "mean"),
        )
        .sort_values(["deployability_score", "median_multiple", "p10_multiple"], ascending=[False, False, False])
        .reset_index(drop=True)
    )

    out_detail = REPORTS / "historical_deployability_detail.csv"
    out_rank = REPORTS / "historical_deployability_ranking.csv"
    scored.to_csv(out_detail, index=False)
    agg.to_csv(out_rank, index=False)

    print("saved", out_detail)
    print("saved", out_rank)
    print("ranking_top")
    print(agg.head(10).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
