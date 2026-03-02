from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ROOT / "reports" / "selected_table_checks"
REPORTS = ROOT / "reports"


def load_decay_rows(label: str, exclude_tickers: set[str]) -> pd.DataFrame:
    files = sorted(CHECKS.glob(f"decay_checks_*_{label}.csv"))
    if not files:
        raise FileNotFoundError(f"No decay checks found for label: {label}")

    frames = []
    for path in files:
        ticker = path.name.split("decay_checks_")[1].split(f"_{label}.csv")[0]
        if ticker in exclude_tickers:
            continue
        df = pd.read_csv(path)
        needed = {"date", "regime_combo", "policy_action"}
        if not needed.issubset(df.columns):
            continue
        for col in [
            "fwd_10d",
            "fwd_5d",
            "vix_bucket",
            "trend_state",
            "drawdown_state",
            "vol_state",
            "decreasing",
            "rolling_sharpe",
        ]:
            if col not in df.columns:
                df[col] = pd.NA

        work = df[
            [
                "date",
                "regime_combo",
                "policy_action",
                "fwd_10d",
                "fwd_5d",
                "vix_bucket",
                "trend_state",
                "drawdown_state",
                "vol_state",
                "decreasing",
                "rolling_sharpe",
            ]
        ].copy()
        if "price" in df.columns:
            price = pd.to_numeric(df["price"], errors="coerce")
            derived_fwd_10d = (price.shift(-10) / price) - 1.0
            derived_fwd_5d = (price.shift(-5) / price) - 1.0
            work["fwd_10d"] = pd.to_numeric(work["fwd_10d"], errors="coerce").fillna(derived_fwd_10d)
            work["fwd_5d"] = pd.to_numeric(work["fwd_5d"], errors="coerce").fillna(derived_fwd_5d)

        work["ticker"] = ticker
        work["date"] = pd.to_datetime(work["date"], errors="coerce", utc=True)
        for col in ["fwd_10d", "fwd_5d"]:
            work[col] = pd.to_numeric(work[col], errors="coerce")
        work = work.dropna(subset=["date", "regime_combo", "policy_action", "fwd_10d"])
        if work.empty:
            continue
        frames.append(work)

    if not frames:
        raise RuntimeError("No usable regime rows found")
    return pd.concat(frames, ignore_index=True)


def parse_feature_sets(raw: str) -> list[list[str]]:
    feature_sets: list[list[str]] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        cols = [c.strip() for c in chunk.split("+") if c.strip()]
        if cols:
            feature_sets.append(cols)
    return feature_sets


def build_rs_bin(series: pd.Series, quantiles: int) -> pd.Series:
    ranks = series.rank(method="first")
    return pd.qcut(
        ranks,
        q=quantiles,
        labels=[f"rs_q{i}" for i in range(1, quantiles + 1)],
        duplicates="drop",
    )


def build_fine_suggestions(
    data: pd.DataFrame,
    *,
    feature_sets: list[list[str]],
    min_samples: int,
    min_edge: float,
    focus_action: str,
    against_action: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for features in feature_sets:
        if any(col not in data.columns for col in features):
            continue

        grouped = (
            data.groupby(features + ["policy_action"], dropna=False, observed=False)
            .agg(samples=("fwd_10d", "count"), mean_fwd10=("fwd_10d", "mean"))
            .reset_index()
        )

        pivot = grouped.pivot_table(
            index=features,
            columns="policy_action",
            values=["samples", "mean_fwd10"],
            aggfunc="first",
            observed=False,
        )

        focus_mean_col = ("mean_fwd10", focus_action)
        against_mean_col = ("mean_fwd10", against_action)
        focus_n_col = ("samples", focus_action)
        against_n_col = ("samples", against_action)
        needed_cols = {focus_mean_col, against_mean_col, focus_n_col, against_n_col}
        if not needed_cols.issubset(set(pivot.columns)):
            continue

        sub = pivot.copy()
        sub["edge"] = sub[focus_mean_col] - sub[against_mean_col]
        sub["focus_n"] = sub[focus_n_col]
        sub["against_n"] = sub[against_n_col]

        sub = sub[
            (sub["focus_n"] >= min_samples)
            & (sub["against_n"] >= min_samples)
            & (sub["edge"] >= min_edge)
        ]
        if sub.empty:
            continue

        out = sub.reset_index()
        normalized = pd.DataFrame(index=out.index)
        for feature in features:
            normalized[feature] = out[feature]
        normalized["feature_set"] = "|".join(features)
        normalized["focus_action"] = focus_action
        normalized["against_action"] = against_action
        normalized["focus_n"] = out["focus_n"]
        normalized["against_n"] = out["against_n"]
        normalized["edge"] = out["edge"]
        normalized["focus_mean_fwd10"] = out[focus_mean_col]
        normalized["against_mean_fwd10"] = out[against_mean_col]
        normalized = normalized.sort_values("edge", ascending=False)
        frames.append(normalized)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    required = [c for c in ["feature_set", "edge", "focus_n", "against_n"] if c in result.columns]
    if required:
        result = result.dropna(subset=required)
    if "edge" in result.columns:
        result = result[result["edge"].map(pd.notna)]
    return result.sort_values("edge", ascending=False).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze day-by-day regime/action opportunities")
    parser.add_argument("--label", required=True)
    parser.add_argument("--exclude", default="^VIX,^VVIX")
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--min-edge", type=float, default=0.005, help="Minimum fwd_10d edge to recommend override")
    parser.add_argument(
        "--feature-sets",
        default="regime_combo,regime_combo+decreasing,regime_combo+rs_bin,vix_bucket+trend_state+drawdown_state+vol_state",
        help="Comma-separated feature sets using '+' between columns",
    )
    parser.add_argument("--rs-quantiles", type=int, default=5)
    parser.add_argument("--focus-action", default="hold")
    parser.add_argument("--against-action", default="de-risk")
    args = parser.parse_args()

    exclude = {x.strip() for x in args.exclude.split(",") if x.strip()}
    data = load_decay_rows(args.label, exclude)
    data["decreasing"] = data["decreasing"].astype(str)
    data["rolling_sharpe"] = pd.to_numeric(data["rolling_sharpe"], errors="coerce")
    if data["rolling_sharpe"].notna().sum() >= args.rs_quantiles:
        data["rs_bin"] = build_rs_bin(data["rolling_sharpe"], args.rs_quantiles)
    else:
        data["rs_bin"] = pd.NA

    grouped = (
        data.groupby(["regime_combo", "policy_action"], as_index=False, observed=False)
        .agg(
            samples=("fwd_10d", "count"),
            mean_fwd10=("fwd_10d", "mean"),
            median_fwd10=("fwd_10d", "median"),
            mean_fwd5=("fwd_5d", "mean"),
        )
        .sort_values(["regime_combo", "mean_fwd10"], ascending=[True, False])
    )

    regime_counts = (
        data.groupby(["regime_combo", "policy_action"], as_index=False, observed=False)
        .size()
        .rename(columns={"size": "action_freq"})
    )

    dominant = regime_counts.sort_values(["regime_combo", "action_freq"], ascending=[True, False]).drop_duplicates(
        ["regime_combo"]
    )
    dominant = dominant[["regime_combo", "policy_action"]].rename(columns={"policy_action": "dominant_action"})

    best = grouped[grouped["samples"] >= args.min_samples].sort_values(
        ["regime_combo", "mean_fwd10"], ascending=[True, False]
    ).drop_duplicates(["regime_combo"])
    best = best[["regime_combo", "policy_action", "samples", "mean_fwd10", "median_fwd10", "mean_fwd5"]].rename(
        columns={"policy_action": "best_action"}
    )

    merged = dominant.merge(best, on="regime_combo", how="inner")

    dominant_perf = grouped.rename(columns={"policy_action": "dominant_action", "mean_fwd10": "dominant_mean_fwd10"})[
        ["regime_combo", "dominant_action", "dominant_mean_fwd10"]
    ]
    merged = merged.merge(dominant_perf, on=["regime_combo", "dominant_action"], how="left")
    merged["edge_vs_dominant"] = merged["mean_fwd10"] - merged["dominant_mean_fwd10"]

    suggestions = merged[
        (merged["best_action"] != merged["dominant_action"])
        & (merged["edge_vs_dominant"] >= args.min_edge)
    ].sort_values("edge_vs_dominant", ascending=False)

    feature_sets = parse_feature_sets(args.feature_sets)
    fine_suggestions = build_fine_suggestions(
        data,
        feature_sets=feature_sets,
        min_samples=args.min_samples,
        min_edge=args.min_edge,
        focus_action=args.focus_action,
        against_action=args.against_action,
    )

    tag = "regime_improvements"
    out_grouped = REPORTS / f"{tag}_{args.label}.csv"
    out_suggest = REPORTS / f"{tag}_{args.label}_suggestions.csv"
    out_fine = REPORTS / f"{tag}_{args.label}_fine_suggestions.csv"
    grouped.to_csv(out_grouped, index=False)
    suggestions.to_csv(out_suggest, index=False)
    fine_suggestions.to_csv(out_fine, index=False)

    print("saved", out_grouped)
    print("saved", out_suggest)
    print("saved", out_fine)
    print("regimes_analyzed", int(data["regime_combo"].nunique()))
    print("suggestions", int(len(suggestions)))
    print("fine_suggestions", int(len(fine_suggestions)))
    if not suggestions.empty:
        print("top_suggestions")
        print(
            suggestions[
                [
                    "regime_combo",
                    "dominant_action",
                    "best_action",
                    "edge_vs_dominant",
                    "samples",
                    "mean_fwd10",
                    "dominant_mean_fwd10",
                ]
            ]
            .head(15)
            .to_string(index=False)
        )
    if not fine_suggestions.empty:
        print("top_fine_suggestions")
        print(
            fine_suggestions[
                [
                    "feature_set",
                    "focus_action",
                    "against_action",
                    "edge",
                    "focus_n",
                    "against_n",
                    "focus_mean_fwd10",
                    "against_mean_fwd10",
                ]
            ]
            .head(15)
            .to_string(index=False)
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
