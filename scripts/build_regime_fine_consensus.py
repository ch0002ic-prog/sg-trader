from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def parse_version(label: str) -> int:
    match = re.search(r"_v(\d+)_", label)
    return int(match.group(1)) if match else -1


def infer_label_from_path(path: Path) -> str:
    stem = path.stem
    prefix = "regime_improvements_"
    suffix = "_fine_suggestions"
    if stem.startswith(prefix) and stem.endswith(suffix):
        return stem[len(prefix) : -len(suffix)]
    return stem


def load_fine_files(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        required = {
            "regime_combo",
            "rs_bin",
            "feature_set",
            "focus_action",
            "against_action",
            "focus_n",
            "against_n",
            "edge",
            "focus_mean_fwd10",
            "against_mean_fwd10",
        }
        if not required.issubset(df.columns):
            continue

        label = infer_label_from_path(path)
        df = df.copy()
        df["label"] = label
        df["version"] = parse_version(label)
        frames.append(df)

    if not frames:
        raise RuntimeError("No valid fine suggestion files found")
    return pd.concat(frames, ignore_index=True)


def build_consensus(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = [
        "regime_combo",
        "rs_bin",
        "feature_set",
        "focus_action",
        "against_action",
    ]

    grouped = (
        df.groupby(key_cols, as_index=False, observed=False)
        .agg(
            version_count=("label", "nunique"),
            labels=("label", lambda s: "|".join(sorted(set(s)))),
            min_edge=("edge", "min"),
            mean_edge=("edge", "mean"),
            max_edge=("edge", "max"),
            min_focus_n=("focus_n", "min"),
            mean_focus_n=("focus_n", "mean"),
            min_against_n=("against_n", "min"),
            mean_against_n=("against_n", "mean"),
            mean_focus_fwd10=("focus_mean_fwd10", "mean"),
            mean_against_fwd10=("against_mean_fwd10", "mean"),
        )
        .sort_values(["version_count", "mean_edge", "min_edge"], ascending=[False, False, False])
        .reset_index(drop=True)
    )

    return grouped


def main() -> int:
    parser = argparse.ArgumentParser(description="Build consensus from fine regime suggestions")
    parser.add_argument(
        "--files",
        nargs="+",
        help="Explicit fine suggestion file paths. If omitted, auto-discovers reports/*_fine_suggestions.csv",
    )
    parser.add_argument("--min-version-count", type=int, default=2)
    parser.add_argument("--min-mean-edge", type=float, default=0.001)
    parser.add_argument("--min-focus-n", type=int, default=50)
    parser.add_argument("--min-against-n", type=int, default=100)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    if args.files:
        files = [Path(p).resolve() for p in args.files]
    else:
        files = sorted(REPORTS.glob("regime_improvements_*_fine_suggestions.csv"))

    fine = load_fine_files(files)
    consensus = build_consensus(fine)

    candidates = consensus[
        (consensus["version_count"] >= args.min_version_count)
        & (consensus["mean_edge"] >= args.min_mean_edge)
        & (consensus["min_focus_n"] >= args.min_focus_n)
        & (consensus["min_against_n"] >= args.min_against_n)
    ].copy()

    candidates = candidates.sort_values(
        ["version_count", "mean_edge", "min_edge", "min_focus_n"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    out_consensus = REPORTS / "regime_improvements_fine_consensus.csv"
    out_candidates = REPORTS / "regime_improvements_policy_candidates.csv"
    out_md = REPORTS / "regime_improvements_policy_candidates.md"

    consensus.to_csv(out_consensus, index=False)
    candidates.to_csv(out_candidates, index=False)

    lines = [
        "# Regime Fine Consensus Candidates",
        "",
        f"Source files: {len(files)}",
        f"Total consensus rows: {len(consensus)}",
        f"Filtered candidates: {len(candidates)}",
        "",
        "## Top candidates",
        "",
        "| regime_combo | rs_bin | focus_action | against_action | version_count | mean_edge | min_edge | min_focus_n | min_against_n |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in candidates.head(args.top).iterrows():
        lines.append(
            "| "
            f"{row['regime_combo']} | {row['rs_bin']} | {row['focus_action']} | {row['against_action']} | "
            f"{int(row['version_count'])} | {row['mean_edge']:.6f} | {row['min_edge']:.6f} | "
            f"{int(row['min_focus_n'])} | {int(row['min_against_n'])} |"
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("saved", out_consensus)
    print("saved", out_candidates)
    print("saved", out_md)
    print("source_files", len(files))
    print("consensus_rows", len(consensus))
    print("candidate_rows", len(candidates))
    if not candidates.empty:
        print("top_candidates")
        print(
            candidates[
                [
                    "regime_combo",
                    "rs_bin",
                    "feature_set",
                    "focus_action",
                    "against_action",
                    "version_count",
                    "mean_edge",
                    "min_edge",
                    "min_focus_n",
                    "min_against_n",
                ]
            ]
            .head(args.top)
            .to_string(index=False)
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
