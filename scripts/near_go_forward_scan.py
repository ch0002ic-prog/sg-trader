from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from analyze_historical_wealth import (
    compute_forward_outcomes,
    compute_portfolio_path,
    load_ticker_frames,
    summarize_outcomes,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

BASE_MAP = {
    "v224": "run1_tight_weak_v223_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol040_vf000",
    "v232": "run1_tight_weak_v231_pr106_ph000_pc021_dd072_cr033_cd062_de05_rvx080_rvv077_hvx058_cvx038_hvv070_cvv043_pcv020_rvto105_nhvto001_cvto015_dvto000_ndvx007_dvv000_ndvol736_vf000",
}


def parse_horizons(text: str) -> list[int]:
    values = []
    for part in str(text).split(","):
        item = part.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("No horizons supplied")
    return values


def portfolio_terminal_and_dd(portfolio: pd.DataFrame) -> tuple[float, float]:
    ret = pd.to_numeric(portfolio["portfolio_ret"], errors="coerce").fillna(0.0).clip(lower=-0.999999)
    wealth = (1.0 + ret).cumprod()
    terminal = float(wealth.iloc[-1]) if not wealth.empty else float("nan")
    max_dd = float((wealth / wealth.cummax() - 1.0).min()) if not wealth.empty else float("nan")
    return terminal, max_dd


def build_markdown(result: pd.DataFrame, args: argparse.Namespace, generated_at: str) -> str:
    lines: list[str] = []
    lines.append("# Near-GO Forward Scan")
    lines.append("")
    lines.append(f"Generated: {generated_at}")
    lines.append("")
    lines.append("## Scan Config")
    lines.append(f"- input: {args.input}")
    lines.append(f"- top viable rows scanned: {args.top_n}")
    lines.append(f"- horizons: {args.horizons}")
    lines.append(f"- p10 tolerance: {args.p10_tol}")
    lines.append(f"- median tolerance: {args.median_tol}")
    lines.append(f"- min terminal wealth: {args.min_terminal_wealth}")
    lines.append(f"- min max drawdown: {args.min_maxdd}")
    lines.append("")

    if result.empty:
        lines.append("No rows evaluated.")
        return "\n".join(lines) + "\n"

    strict_n = int(result["strict_go"].fillna(False).sum())
    near_n = int(result["near_go"].fillna(False).sum())
    lines.append("## Summary")
    lines.append(f"- rows evaluated: {len(result)}")
    lines.append(f"- strict GO rows: {strict_n}")
    lines.append(f"- near-GO rows: {near_n}")
    lines.append("")

    show = result.sort_values(["near_go", "terminal_wealth_candidate", "maxdd_candidate"], ascending=[False, False, False]).head(args.top_out)
    cols = [
        "rank_scan",
        "label",
        "version",
        "rerisk",
        "hold",
        "cap",
        "derisk",
        "terminal_wealth_candidate",
        "maxdd_candidate",
        "strict_go",
        "near_go",
        "worst_p10_delta",
        "worst_median_delta",
    ]
    cols = [c for c in cols if c in show.columns]

    lines.append(f"Top {min(args.top_out, len(show))} rows:")
    lines.append("")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in show[cols].iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Deeper forward scan for near-GO candidates")
    parser.add_argument("--input", default="reports/wealth_constrained_retune_results.csv")
    parser.add_argument("--top-n", type=int, default=400, help="Number of viable rows (by terminal wealth) to evaluate")
    parser.add_argument("--top-out", type=int, default=40, help="Rows to show in markdown")
    parser.add_argument("--horizons", default="21,63")
    parser.add_argument("--p10-tol", type=float, default=-0.001)
    parser.add_argument("--median-tol", type=float, default=0.0)
    parser.add_argument("--min-terminal-wealth", type=float, default=1.0)
    parser.add_argument("--min-maxdd", type=float, default=-0.99)
    parser.add_argument("--exclude", default="^VIX,^VVIX")
    parser.add_argument("--out-csv", default="reports/near_go_forward_scan.csv")
    parser.add_argument("--out-md", default="reports/near_go_forward_scan.md")
    args = parser.parse_args()

    horizons = parse_horizons(args.horizons)
    exclude = {x.strip() for x in args.exclude.split(",") if x.strip()}

    source = ROOT / args.input
    if not source.exists():
        raise FileNotFoundError(f"Input not found: {source}")

    df = pd.read_csv(source)
    viable = df[df["wealth_viable"].fillna(False)].copy()
    viable = viable.sort_values(["terminal_wealth", "max_drawdown", "avg_daily_ret"], ascending=[False, False, False]).head(args.top_n)
    viable = viable.reset_index(drop=True)

    if viable.empty:
        out = pd.DataFrame()
        out.to_csv(ROOT / args.out_csv, index=False)
        (ROOT / args.out_md).write_text("# Near-GO Forward Scan\n\nNo viable rows in input.\n", encoding="utf-8")
        print("rows_input", len(df))
        print("rows_viable_input", 0)
        print("rows_scanned", 0)
        return 0

    frames_cache: dict[str, dict[str, pd.DataFrame]] = {}
    base_portfolio_cache: dict[tuple[str, float, float, float, float], pd.DataFrame] = {}
    rows: list[dict] = []

    for idx, row in viable.iterrows():
        if (idx + 1) % 25 == 0 or (idx + 1) == len(viable):
            print(f"progress scan {idx + 1}/{len(viable)}")

        label = str(row["label"])
        version = "v232" if "_v232_" in label else ("v224" if "_v224_" in label else "")
        if version not in BASE_MAP:
            continue
        base_label = BASE_MAP[version]

        rerisk = float(row["rerisk"])
        hold = float(row["hold"])
        cap = float(row["cap"])
        derisk = float(row["derisk"])

        if label not in frames_cache:
            frames_cache[label] = load_ticker_frames(label)
        candidate_frames = frames_cache[label]

        _, port_candidate = compute_portfolio_path(
            candidate_frames,
            rerisk=rerisk,
            hold=hold,
            cap=cap,
            derisk=derisk,
            exclude_tickers=exclude,
        )

        base_key = (version, rerisk, hold, cap, derisk)
        if base_key not in base_portfolio_cache:
            if base_label not in frames_cache:
                frames_cache[base_label] = load_ticker_frames(base_label)
            _, base_portfolio_cache[base_key] = compute_portfolio_path(
                frames_cache[base_label],
                rerisk=rerisk,
                hold=hold,
                cap=cap,
                derisk=derisk,
                exclude_tickers=exclude,
            )
        port_base = base_portfolio_cache[base_key]

        tw_c, dd_c = portfolio_terminal_and_dd(port_candidate)
        tw_b, dd_b = portfolio_terminal_and_dd(port_base)

        out_row = {
            "label": label,
            "version": version,
            "base_label": base_label,
            "rerisk": rerisk,
            "hold": hold,
            "cap": cap,
            "derisk": derisk,
            "terminal_wealth_candidate": tw_c,
            "terminal_wealth_base": tw_b,
            "maxdd_candidate": dd_c,
            "maxdd_base": dd_b,
        }

        median_deltas = []
        p10_deltas = []
        for horizon in horizons:
            cand_out = compute_forward_outcomes(port_candidate, horizon_days=horizon)
            base_out = compute_forward_outcomes(port_base, horizon_days=horizon)
            cand_sum = summarize_outcomes(cand_out)
            base_sum = summarize_outcomes(base_out)

            d_med = float(cand_sum["median_multiple"] - base_sum["median_multiple"])
            d_p10 = float(cand_sum["p10_multiple"] - base_sum["p10_multiple"])
            d_hit = float(cand_sum["hit_rate_gt_1"] - base_sum["hit_rate_gt_1"])
            median_deltas.append(d_med)
            p10_deltas.append(d_p10)

            out_row[f"T{horizon}_delta_median_multiple"] = d_med
            out_row[f"T{horizon}_delta_p10_multiple"] = d_p10
            out_row[f"T{horizon}_delta_hit_rate_gt_1"] = d_hit

        worst_med = float(min(median_deltas)) if median_deltas else float("nan")
        worst_p10 = float(min(p10_deltas)) if p10_deltas else float("nan")

        strict_go = (
            (tw_c > args.min_terminal_wealth)
            and (dd_c > args.min_maxdd)
            and all(delta >= 0.0 for delta in median_deltas)
            and all(delta >= 0.0 for delta in p10_deltas)
        )
        near_go = (
            (tw_c > args.min_terminal_wealth)
            and (dd_c > args.min_maxdd)
            and all(delta >= args.median_tol for delta in median_deltas)
            and all(delta >= args.p10_tol for delta in p10_deltas)
        )

        out_row["worst_median_delta"] = worst_med
        out_row["worst_p10_delta"] = worst_p10
        out_row["strict_go"] = bool(strict_go)
        out_row["near_go"] = bool(near_go)
        rows.append(out_row)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["near_go", "terminal_wealth_candidate", "maxdd_candidate"], ascending=[False, False, False]).reset_index(drop=True)
        result.insert(0, "rank_scan", range(1, len(result) + 1))

    out_csv = ROOT / args.out_csv
    out_md = ROOT / args.out_md
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(out_csv, index=False)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_md.write_text(build_markdown(result, args, generated_at), encoding="utf-8")

    print("saved", out_csv)
    print("saved", out_md)
    print("rows_input", len(df))
    print("rows_viable_input", len(viable))
    print("rows_scanned", len(result))
    if not result.empty:
        print("strict_go", int(result["strict_go"].fillna(False).sum()))
        print("near_go", int(result["near_go"].fillna(False).sum()))
        print("top_label", result.iloc[0]["label"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
