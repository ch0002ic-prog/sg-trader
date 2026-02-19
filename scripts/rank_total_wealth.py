from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_historical_wealth import (
    compute_portfolio_path,
    load_ticker_frames,
    parse_dvol_from_label,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def compute_drawdown(wealth: pd.Series) -> float:
    if wealth.empty:
        return float("nan")
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return float(dd.min())


def wealth_from_returns(portfolio_ret: pd.Series) -> pd.Series:
    if portfolio_ret.empty:
        return pd.Series(dtype=float)
    clipped = pd.to_numeric(portfolio_ret, errors="coerce").fillna(0.0).clip(lower=-0.999999)
    logw = np.log1p(clipped).cumsum()
    logw = np.clip(logw, -700.0, 700.0)
    wealth = np.exp(logw)
    return pd.Series(wealth, index=portfolio_ret.index)


def slice_recent(portfolio: pd.DataFrame, days: int) -> pd.DataFrame:
    if portfolio.empty:
        return portfolio
    if days <= 0 or len(portfolio) <= days:
        return portfolio.copy()
    return portfolio.iloc[-days:].copy()


def summarize_wealth(portfolio: pd.DataFrame) -> dict[str, float]:
    if portfolio.empty:
        return {
            "rows": 0,
            "terminal_wealth": float("nan"),
            "max_drawdown": float("nan"),
            "avg_daily_ret": float("nan"),
            "vol_daily_ret": float("nan"),
        }

    wealth = wealth_from_returns(portfolio["portfolio_ret"])
    rets = portfolio["portfolio_ret"].astype(float)
    return {
        "rows": int(len(portfolio)),
        "terminal_wealth": float(wealth.iloc[-1]),
        "max_drawdown": compute_drawdown(wealth),
        "avg_daily_ret": float(rets.mean()),
        "vol_daily_ret": float(rets.std(ddof=1)) if len(rets) > 1 else float("nan"),
    }


def build_markdown(df: pd.DataFrame, top_n: int, generated_at: str) -> str:
    lines: list[str] = []
    lines.append("# v232 Total Wealth Ranking")
    lines.append("")
    lines.append(f"Generated: {generated_at}")
    lines.append("")
    lines.append("Objective: maximize total wealth earned from $1 initial investment (terminal wealth multiple).")
    lines.append("")
    if df.empty:
        lines.append("No rows produced.")
        return "\n".join(lines) + "\n"

    show = df.head(top_n).copy()
    cols = [
        "rank_total_wealth",
        "label",
        "terminal_wealth",
        "terminal_wealth_252d",
        "max_drawdown",
        "max_drawdown_252d",
        "strict_plus_pass",
        "stable_numeric",
    ]
    cols = [c for c in cols if c in show.columns]

    viable_count = int(df["wealth_viable"].fillna(False).sum()) if "wealth_viable" in df.columns else 0
    lines.append(f"Wealth-viable labels: {viable_count} / {len(df)}")
    lines.append("")
    if viable_count == 0:
        lines.append("Warning: no label passed wealth viability guardrails (terminal_wealth > 1 and max_drawdown > -0.99).")
        lines.append("")

    lines.append(f"Top {min(top_n, len(show))} labels:")
    lines.append("")
    hdr = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines.extend([hdr, sep])
    for _, row in show[cols].iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank labels by terminal wealth from $1 initial investment")
    parser.add_argument("--labels", nargs="*", default=None, help="Optional explicit labels to rank")
    parser.add_argument(
        "--shortlist",
        type=str,
        default="reports/v232_promotion_safe_shortlist.csv",
        help="CSV containing a label column; used when --labels not provided",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional CSV containing a label column; overrides --shortlist when provided and --labels absent.",
    )
    parser.add_argument("--top", type=int, default=20, help="Rows to show in markdown")
    parser.add_argument("--rerisk", type=float, default=1.06)
    parser.add_argument("--hold", type=float, default=0.00)
    parser.add_argument("--cap", type=float, default=0.21)
    parser.add_argument("--exclude", default="^VIX,^VVIX")
    parser.add_argument("--recent-days", type=int, default=252)
    parser.add_argument(
        "--out-csv",
        default="reports/v232_total_wealth_ranking.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--out-md",
        default="reports/v232_total_wealth_ranking.md",
        help="Output markdown path",
    )
    parser.add_argument(
        "--min-terminal-wealth",
        type=float,
        default=1.0,
        help="Minimum terminal wealth to mark a label as wealth-viable.",
    )
    parser.add_argument(
        "--min-maxdd",
        type=float,
        default=-0.99,
        help="Minimum acceptable max drawdown (e.g. -0.99).",
    )
    parser.add_argument(
        "--exclude-pattern",
        default="_base",
        help="Exclude labels containing this substring (empty disables).",
    )
    args = parser.parse_args()

    if args.labels:
        labels = [x.strip() for x in args.labels if x.strip()]
    else:
        source_csv = args.input if args.input else args.shortlist
        source_path = ROOT / source_csv
        if not source_path.exists():
            raise FileNotFoundError(f"Label source not found: {source_path}")
        sdf = pd.read_csv(source_path)
        if "label" not in sdf.columns:
            raise ValueError("Label source must contain a 'label' column")
        labels = sdf["label"].dropna().astype(str).tolist()

    labels = sorted(set(labels))
    if args.exclude_pattern:
        labels = [label for label in labels if args.exclude_pattern not in label]
    if not labels:
        raise SystemExit("No labels to evaluate")

    canon_path = REPORTS / "v232_canonical_calibrated_vs_v80_v141.csv"
    canon = pd.read_csv(canon_path) if canon_path.exists() else pd.DataFrame()

    exclude = {x.strip() for x in args.exclude.split(",") if x.strip()}
    rows: list[dict] = []

    for label in labels:
        try:
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
            full = summarize_wealth(portfolio)
            recent = summarize_wealth(slice_recent(portfolio, args.recent_days))
        except Exception as exc:
            rows.append(
                {
                    "label": label,
                    "error": str(exc),
                    "terminal_wealth": float("nan"),
                    "terminal_wealth_252d": float("nan"),
                }
            )
            continue

        row = {
            "label": label,
            "terminal_wealth": full["terminal_wealth"],
            "max_drawdown": full["max_drawdown"],
            "avg_daily_ret": full["avg_daily_ret"],
            "vol_daily_ret": full["vol_daily_ret"],
            "rows_full": full["rows"],
            "terminal_wealth_252d": recent["terminal_wealth"],
            "max_drawdown_252d": recent["max_drawdown"],
            "rows_252d": recent["rows"],
            "derisk": derisk,
            "error": "",
        }

        if not canon.empty and "label" in canon.columns:
            hit = canon[canon["label"] == label]
            if not hit.empty:
                for col in [
                    "strict_plus_pass",
                    "stable_numeric",
                    "mean_sharpe_oos",
                    "mean_cagr_oos",
                    "trimmed_cagr_oos",
                    "mean_maxdd_oos",
                    "delta_sharpe_mean_vs_v141",
                    "delta_cagr_mean_vs_v141",
                    "delta_maxdd_mean_vs_v141",
                ]:
                    if col in hit.columns:
                        row[col] = hit.iloc[0][col]

        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df["is_valid"] = out_df["error"].astype(str).str.len() == 0
    out_df["wealth_viable"] = (
        out_df["is_valid"]
        & (pd.to_numeric(out_df["terminal_wealth"], errors="coerce") > float(args.min_terminal_wealth))
        & (pd.to_numeric(out_df["max_drawdown"], errors="coerce") > float(args.min_maxdd))
    )

    valid = out_df[out_df["is_valid"]].copy()
    if not valid.empty:
        valid = valid.sort_values(
            [
                "wealth_viable",
                "terminal_wealth",
                "terminal_wealth_252d",
                "avg_daily_ret",
                "max_drawdown",
            ],
            ascending=[False, False, False, False, False],
        ).reset_index(drop=True)
        valid.insert(0, "rank_total_wealth", range(1, len(valid) + 1))

    invalid = out_df[~out_df["is_valid"]].copy()
    result = pd.concat([valid, invalid], ignore_index=True)

    out_csv = ROOT / args.out_csv
    out_md = ROOT / args.out_md
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(out_csv, index=False)

    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_md.write_text(build_markdown(valid, args.top, generated), encoding="utf-8")

    print("saved", out_csv)
    print("saved", out_md)
    print("rows_total", len(result))
    print("rows_valid", len(valid))
    if not valid.empty:
        viable_n = int(valid["wealth_viable"].fillna(False).sum())
        print("rows_wealth_viable", viable_n)
    if not valid.empty:
        print("top_label", valid.iloc[0]["label"])
        print("top_terminal_wealth", float(valid.iloc[0]["terminal_wealth"]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
