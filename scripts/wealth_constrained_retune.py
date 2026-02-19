from __future__ import annotations

import argparse
import itertools
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_historical_wealth import compute_portfolio_path, load_ticker_frames, parse_dvol_from_label

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def parse_grid(text: str) -> list[float]:
    values = []
    for part in str(text).split(","):
        item = part.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("Grid cannot be empty")
    return values


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


def summarize_portfolio(portfolio: pd.DataFrame) -> dict[str, float]:
    if portfolio.empty:
        return {
            "rows": 0,
            "terminal_wealth": float("nan"),
            "max_drawdown": float("nan"),
            "avg_daily_ret": float("nan"),
            "vol_daily_ret": float("nan"),
        }

    wealth = wealth_from_returns(portfolio["portfolio_ret"])
    rets = pd.to_numeric(portfolio["portfolio_ret"], errors="coerce").fillna(0.0)
    return {
        "rows": int(len(portfolio)),
        "terminal_wealth": float(wealth.iloc[-1]),
        "max_drawdown": compute_drawdown(wealth),
        "avg_daily_ret": float(rets.mean()),
        "vol_daily_ret": float(rets.std(ddof=1)) if len(rets) > 1 else float("nan"),
    }


def load_labels(args: argparse.Namespace) -> list[str]:
    if args.labels:
        labels = [x.strip() for x in args.labels if x.strip()]
    else:
        source = ROOT / args.input
        if not source.exists():
            raise FileNotFoundError(f"Input CSV not found: {source}")
        df = pd.read_csv(source)
        if "label" not in df.columns:
            raise ValueError("Input CSV must contain a 'label' column")
        labels = df["label"].dropna().astype(str).tolist()

    labels = sorted(set(labels))
    if args.exclude_pattern:
        labels = [label for label in labels if args.exclude_pattern not in label]

    if args.version_filter:
        allowed = {item.strip() for item in str(args.version_filter).split(",") if item.strip()}
        labels = [label for label in labels if any(f"_{version}_" in label for version in allowed)]

    if not labels:
        raise SystemExit("No labels to evaluate")
    return labels


def build_markdown(df: pd.DataFrame, viable_only: pd.DataFrame, top_n: int, generated_at: str, args: argparse.Namespace) -> str:
    lines: list[str] = []
    lines.append("# Wealth-Constrained Retune Sweep")
    lines.append("")
    lines.append(f"Generated: {generated_at}")
    lines.append("")
    lines.append("## Objective")
    lines.append("- Search exposure combinations while enforcing wealth viability guardrails.")
    lines.append("")
    lines.append("## Grid")
    lines.append(f"- rerisk values: {args.rerisk_grid}")
    lines.append(f"- hold values: {args.hold_grid}")
    lines.append(f"- cap values: {args.cap_grid}")
    lines.append(
        f"- derisk values: {args.derisk_grid if str(args.derisk_grid).strip() else 'label-derived'}"
    )
    lines.append(f"- labels evaluated: {df['label'].nunique() if not df.empty else 0}")
    lines.append("")
    lines.append("## Guardrails")
    lines.append(f"- terminal_wealth > {args.min_terminal_wealth}")
    lines.append(f"- max_drawdown > {args.min_maxdd}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- total combinations: {len(df)}")
    lines.append(f"- viable combinations: {len(viable_only)}")
    lines.append("")

    show = viable_only.head(top_n)
    if show.empty:
        lines.append("No viable combinations found.")
        return "\n".join(lines) + "\n"

    cols = [
        "rank_viable",
        "label",
        "rerisk",
        "hold",
        "cap",
        "derisk",
        "terminal_wealth",
        "max_drawdown",
        "avg_daily_ret",
    ]
    lines.append(f"Top {min(top_n, len(show))} viable combinations:")
    lines.append("")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in show[cols].iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Wealth-constrained retune sweep on existing label artifacts")
    parser.add_argument("--labels", nargs="*", default=None, help="Optional explicit labels")
    parser.add_argument(
        "--input",
        default="reports/v232_canonical_calibrated_vs_v80_v141.csv",
        help="CSV input with a label column when --labels not provided",
    )
    parser.add_argument(
        "--version-filter",
        default="v224,v232",
        help="Optional comma-separated version tags (e.g. v224,v232)",
    )
    parser.add_argument("--exclude-pattern", default="_base", help="Exclude labels containing this substring")
    parser.add_argument("--rerisk-grid", default="0.4,0.6,0.8,1.0", help="Comma-separated rerisk exposure values")
    parser.add_argument("--hold-grid", default="0.0,0.05,0.1", help="Comma-separated hold exposure values")
    parser.add_argument("--cap-grid", default="0.1,0.2,0.3", help="Comma-separated cap exposure values")
    parser.add_argument(
        "--derisk-grid",
        default="",
        help="Optional comma-separated derisk exposure values; when omitted uses derisk parsed from each label.",
    )
    parser.add_argument("--exclude", default="^VIX,^VVIX", help="Comma-separated tickers to exclude")
    parser.add_argument("--min-terminal-wealth", type=float, default=1.0)
    parser.add_argument("--min-maxdd", type=float, default=-0.99)
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--out-csv", default="reports/wealth_constrained_retune_results.csv")
    parser.add_argument("--out-md", default="reports/wealth_constrained_retune_results.md")
    args = parser.parse_args()

    labels = load_labels(args)
    rerisk_values = parse_grid(args.rerisk_grid)
    hold_values = parse_grid(args.hold_grid)
    cap_values = parse_grid(args.cap_grid)
    derisk_values = parse_grid(args.derisk_grid) if str(args.derisk_grid).strip() else None
    exclude_tickers = {x.strip() for x in args.exclude.split(",") if x.strip()}

    frame_cache: dict[str, dict[str, pd.DataFrame]] = {}
    rows: list[dict] = []

    for idx, label in enumerate(labels, start=1):
        if idx % 25 == 0 or idx == len(labels):
            print(f"progress labels {idx}/{len(labels)}")

        try:
            label_derisk = parse_dvol_from_label(label)
            if label not in frame_cache:
                frame_cache[label] = load_ticker_frames(label)
            frames = frame_cache[label]
        except Exception as exc:
            rows.append(
                {
                    "label": label,
                    "rerisk": float("nan"),
                    "hold": float("nan"),
                    "cap": float("nan"),
                    "derisk": float("nan"),
                    "terminal_wealth": float("nan"),
                    "max_drawdown": float("nan"),
                    "avg_daily_ret": float("nan"),
                    "vol_daily_ret": float("nan"),
                    "wealth_viable": False,
                    "error": str(exc),
                }
            )
            continue

        derisk_grid = derisk_values if derisk_values is not None else [label_derisk]
        for rerisk, hold, cap, derisk in itertools.product(rerisk_values, hold_values, cap_values, derisk_grid):
            try:
                _, portfolio = compute_portfolio_path(
                    frames,
                    rerisk=float(rerisk),
                    hold=float(hold),
                    cap=float(cap),
                    derisk=float(derisk),
                    exclude_tickers=exclude_tickers,
                )
                summary = summarize_portfolio(portfolio)
                terminal_wealth = float(summary["terminal_wealth"])
                max_drawdown = float(summary["max_drawdown"])
                wealth_viable = bool(
                    np.isfinite(terminal_wealth)
                    and np.isfinite(max_drawdown)
                    and (terminal_wealth > float(args.min_terminal_wealth))
                    and (max_drawdown > float(args.min_maxdd))
                )
                rows.append(
                    {
                        "label": label,
                        "rerisk": float(rerisk),
                        "hold": float(hold),
                        "cap": float(cap),
                        "derisk": float(derisk),
                        "terminal_wealth": terminal_wealth,
                        "max_drawdown": max_drawdown,
                        "avg_daily_ret": float(summary["avg_daily_ret"]),
                        "vol_daily_ret": float(summary["vol_daily_ret"]),
                        "wealth_viable": wealth_viable,
                        "error": "",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "label": label,
                        "rerisk": float(rerisk),
                        "hold": float(hold),
                        "cap": float(cap),
                        "derisk": float(derisk),
                        "terminal_wealth": float("nan"),
                        "max_drawdown": float("nan"),
                        "avg_daily_ret": float("nan"),
                        "vol_daily_ret": float("nan"),
                        "wealth_viable": False,
                        "error": str(exc),
                    }
                )

    result = pd.DataFrame(rows)
    result["is_valid"] = result["error"].astype(str).str.len() == 0

    valid = result[result["is_valid"]].copy()
    if not valid.empty:
        valid = valid.sort_values(
            ["wealth_viable", "terminal_wealth", "max_drawdown", "avg_daily_ret"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)

    viable_only = valid[valid["wealth_viable"]].copy()
    if not viable_only.empty:
        viable_only = viable_only.sort_values(
            ["terminal_wealth", "max_drawdown", "avg_daily_ret"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        viable_only.insert(0, "rank_viable", range(1, len(viable_only) + 1))

    out_csv = ROOT / args.out_csv
    out_md = ROOT / args.out_md
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    valid.to_csv(out_csv, index=False)

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_md.write_text(build_markdown(valid, viable_only, args.top, generated_at, args), encoding="utf-8")

    print("saved", out_csv)
    print("saved", out_md)
    print("rows_total", len(result))
    print("rows_valid", len(valid))
    print("rows_wealth_viable", len(viable_only))
    if not valid.empty:
        print("top_label", valid.iloc[0]["label"])
        print("top_terminal_wealth", float(valid.iloc[0]["terminal_wealth"]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
