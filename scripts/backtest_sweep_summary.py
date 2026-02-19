#!/usr/bin/env python3
import argparse
import json
from math import isfinite
from pathlib import Path
import re
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize backtest validation sweeps and optionally plot tradeoff charts.",
    )
    parser.add_argument(
        "--input-glob",
        default="reports/backtest_validation_vvix_q*.json",
        help="Glob pattern for validation JSON files.",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=15,
        help="Minimum trade count for ranking outputs.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of rows to print.",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Dedupe by (alpha, vvix) before ranking.",
    )
    parser.add_argument(
        "--out-csv",
        default="",
        help="Optional CSV output path for ranked results.",
    )
    parser.add_argument(
        "--out-md",
        default="",
        help="Optional Markdown output path for ranked results.",
    )
    parser.add_argument(
        "--plot",
        default="",
        help="Optional PNG output path for tradeoff plot.",
    )
    return parser.parse_args()


def load_points(files: list[Path], min_trades: int) -> list[dict[str, Any]]:
    points = []
    for path in files:
        m = re.search(r"vvix_q([0-9.]+)", path.stem)
        label = m.group(1) if m else path.stem
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data.get("scenarios", []):
            stats = row.get("stats", {})
            ret = stats.get("annualized_return")
            vol = stats.get("annualized_volatility")
            trades = stats.get("trade_count")
            if not isinstance(ret, (int, float)) or not isinstance(vol, (int, float)):
                continue
            if not isinstance(trades, (int, float)):
                continue
            if vol <= 0 or not isfinite(ret) or not isfinite(vol):
                continue
            score = ret / vol
            robust = score
            if trades > 0:
                scale = min(1.0, trades / float(min_trades))
                robust = score * (scale ** 0.5)
            points.append(
                {
                    "q": label,
                    "alpha": row.get("alpha_spread_threshold"),
                    "vvix": row.get("vvix_safe_threshold"),
                    "regime": row.get("regime"),
                    "trades": trades,
                    "score": score,
                    "robust": robust,
                    "ret": ret,
                    "vol": vol,
                }
            )
    return points


def rank_points(points: list[dict[str, Any]], min_trades: int, dedupe: bool) -> list[dict[str, Any]]:
    filtered = [p for p in points if p["trades"] >= min_trades]
    if not dedupe:
        return sorted(filtered, key=lambda p: p["score"], reverse=True)

    best = {}
    for p in filtered:
        key = (float(p["alpha"]), float(p["vvix"]))
        if key not in best or p["score"] > best[key]["score"]:
            best[key] = p
    return sorted(best.values(), key=lambda p: p["score"], reverse=True)


def print_top(points: list[dict[str, Any]], top_n: int) -> None:
    print(f"Top-{top_n} results:")
    for idx, p in enumerate(points[:top_n], start=1):
        print(
            f"{idx}) q={p['q']} a={p['alpha']} v={p['vvix']} "
            f"score={p['score']:.3f} robust={p['robust']:.3f} "
            f"trades={p['trades']:.0f} ret={p['ret']:.3f} vol={p['vol']:.3f}"
        )


def write_csv(points: list[dict[str, Any]], path: Path) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "q",
                "alpha",
                "vvix",
                "regime",
                "trades",
                "score",
                "robust",
                "ret",
                "vol",
            ],
        )
        writer.writeheader()
        writer.writerows(points)


def write_markdown(points: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Backtest Sweep Summary",
        "",
        "q | alpha | vvix | regime | trades | score | robust | ret | vol",
        "--- | --- | --- | --- | --- | --- | --- | --- | ---",
    ]
    for row in points:
        lines.append(
            f"{row['q']} | {row['alpha']} | {row['vvix']} | {row['regime']} | "
            f"{row['trades']:.0f} | {row['score']:.3f} | {row['robust']:.3f} | "
            f"{row['ret']:.3f} | {row['vol']:.3f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(points: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    for label in sorted({p["q"] for p in points}):
        xs = [p["trades"] for p in points if p["q"] == label]
        ys = [p["score"] for p in points if p["q"] == label]
        if xs:
            plt.scatter(xs, ys, s=12, alpha=0.6, label=f"q={label}")

    plt.title("Return/Vol vs Trade Count")
    plt.xlabel("Trade count")
    plt.ylabel("Return/Vol")
    plt.legend(title="VVIX quantile", fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)


def main() -> None:
    args = parse_args()
    files = sorted(Path(".").glob(args.input_glob))
    if not files:
        raise SystemExit(f"No files found for pattern: {args.input_glob}")

    points = load_points(files, args.min_trades)
    ranked = rank_points(points, args.min_trades, args.dedupe)
    print_top(ranked, args.top)

    if args.out_csv:
        write_csv(ranked, Path(args.out_csv))
        print(f"Saved {args.out_csv}")

    if args.out_md:
        write_markdown(ranked, Path(args.out_md))
        print(f"Saved {args.out_md}")

    if args.plot:
        write_plot(points, Path(args.plot))
        print(f"Saved {args.plot}")


if __name__ == "__main__":
    main()
