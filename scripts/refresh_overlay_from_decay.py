from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ROOT / "reports" / "selected_table_checks"

RERISK_EXPOSURE = 1.06
HOLD_EXPOSURE = 0.0
CAP_EXPOSURE = 0.21


def parse_derisk_from_label(label: str) -> float:
    match_neg = re.search(r"_ndvol(\d+)_", label)
    if match_neg:
        return -float(match_neg.group(1)) / 100.0
    match_pos = re.search(r"_dvol(\d+)_", label)
    if match_pos:
        return float(match_pos.group(1)) / 100.0
    return -1.0


def action_to_exposure(action: str, derisk: float) -> float:
    action_l = str(action).strip().lower()
    if "re-risk" in action_l:
        return RERISK_EXPOSURE
    if "de-risk" in action_l:
        return derisk
    if action_l == "hold" or action_l.startswith("hold_"):
        return HOLD_EXPOSURE
    if action_l == "cap" or action_l.startswith("cap"):
        return CAP_EXPOSURE
    if action_l.startswith("vol_boost") or action_l.startswith("hold_boost"):
        return HOLD_EXPOSURE
    return HOLD_EXPOSURE


def segment_metrics(frame: pd.DataFrame, derisk: float) -> dict[str, float]:
    frame = frame.sort_values("date")
    returns = pd.to_numeric(frame["price"], errors="coerce").pct_change().fillna(0.0)
    exposure = frame["policy_action"].map(lambda action: action_to_exposure(action, derisk=derisk))
    pnl = (returns * exposure.shift(1).fillna(HOLD_EXPOSURE)).dropna()

    if pnl.empty:
        return {
            "sharpe": float("nan"),
            "cagr": float("nan"),
            "vol": float("nan"),
            "max_dd": float("nan"),
        }

    vol_daily = float(pnl.std(ddof=1))
    sharpe = float(np.sqrt(252.0) * pnl.mean() / vol_daily) if np.isfinite(vol_daily) and vol_daily > 0 else float("nan")
    vol = float(vol_daily * np.sqrt(252.0)) if np.isfinite(vol_daily) else float("nan")

    clipped = pnl.clip(lower=-0.999999)
    cum = (1.0 + clipped).cumprod()
    years = len(cum) / 252.0
    cagr = float(cum.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    max_dd = float((cum / cum.cummax() - 1.0).min())

    return {
        "sharpe": sharpe,
        "cagr": cagr,
        "vol": vol,
        "max_dd": max_dd,
    }


def refresh_overlay(label: str) -> tuple[int, int, int]:
    derisk = parse_derisk_from_label(label)
    paths = sorted(CHECKS.glob(f"decay_checks_*_{label}.csv"))

    rows: list[dict[str, float | str]] = []
    ticker_count = 0

    for path in paths:
        ticker = path.name.replace("decay_checks_", "").replace(f"_{label}.csv", "")
        df = pd.read_csv(path)
        if "date" not in df.columns or "price" not in df.columns:
            continue

        action_col = "policy_action" if "policy_action" in df.columns else ("action" if "action" in df.columns else None)
        if action_col is None:
            continue

        work = df[["date", "price", action_col] + (["segment"] if "segment" in df.columns else [])].copy()
        work.columns = ["date", "price", "policy_action"] + (["segment"] if "segment" in df.columns else [])
        work["date"] = pd.to_datetime(work["date"], errors="coerce", utc=True)
        work = work.dropna(subset=["date", "price", "policy_action"])
        if work.empty:
            continue

        ticker_count += 1

        available_segments: set[str] = set()
        if "segment" in work.columns:
            available_segments = set(work["segment"].astype(str))

        if available_segments:
            segments = [s for s in ["full", "in_sample", "oos"] if (s == "full" or s in available_segments)]
        else:
            segments = ["full", "oos"]

        for segment in segments:
            if segment == "full":
                sub = work
            elif "segment" in work.columns and segment in available_segments:
                sub = work[work["segment"].astype(str) == segment]
            elif segment == "oos":
                sub = work
            else:
                sub = work.iloc[0:0]
            if sub.empty:
                continue
            stats = segment_metrics(sub, derisk=derisk)
            rows.append(
                {
                    "ticker": ticker,
                    "segment": segment,
                    "sharpe": stats["sharpe"],
                    "cagr": stats["cagr"],
                    "vol": stats["vol"],
                    "max_dd": stats["max_dd"],
                }
            )

    out_path = CHECKS / f"overlay_summary_{label}.csv"
    out_df = pd.DataFrame(rows)
    if out_df.empty:
        return (0, 0, 0)

    out_df = out_df.sort_values(["ticker", "segment"]).reset_index(drop=True)
    out_df.to_csv(out_path, index=False)

    oos_count = int((out_df["segment"] == "oos").sum()) if "segment" in out_df.columns else 0
    return (ticker_count, int(len(out_df)), oos_count)


def collect_labels(versions: list[str]) -> list[str]:
    labels: set[str] = set()
    for version in versions:
        for path in CHECKS.glob(f"overlay_summary_run1_tight_weak_{version}_*.csv"):
            label = path.name.replace("overlay_summary_", "").replace(".csv", "")
            labels.add(label)
    return sorted(labels)


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh overlay_summary files from decay_checks with lagged exposure")
    parser.add_argument("--versions", nargs="+", default=["v224", "v232"], help="Versions to refresh, e.g., v224 v232")
    parser.add_argument("--report", default="reports/overlay_refresh_report.csv", help="Output refresh report CSV")
    args = parser.parse_args()

    labels = collect_labels(args.versions)
    if not labels:
        raise SystemExit("No labels found for requested versions")

    report_rows = []
    for idx, label in enumerate(labels, start=1):
        tickers, rows, oos_rows = refresh_overlay(label)
        report_rows.append(
            {
                "label": label,
                "tickers_refreshed": tickers,
                "rows_written": rows,
                "oos_rows": oos_rows,
            }
        )
        if idx % 25 == 0 or idx == len(labels):
            print(f"progress {idx}/{len(labels)}")

    rep = pd.DataFrame(report_rows)
    out_report = ROOT / args.report
    out_report.parent.mkdir(parents=True, exist_ok=True)
    rep.to_csv(out_report, index=False)

    print("saved", out_report)
    print("labels", len(rep))
    print("labels_with_rows", int((rep["rows_written"] > 0).sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
