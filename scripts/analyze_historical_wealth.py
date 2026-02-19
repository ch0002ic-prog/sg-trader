from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ROOT / "reports" / "selected_table_checks"
REPORTS = ROOT / "reports"


def parse_dvol_from_label(label: str) -> float:
    match_neg = re.search(r"_ndvol(\d+)_", label)
    if match_neg:
        return -float(match_neg.group(1)) / 100.0
    match_pos = re.search(r"_dvol(\d+)_", label)
    if match_pos:
        return float(match_pos.group(1)) / 100.0
    raise ValueError(f"Could not parse dvol from label: {label}")


def parse_version_from_label(label: str) -> str:
    match = re.search(r"run1_tight_weak_(v\d+)_", label)
    return match.group(1) if match else "vx"


def load_ticker_frames(label: str) -> dict[str, pd.DataFrame]:
    pattern = f"decay_checks_*_{label}.csv"
    files = sorted(CHECKS.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No decay_check files found for label: {label}")

    frames: dict[str, pd.DataFrame] = {}
    for path in files:
        ticker = path.name.split("decay_checks_")[1].split(f"_{label}.csv")[0]
        df = pd.read_csv(path)
        if "date" not in df.columns or "price" not in df.columns:
            continue
        action_col = "policy_action" if "policy_action" in df.columns else ("action" if "action" in df.columns else None)
        if action_col is None:
            continue
        work = df[["date", "price", action_col]].copy()
        work.columns = ["date", "price", "policy_action"]
        work["date"] = pd.to_datetime(work["date"], errors="coerce", utc=True)
        work = work.dropna(subset=["date", "price", "policy_action"]).sort_values("date")
        if work.empty:
            continue
        frames[ticker] = work

    if not frames:
        raise RuntimeError(f"No usable ticker frames for label: {label}")
    return frames


def action_to_exposure(action: str, rerisk: float, hold: float, cap: float, derisk: float) -> float:
    action_l = str(action).strip().lower()
    if "re-risk" in action_l:
        return rerisk
    if "de-risk" in action_l:
        return derisk
    if action_l == "hold" or action_l.startswith("hold_"):
        return hold
    if action_l == "cap" or action_l.startswith("cap"):
        return cap
    if action_l.startswith("vol_boost") or action_l.startswith("hold_boost"):
        return hold
    return hold


def compute_portfolio_path(
    frames: dict[str, pd.DataFrame],
    rerisk: float,
    hold: float,
    cap: float,
    derisk: float,
    exclude_tickers: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ticker_daily = []

    for ticker, df in frames.items():
        if ticker in exclude_tickers:
            continue

        work = df.copy()
        work["ret"] = work["price"].pct_change().fillna(0.0)
        work["exposure"] = work["policy_action"].map(
            lambda x: action_to_exposure(x, rerisk=rerisk, hold=hold, cap=cap, derisk=derisk)
        )
        work["exposure_lag"] = work["exposure"].shift(1).fillna(hold)
        work["strategy_ret"] = work["ret"] * work["exposure_lag"]
        work["ticker"] = ticker
        ticker_daily.append(work[["date", "ticker", "ret", "exposure_lag", "strategy_ret"]])

    if not ticker_daily:
        raise RuntimeError("All tickers excluded or unusable; no portfolio path available")

    daily = pd.concat(ticker_daily, ignore_index=True)

    portfolio = (
        daily.groupby("date", as_index=False)
        .agg(
            portfolio_ret=("strategy_ret", "mean"),
            active_tickers=("ticker", "nunique"),
        )
        .sort_values("date")
    )
    clipped_ret = pd.to_numeric(portfolio["portfolio_ret"], errors="coerce").fillna(0.0).clip(lower=-0.999999)
    log_wealth = np.log1p(clipped_ret).cumsum()
    portfolio["wealth"] = np.exp(np.clip(log_wealth, -700.0, 700.0))

    return daily, portfolio


def compute_forward_outcomes(portfolio: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    wealth = portfolio["wealth"].to_numpy(dtype=float)
    dates = portfolio["date"].to_list()

    rows = []
    max_i = len(portfolio) - horizon_days - 1
    for i in range(max_i + 1):
        j = i + horizon_days
        start_wealth = wealth[i]
        end_wealth = wealth[j]
        if start_wealth <= 0 or not np.isfinite(start_wealth) or not np.isfinite(end_wealth):
            continue
        multiple = float(end_wealth / start_wealth)
        rows.append(
            {
                "start_date": dates[i],
                "end_date": dates[j],
                "horizon_days": horizon_days,
                "wealth_multiple": multiple,
            }
        )

    return pd.DataFrame(rows)


def summarize_outcomes(outcomes: pd.DataFrame) -> dict[str, float]:
    if outcomes.empty:
        return {
            "count": 0,
            "mean_multiple": float("nan"),
            "median_multiple": float("nan"),
            "p10_multiple": float("nan"),
            "p90_multiple": float("nan"),
            "hit_rate_gt_1": float("nan"),
        }

    vals = outcomes["wealth_multiple"].astype(float)
    return {
        "count": int(len(vals)),
        "mean_multiple": float(vals.mean()),
        "median_multiple": float(vals.median()),
        "p10_multiple": float(vals.quantile(0.10)),
        "p90_multiple": float(vals.quantile(0.90)),
        "hit_rate_gt_1": float((vals > 1.0).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Historical day-by-day $1 wealth analysis")
    parser.add_argument("--label", required=True, help="Run label to analyze")
    parser.add_argument("--horizon-days", type=int, default=63, help="Forward holding horizon in trading days")
    parser.add_argument("--rerisk", type=float, default=1.06)
    parser.add_argument("--hold", type=float, default=0.00)
    parser.add_argument("--cap", type=float, default=0.21)
    parser.add_argument("--derisk", type=float, default=None, help="Optional override for derisk; defaults parsed from label")
    parser.add_argument("--exclude", default="^VIX,^VVIX", help="Comma-separated tickers to exclude from portfolio aggregate")
    args = parser.parse_args()

    label = args.label.strip()
    version = parse_version_from_label(label)
    derisk = parse_dvol_from_label(label) if args.derisk is None else float(args.derisk)
    exclude = {x.strip() for x in args.exclude.split(",") if x.strip()}

    frames = load_ticker_frames(label)
    ticker_daily, portfolio = compute_portfolio_path(
        frames,
        rerisk=args.rerisk,
        hold=args.hold,
        cap=args.cap,
        derisk=derisk,
        exclude_tickers=exclude,
    )

    outcomes = compute_forward_outcomes(portfolio, horizon_days=args.horizon_days)
    summary = summarize_outcomes(outcomes)

    tag = re.search(r"(n?dvol\d+)", label)
    dvol_tag = tag.group(1) if tag else "dvolX"

    out_daily = REPORTS / f"{version}_historical_portfolio_daily_{dvol_tag}.csv"
    out_ticker = REPORTS / f"{version}_historical_ticker_daily_{dvol_tag}.csv"
    out_outcomes = REPORTS / f"{version}_historical_outcomes_{dvol_tag}_T{args.horizon_days}.csv"
    out_summary = REPORTS / f"{version}_historical_outcomes_{dvol_tag}_T{args.horizon_days}_summary.json"

    portfolio.to_csv(out_daily, index=False)
    ticker_daily.to_csv(out_ticker, index=False)
    outcomes.to_csv(out_outcomes, index=False)

    summary_payload = {
        "label": label,
        "version": version,
        "derisk": derisk,
        "horizon_days": args.horizon_days,
        "exclude_tickers": sorted(exclude),
        "summary": summary,
    }
    out_summary.write_text(pd.Series(summary_payload).to_json(indent=2))

    print("saved", out_daily)
    print("saved", out_ticker)
    print("saved", out_outcomes)
    print("saved", out_summary)
    print("summary", summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
