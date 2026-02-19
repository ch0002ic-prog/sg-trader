from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ROOT / "reports" / "selected_table_checks"
REPORTS = ROOT / "reports"


@dataclass
class Perf:
    rows: int
    terminal_wealth: float
    cagr: float
    sharpe: float
    max_drawdown: float
    avg_daily_ret: float
    vol_daily_ret: float


def parse_grid(text: str) -> list[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty grid")
    return vals


def parse_derisk_from_label(label: str) -> float:
    import re

    neg = re.search(r"_ndvol(\d+)_", label)
    if neg:
        return -float(neg.group(1)) / 100.0
    pos = re.search(r"_dvol(\d+)_", label)
    if pos:
        return float(pos.group(1)) / 100.0
    raise ValueError(f"Could not parse dvol from label: {label}")


def action_to_default(action: str, *, rerisk: float, hold: float, cap: float, derisk: float) -> float:
    a = str(action).strip().lower()
    if "re-risk" in a:
        return rerisk
    if "de-risk" in a:
        return derisk
    if a == "hold" or a.startswith("hold"):
        return hold
    if a == "cap leverage" or a.startswith("cap"):
        return cap
    if a.startswith("vol_boost"):
        return hold
    return hold


def rs_bucket(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    out = pd.Series("rs_unknown", index=s.index, dtype="object")
    if len(valid) < 10:
        return out
    try:
        q = pd.qcut(valid, q=3, labels=["rs_l", "rs_m", "rs_h"], duplicates="drop")
    except ValueError:
        return out
    out.loc[valid.index] = q.astype(str)
    return out


def dd_bucket(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series("dd_unknown", index=s.index, dtype="object")
    out.loc[s > -0.03] = "dd_ok"
    out.loc[(s <= -0.03) & (s > -0.10)] = "dd_mid"
    out.loc[s <= -0.10] = "dd_deep"
    return out


def apply_persistence(states: pd.Series, persistence_days: int) -> pd.Series:
    if persistence_days <= 1:
        return states
    run_id = (states != states.shift(1)).cumsum()
    run_pos = states.groupby(run_id).cumcount() + 1
    stable = run_pos >= persistence_days
    out = states.copy()
    out.loc[~stable] = pd.NA
    return out


def load_regime_frames(label: str) -> dict[str, pd.DataFrame]:
    paths = sorted(CHECKS.glob(f"decay_checks_*_{label}.csv"))
    if not paths:
        raise FileNotFoundError(f"No decay checks for {label}")

    out: dict[str, pd.DataFrame] = {}
    for p in paths:
        ticker = p.name.split("decay_checks_")[1].split(f"_{label}.csv")[0]
        df = pd.read_csv(p)
        if "date" not in df.columns or "price" not in df.columns:
            continue
        action_col = "policy_action" if "policy_action" in df.columns else ("action" if "action" in df.columns else None)
        if action_col is None:
            continue

        work = df.copy()
        work["date"] = pd.to_datetime(work["date"], errors="coerce", utc=True)
        work["date"] = work["date"].dt.tz_convert(None)
        work = work.dropna(subset=["date", "price", action_col]).sort_values("date")
        if work.empty:
            continue

        if "regime_combo" not in work.columns:
            work["regime_combo"] = "unknown"
        work["policy_action"] = work[action_col].astype(str)

        if "rolling_sharpe" not in work.columns:
            work["rolling_sharpe"] = np.nan
        if "drawdown" not in work.columns:
            work["drawdown"] = np.nan

        out[ticker] = work[["date", "price", "policy_action", "regime_combo", "rolling_sharpe", "drawdown"]].copy()

    if not out:
        raise RuntimeError(f"No usable decay frames for {label}")
    return out


def state_key(regime_combo: str, policy_action: str) -> str:
    return f"{regime_combo}::{policy_action}"


def prepare_ticker_data(
    frames: dict[str, pd.DataFrame],
    *,
    rerisk: float,
    hold: float,
    cap: float,
    derisk: float,
    exclude_tickers: set[str],
    state_schema: str = "standard",
    persistence_days: int = 1,
    override_keys: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    schema = str(state_schema).strip().lower()
    if schema == "auto":
        if override_keys:
            schema = "augmented" if any(str(k).count("::") >= 3 for k in override_keys) else "standard"
        else:
            schema = "standard"

    out: dict[str, pd.DataFrame] = {}
    for ticker, df in frames.items():
        if ticker in exclude_tickers:
            continue
        w = df.copy()
        w["ret"] = pd.to_numeric(w["price"], errors="coerce").pct_change().fillna(0.0)
        if schema == "augmented":
            rb = rs_bucket(w.get("rolling_sharpe", pd.Series(np.nan, index=w.index)))
            db = dd_bucket(w.get("drawdown", pd.Series(np.nan, index=w.index)))
            raw_state = pd.Series(
                [
                    f"{r}::{a}::{rsi}::{ddi}"
                    for r, a, rsi, ddi in zip(w["regime_combo"], w["policy_action"], rb, db)
                ],
                index=w.index,
                dtype="object",
            )
            w["state"] = apply_persistence(raw_state, int(persistence_days))
        else:
            w["state"] = [state_key(r, a) for r, a in zip(w["regime_combo"], w["policy_action"]) ]
        w["default_exp"] = [
            action_to_default(a, rerisk=rerisk, hold=hold, cap=cap, derisk=derisk)
            for a in w["policy_action"]
        ]
        out[ticker] = w[["date", "ret", "state", "default_exp"]].copy()
    if not out:
        raise RuntimeError("No ticker data after exclusions")
    return out


def split_dates(all_dates: pd.Index, train_frac: float, val_frac: float) -> tuple[pd.Timestamp, pd.Timestamp]:
    uniq = pd.Index(sorted(pd.to_datetime(pd.Series(all_dates).dropna().unique())))
    n = len(uniq)
    if n < 20:
        raise ValueError("Not enough dates for split")
    train_end = uniq[int(max(1, min(n - 3, round(n * train_frac))))]
    val_end = uniq[int(max(2, min(n - 2, round(n * (train_frac + val_frac)))))]
    return pd.Timestamp(train_end), pd.Timestamp(val_end)


def simulate_returns(
    ticker_data: dict[str, pd.DataFrame],
    overrides: dict[str, float],
    hold_fill: float,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.Series:
    parts = []
    for ticker, df in ticker_data.items():
        w = df.copy()
        if start is not None:
            w = w[w["date"] >= start]
        if end is not None:
            w = w[w["date"] < end]
        if w.empty:
            continue

        exp = w["default_exp"].copy()
        if overrides:
            mask = w["state"].isin(overrides.keys())
            if mask.any():
                exp.loc[mask] = w.loc[mask, "state"].map(overrides)
        exp_lag = exp.shift(1).fillna(hold_fill)
        strat = pd.to_numeric(w["ret"], errors="coerce").fillna(0.0) * exp_lag
        parts.append(pd.DataFrame({"date": w["date"], "strategy_ret": strat}))

    if not parts:
        return pd.Series(dtype=float)

    merged = pd.concat(parts, ignore_index=True)
    port = merged.groupby("date", as_index=True)["strategy_ret"].mean().sort_index()
    return port


def perf_from_returns(ret: pd.Series) -> Perf:
    r = pd.to_numeric(ret, errors="coerce").fillna(0.0)
    if r.empty:
        return Perf(0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    clipped = r.clip(lower=-0.999999)
    wealth = (1.0 + clipped).cumprod()
    rows = int(len(r))
    years = rows / 252.0
    terminal = float(wealth.iloc[-1])
    cagr = float(terminal ** (1.0 / years) - 1.0) if years > 0 else np.nan
    vol = float(r.std(ddof=1)) if rows > 1 else np.nan
    sharpe = float(np.sqrt(252.0) * r.mean() / vol) if np.isfinite(vol) and vol > 0 else np.nan
    max_dd = float((wealth / wealth.cummax() - 1.0).min())
    return Perf(rows, terminal, cagr, sharpe, max_dd, float(r.mean()), vol)


def objective(perf: Perf, min_maxdd: float, dd_penalty: float) -> float:
    if not np.isfinite(perf.terminal_wealth) or perf.rows == 0:
        return -1e12
    base = float(np.log(max(perf.terminal_wealth, 1e-300)))
    penalty = dd_penalty * max(0.0, float(min_maxdd - perf.max_drawdown))
    return base - penalty


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def optimize_overrides(
    ticker_data: dict[str, pd.DataFrame],
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    candidate_grid: list[float],
    hold_fill: float,
    min_maxdd: float,
    dd_penalty: float,
    max_states: int,
    max_iters: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    train_frames = []
    for df in ticker_data.values():
        train_frames.append(df[(df["date"] >= train_start) & (df["date"] < train_end)][["state"]])
    train_states = pd.concat(train_frames, ignore_index=True)
    counts = train_states["state"].value_counts()
    states = counts.head(max_states).index.tolist()

    overrides: dict[str, float] = {}
    history = []

    current_ret = simulate_returns(ticker_data, overrides, hold_fill, start=train_start, end=train_end)
    current_perf = perf_from_returns(current_ret)
    current_score = objective(current_perf, min_maxdd=min_maxdd, dd_penalty=dd_penalty)

    for it in range(1, max_iters + 1):
        improved = False
        for state in states:
            best_val = overrides.get(state, None)
            best_score = current_score
            for val in candidate_grid:
                trial = dict(overrides)
                trial[state] = float(val)
                trial_ret = simulate_returns(ticker_data, trial, hold_fill, start=train_start, end=train_end)
                trial_perf = perf_from_returns(trial_ret)
                score = objective(trial_perf, min_maxdd=min_maxdd, dd_penalty=dd_penalty)
                if score > best_score + 1e-12:
                    best_score = score
                    best_val = float(val)
            if best_val is not None and overrides.get(state) != best_val:
                overrides[state] = best_val
                current_score = best_score
                improved = True

        history.append({"iter": it, "score": current_score, "num_overrides": len(overrides), "improved": improved})
        if not improved:
            break

    return overrides, pd.DataFrame(history)


def main() -> int:
    parser = argparse.ArgumentParser(description="Regime-aware portfolio optimizer (wealth-first)")
    parser.add_argument("--label", required=True)
    parser.add_argument("--rerisk", type=float, default=0.2)
    parser.add_argument("--hold", type=float, default=0.05)
    parser.add_argument("--cap", type=float, default=0.05)
    parser.add_argument("--derisk", type=float, default=None, help="Defaults to label-derived dvol")
    parser.add_argument("--exclude", default="^VIX,^VVIX")
    parser.add_argument("--candidate-grid", default="-0.2,-0.1,-0.05,0.0,0.05,0.1,0.2")
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--min-maxdd", type=float, default=-0.99)
    parser.add_argument("--dd-penalty", type=float, default=20.0)
    parser.add_argument("--max-states", type=int, default=16)
    parser.add_argument("--max-iters", type=int, default=4)
    parser.add_argument("--out-prefix", default="reports/regime_optimizer")
    args = parser.parse_args()

    derisk = float(args.derisk) if args.derisk is not None else parse_derisk_from_label(args.label)
    grid = parse_grid(args.candidate_grid)
    exclude = {x.strip() for x in args.exclude.split(",") if x.strip()}

    frames = load_regime_frames(args.label)
    ticker_data = prepare_ticker_data(
        frames,
        rerisk=args.rerisk,
        hold=args.hold,
        cap=args.cap,
        derisk=derisk,
        exclude_tickers=exclude,
    )

    all_date_series = [pd.to_datetime(df["date"], errors="coerce") for df in ticker_data.values()]
    all_dates = pd.Index(pd.concat(all_date_series, ignore_index=True).dropna())
    train_end, val_end = split_dates(all_dates, args.train_frac, args.val_frac)
    start = pd.to_datetime(pd.Series(all_dates).dropna().min())

    baseline = {}
    optimized, hist = optimize_overrides(
        ticker_data,
        train_start=start,
        train_end=train_end,
        candidate_grid=grid,
        hold_fill=args.hold,
        min_maxdd=args.min_maxdd,
        dd_penalty=args.dd_penalty,
        max_states=args.max_states,
        max_iters=args.max_iters,
    )

    segments = {
        "train": (start, train_end),
        "val": (train_end, val_end),
        "test": (val_end, None),
        "full": (start, None),
    }

    rows = []
    for seg, (seg_start, seg_end) in segments.items():
        ret_base = simulate_returns(ticker_data, baseline, args.hold, start=seg_start, end=seg_end)
        ret_opt = simulate_returns(ticker_data, optimized, args.hold, start=seg_start, end=seg_end)
        p_base = perf_from_returns(ret_base)
        p_opt = perf_from_returns(ret_opt)
        rows.append(
            {
                "segment": seg,
                "base_terminal_wealth": p_base.terminal_wealth,
                "opt_terminal_wealth": p_opt.terminal_wealth,
                "delta_terminal_wealth": p_opt.terminal_wealth - p_base.terminal_wealth,
                "base_cagr": p_base.cagr,
                "opt_cagr": p_opt.cagr,
                "base_sharpe": p_base.sharpe,
                "opt_sharpe": p_opt.sharpe,
                "base_maxdd": p_base.max_drawdown,
                "opt_maxdd": p_opt.max_drawdown,
                "base_rows": p_base.rows,
                "opt_rows": p_opt.rows,
            }
        )

    out_perf = pd.DataFrame(rows)
    out_overrides = pd.DataFrame(
        [{"state": s, "override_exposure": v} for s, v in sorted(optimized.items(), key=lambda x: x[0])]
    )

    prefix = ROOT / args.out_prefix
    perf_path = Path(str(prefix) + "_perf.csv")
    ovr_path = Path(str(prefix) + "_overrides.csv")
    hist_path = Path(str(prefix) + "_history.csv")
    md_path = Path(str(prefix) + "_report.md")

    perf_path.parent.mkdir(parents=True, exist_ok=True)
    out_perf.to_csv(perf_path, index=False)
    out_overrides.to_csv(ovr_path, index=False)
    hist.to_csv(hist_path, index=False)

    gen = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Regime Portfolio Optimizer Report",
        "",
        f"Generated: {gen}",
        "",
        f"Label: {args.label}",
        f"Baseline exposures: rerisk={args.rerisk}, hold={args.hold}, cap={args.cap}, derisk={derisk}",
        f"Candidate grid: {args.candidate_grid}",
        f"Train end: {train_end.date()}, Validation end: {val_end.date()}",
        "",
        "## Segment Performance (baseline vs optimized)",
        "",
        markdown_table(out_perf),
        "",
        f"## Overrides learned ({len(out_overrides)})",
        "",
        markdown_table(out_overrides.head(30)) if not out_overrides.empty else "No overrides learned.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("saved", perf_path)
    print("saved", ovr_path)
    print("saved", hist_path)
    print("saved", md_path)
    print("overrides", len(out_overrides))
    if not out_perf.empty:
        full = out_perf[out_perf["segment"] == "full"].iloc[0]
        print("full_base_terminal", float(full["base_terminal_wealth"]))
        print("full_opt_terminal", float(full["opt_terminal_wealth"]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
