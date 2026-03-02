from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ROOT / "reports" / "selected_table_checks"


def parse_grid(text: str, cast=float) -> list:
    vals = [cast(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty grid")
    return vals


def action_to_default(action: str, rerisk: float, hold: float, cap: float, derisk: float) -> float:
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


def load_regime_frames(label: str, exclude_tickers: set[str]) -> dict[str, pd.DataFrame]:
    paths = sorted(CHECKS.glob(f"decay_checks_*_{label}.csv"))
    if not paths:
        raise FileNotFoundError(f"No decay checks for {label}")

    frames: dict[str, pd.DataFrame] = {}
    for path in paths:
        ticker = path.name.split("decay_checks_")[1].split(f"_{label}.csv")[0]
        if ticker in exclude_tickers:
            continue
        df = pd.read_csv(path)
        if "date" not in df.columns or "price" not in df.columns:
            continue
        action_col = "policy_action" if "policy_action" in df.columns else ("action" if "action" in df.columns else None)
        if action_col is None:
            continue

        w = df.copy()
        w["date"] = pd.to_datetime(w["date"], errors="coerce", utc=True)
        w["date"] = w["date"].dt.tz_convert(None)
        w = w.dropna(subset=["date", "price", action_col]).sort_values("date")
        if w.empty:
            continue

        if "regime_combo" not in w.columns:
            w["regime_combo"] = "unknown"

        w["policy_action"] = w[action_col].astype(str)
        w["ret"] = pd.to_numeric(w["price"], errors="coerce").pct_change().fillna(0.0)
        w = w[["date", "ret", "regime_combo", "policy_action"]].copy()
        frames[ticker] = w

    if not frames:
        raise RuntimeError(f"No usable ticker frames for {label}")
    return frames


def state_key(regime_combo: str, action: str) -> str:
    return f"{regime_combo}::{action}"


def apply_persistence(states: pd.Series, persistence_days: int) -> pd.Series:
    if persistence_days <= 1:
        return states
    run_id = (states != states.shift(1)).cumsum()
    run_pos = states.groupby(run_id).cumcount() + 1
    stable = run_pos >= persistence_days
    out = states.copy()
    out.loc[~stable] = pd.NA
    return out


def prepare_ticker_data(
    frames: dict[str, pd.DataFrame],
    rerisk: float,
    hold: float,
    cap: float,
    derisk: float,
    persistence_days: int,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for ticker, df in frames.items():
        w = df.copy()
        raw_state = pd.Series(
            [state_key(r, a) for r, a in zip(w["regime_combo"], w["policy_action"])],
            index=w.index,
            dtype="object",
        )
        w["state"] = apply_persistence(raw_state, persistence_days)
        w["default_exp"] = [
            action_to_default(a, rerisk=rerisk, hold=hold, cap=cap, derisk=derisk)
            for a in w["policy_action"]
        ]
        out[ticker] = w[["date", "ret", "state", "default_exp"]].copy()
    return out


@dataclass
class SimResult:
    ret: pd.Series
    turnover_daily: pd.Series


def simulate_returns(
    ticker_data: dict[str, pd.DataFrame],
    overrides: dict[str, float],
    hold_fill: float,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> SimResult:
    parts = []
    turns = []

    for ticker, df in ticker_data.items():
        w = df.copy()
        if start is not None:
            w = w[w["date"] >= start]
        if end is not None:
            w = w[w["date"] < end]
        if w.empty:
            continue

        exp = w["default_exp"].astype(float).copy()
        if overrides:
            mask = w["state"].isin(overrides)
            if mask.any():
                exp.loc[mask] = w.loc[mask, "state"].map(overrides).astype(float)

        exp_lag = exp.shift(1).fillna(hold_fill)
        strat = pd.to_numeric(w["ret"], errors="coerce").fillna(0.0) * exp_lag
        turn = exp_lag.diff().abs().fillna(0.0)

        parts.append(pd.DataFrame({"date": w["date"], "strategy_ret": strat}))
        turns.append(pd.DataFrame({"date": w["date"], "turnover": turn}))

    if not parts:
        return SimResult(pd.Series(dtype=float), pd.Series(dtype=float))

    ret_df = pd.concat(parts, ignore_index=True)
    ret = ret_df.groupby("date", as_index=True)["strategy_ret"].mean().sort_index()

    turn_df = pd.concat(turns, ignore_index=True)
    turnover = turn_df.groupby("date", as_index=True)["turnover"].mean().sort_index()
    return SimResult(ret=ret, turnover_daily=turnover)


@dataclass
class Perf:
    rows: int
    terminal_wealth: float
    cagr: float
    sharpe: float
    max_drawdown: float
    avg_daily_ret: float
    turnover_mean: float


def perf_from_sim(sim: SimResult) -> Perf:
    r = pd.to_numeric(sim.ret, errors="coerce").fillna(0.0)
    t = pd.to_numeric(sim.turnover_daily, errors="coerce").fillna(0.0)
    if r.empty:
        return Perf(0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    clipped = r.clip(lower=-0.999999)
    wealth = (1.0 + clipped).cumprod()
    rows = len(r)
    years = rows / 252.0
    terminal = float(wealth.iloc[-1])
    cagr = float(terminal ** (1.0 / years) - 1.0) if years > 0 else np.nan
    vol = float(r.std(ddof=1)) if rows > 1 else np.nan
    sharpe = float(np.sqrt(252.0) * r.mean() / vol) if np.isfinite(vol) and vol > 0 else np.nan
    maxdd = float((wealth / wealth.cummax() - 1.0).min())
    return Perf(
        rows=int(rows),
        terminal_wealth=terminal,
        cagr=cagr,
        sharpe=sharpe,
        max_drawdown=maxdd,
        avg_daily_ret=float(r.mean()),
        turnover_mean=float(t.mean()) if not t.empty else 0.0,
    )


def objective(
    perf: Perf,
    *,
    min_maxdd: float,
    dd_penalty: float,
    turnover_penalty: float,
    complexity_penalty: float,
    num_overrides: int,
) -> float:
    if not np.isfinite(perf.terminal_wealth) or perf.rows == 0:
        return -1e12
    score = float(np.log(max(perf.terminal_wealth, 1e-300)))
    dd_shortfall = max(0.0, min_maxdd - perf.max_drawdown)
    score -= dd_penalty * dd_shortfall
    score -= turnover_penalty * perf.turnover_mean
    score -= complexity_penalty * num_overrides
    return score


def build_folds(dates: pd.Index, train_frac: float, val_frac: float, step_frac: float) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    uniq = pd.Index(sorted(pd.to_datetime(pd.Series(dates).dropna().unique())))
    n = len(uniq)
    if n < 120:
        return []

    train_n = max(60, int(n * train_frac))
    val_n = max(30, int(n * val_frac))
    step_n = max(15, int(n * step_frac))

    folds = []
    start_idx = 0
    while True:
        train_start_i = start_idx
        train_end_i = train_start_i + train_n
        val_end_i = train_end_i + val_n
        if val_end_i >= n:
            break
        folds.append((uniq[train_start_i], uniq[train_end_i], uniq[val_end_i]))
        start_idx += step_n
    return folds


def optimize_overrides_train(
    ticker_data: dict[str, pd.DataFrame],
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    candidate_grid: list[float],
    hold_fill: float,
    min_maxdd: float,
    dd_penalty: float,
    turnover_penalty: float,
    complexity_penalty: float,
    max_states: int,
    max_iters: int,
) -> dict[str, float]:
    state_frames = []
    for df in ticker_data.values():
        w = df[(df["date"] >= train_start) & (df["date"] < train_end)]
        if not w.empty:
            state_frames.append(w[["state"]])
    if not state_frames:
        return {}

    states_df = pd.concat(state_frames, ignore_index=True)
    counts = states_df["state"].dropna().value_counts()
    states = counts.head(max_states).index.tolist()

    overrides: dict[str, float] = {}
    current = perf_from_sim(simulate_returns(ticker_data, overrides, hold_fill, train_start, train_end))
    current_score = objective(
        current,
        min_maxdd=min_maxdd,
        dd_penalty=dd_penalty,
        turnover_penalty=turnover_penalty,
        complexity_penalty=complexity_penalty,
        num_overrides=len(overrides),
    )

    for _ in range(max_iters):
        improved = False
        for state in states:
            best_val = overrides.get(state, None)
            best_score = current_score
            for val in candidate_grid:
                trial = dict(overrides)
                trial[state] = float(val)
                perf = perf_from_sim(simulate_returns(ticker_data, trial, hold_fill, train_start, train_end))
                score = objective(
                    perf,
                    min_maxdd=min_maxdd,
                    dd_penalty=dd_penalty,
                    turnover_penalty=turnover_penalty,
                    complexity_penalty=complexity_penalty,
                    num_overrides=len(trial),
                )
                if score > best_score + 1e-12:
                    best_score = score
                    best_val = float(val)
            if best_val is not None and overrides.get(state) != best_val:
                overrides[state] = best_val
                current_score = best_score
                improved = True
        if not improved:
            break
    return overrides


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Regime portfolio optimizer v2 (walk-forward + penalties)")
    parser.add_argument("--labels", required=True, help="Comma-separated labels")
    parser.add_argument("--rerisk-grid", default="0.2")
    parser.add_argument("--hold-grid", default="0.05")
    parser.add_argument("--cap-grid", default="0.05")
    parser.add_argument("--derisk-grid", default="0.0")
    parser.add_argument("--candidate-grid", default="-0.05,0.0,0.05,0.1,0.15,0.2")
    parser.add_argument("--exclude", default="^VIX,^VVIX")
    parser.add_argument("--persistence-days", type=int, default=2)
    parser.add_argument("--train-frac", type=float, default=0.55)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--step-frac", type=float, default=0.1)
    parser.add_argument("--min-maxdd", type=float, default=-0.05)
    parser.add_argument("--dd-penalty", type=float, default=300.0)
    parser.add_argument("--turnover-penalty", type=float, default=5.0)
    parser.add_argument("--complexity-penalty", type=float, default=0.002)
    parser.add_argument("--max-states", type=int, default=20)
    parser.add_argument("--max-iters", type=int, default=4)
    parser.add_argument("--out-prefix", default="reports/regime_optimizer_v2")
    args = parser.parse_args()

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    rerisk_vals = parse_grid(args.rerisk_grid)
    hold_vals = parse_grid(args.hold_grid)
    cap_vals = parse_grid(args.cap_grid)
    derisk_vals = parse_grid(args.derisk_grid)
    candidate_grid = parse_grid(args.candidate_grid)
    exclude = {x.strip() for x in args.exclude.split(",") if x.strip()}

    all_rows = []
    fold_rows = []

    for label in labels:
        frames = load_regime_frames(label, exclude)
        combo_iter = list(itertools.product(rerisk_vals, hold_vals, cap_vals, derisk_vals))

        for rerisk, hold, cap, derisk in combo_iter:
            ticker_data = prepare_ticker_data(
                frames,
                rerisk=float(rerisk),
                hold=float(hold),
                cap=float(cap),
                derisk=float(derisk),
                persistence_days=args.persistence_days,
            )
            date_series = pd.concat([df["date"] for df in ticker_data.values()], ignore_index=True)
            folds = build_folds(
                pd.Index(pd.to_datetime(date_series, errors="coerce").dropna()),
                train_frac=args.train_frac,
                val_frac=args.val_frac,
                step_frac=args.step_frac,
            )
            if not folds:
                continue

            fold_scores = []
            fold_terminal = []
            fold_sharpe = []
            fold_maxdd = []
            fold_turn = []
            fold_overrides = []
            pass_count = 0

            for fold_id, (train_start, train_end, val_end) in enumerate(folds, start=1):
                overrides = optimize_overrides_train(
                    ticker_data,
                    train_start=train_start,
                    train_end=train_end,
                    candidate_grid=candidate_grid,
                    hold_fill=float(hold),
                    min_maxdd=args.min_maxdd,
                    dd_penalty=args.dd_penalty,
                    turnover_penalty=args.turnover_penalty,
                    complexity_penalty=args.complexity_penalty,
                    max_states=args.max_states,
                    max_iters=args.max_iters,
                )

                val_sim = simulate_returns(ticker_data, overrides, float(hold), train_end, val_end)
                val_perf = perf_from_sim(val_sim)
                val_score = objective(
                    val_perf,
                    min_maxdd=args.min_maxdd,
                    dd_penalty=args.dd_penalty,
                    turnover_penalty=args.turnover_penalty,
                    complexity_penalty=args.complexity_penalty,
                    num_overrides=len(overrides),
                )

                fold_scores.append(val_score)
                fold_terminal.append(val_perf.terminal_wealth)
                fold_sharpe.append(val_perf.sharpe)
                fold_maxdd.append(val_perf.max_drawdown)
                fold_turn.append(val_perf.turnover_mean)
                fold_overrides.append(len(overrides))

                passes = (
                    (val_perf.terminal_wealth > 1.0)
                    and (val_perf.max_drawdown > args.min_maxdd)
                )
                pass_count += int(passes)

                fold_rows.append(
                    {
                        "label": label,
                        "rerisk": rerisk,
                        "hold": hold,
                        "cap": cap,
                        "derisk": derisk,
                        "fold": fold_id,
                        "train_start": train_start,
                        "train_end": train_end,
                        "val_end": val_end,
                        "val_terminal_wealth": val_perf.terminal_wealth,
                        "val_sharpe": val_perf.sharpe,
                        "val_maxdd": val_perf.max_drawdown,
                        "val_turnover_mean": val_perf.turnover_mean,
                        "val_score": val_score,
                        "overrides_count": len(overrides),
                        "fold_pass": bool(passes),
                    }
                )

            if not fold_scores:
                continue

            row = {
                "label": label,
                "rerisk": rerisk,
                "hold": hold,
                "cap": cap,
                "derisk": derisk,
                "folds": len(fold_scores),
                "median_val_score": float(np.median(fold_scores)),
                "median_val_terminal_wealth": float(np.median(fold_terminal)),
                "median_val_sharpe": float(np.median(fold_sharpe)),
                "median_val_maxdd": float(np.median(fold_maxdd)),
                "mean_val_turnover": float(np.mean(fold_turn)),
                "mean_overrides": float(np.mean(fold_overrides)),
                "pass_rate": float(pass_count / len(fold_scores)),
            }
            all_rows.append(row)

    leaderboard = pd.DataFrame(all_rows)
    folds_df = pd.DataFrame(fold_rows)

    if not leaderboard.empty:
        leaderboard = leaderboard.sort_values(
            ["pass_rate", "median_val_score", "median_val_terminal_wealth", "median_val_sharpe"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
        leaderboard.insert(0, "rank", range(1, len(leaderboard) + 1))

    prefix = ROOT / args.out_prefix
    out_csv = Path(str(prefix) + "_leaderboard.csv")
    out_fold = Path(str(prefix) + "_folds.csv")
    out_md = Path(str(prefix) + "_report.md")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(out_csv, index=False)
    folds_df.to_csv(out_fold, index=False)

    gen = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Regime Optimizer v2 Report",
        "",
        f"Generated: {gen}",
        "",
        "## Config",
        f"- labels: {args.labels}",
        f"- rerisk-grid: {args.rerisk_grid}",
        f"- hold-grid: {args.hold_grid}",
        f"- cap-grid: {args.cap_grid}",
        f"- derisk-grid: {args.derisk_grid}",
        f"- candidate-grid: {args.candidate_grid}",
        f"- persistence-days: {args.persistence_days}",
        f"- penalties: dd={args.dd_penalty}, turnover={args.turnover_penalty}, complexity={args.complexity_penalty}",
        "",
        "## Leaderboard",
        markdown_table(leaderboard.head(20)) if not leaderboard.empty else "No rows.",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("saved", out_csv)
    print("saved", out_fold)
    print("saved", out_md)
    print("rows", len(leaderboard))
    if not leaderboard.empty:
        print("top_label", leaderboard.iloc[0]["label"])
        print("top_score", float(leaderboard.iloc[0]["median_val_score"]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
