from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from regime_portfolio_optimizer_v2 import (
        optimize_overrides_train,
        perf_from_sim,
        simulate_returns,
    )
except ModuleNotFoundError:
    from scripts.regime_portfolio_optimizer_v2 import (
        optimize_overrides_train,
        perf_from_sim,
        simulate_returns,
    )

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ROOT / "reports" / "selected_table_checks"
REPORTS = ROOT / "reports"


def parse_grid(text: str, cast=float) -> list:
    values = [cast(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError("Empty grid")
    return values


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


def load_augmented_ticker_data(
    label: str,
    exclude_tickers: set[str],
    rerisk: float,
    hold: float,
    cap: float,
    derisk: float,
    persistence_days: int,
) -> dict[str, pd.DataFrame]:
    paths = sorted(CHECKS.glob(f"decay_checks_*_{label}.csv"))
    if not paths:
        raise FileNotFoundError(f"No decay checks for {label}")

    out: dict[str, pd.DataFrame] = {}
    for p in paths:
        ticker = p.name.split("decay_checks_")[1].split(f"_{label}.csv")[0]
        if ticker in exclude_tickers:
            continue

        df = pd.read_csv(p)
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
        if "rolling_sharpe" not in w.columns:
            w["rolling_sharpe"] = np.nan
        if "drawdown" not in w.columns:
            w["drawdown"] = np.nan

        rb = rs_bucket(w["rolling_sharpe"])
        db = dd_bucket(w["drawdown"])
        base_state = (
            w["regime_combo"].astype(str)
            + "::"
            + w[action_col].astype(str)
            + "::"
            + rb.astype(str)
            + "::"
            + db.astype(str)
        )
        state = apply_persistence(base_state, persistence_days)

        ret = pd.to_numeric(w["price"], errors="coerce").pct_change().fillna(0.0)
        default_exp = [
            action_to_default(a, rerisk=rerisk, hold=hold, cap=cap, derisk=derisk)
            for a in w[action_col]
        ]

        out[ticker] = pd.DataFrame(
            {
                "date": w["date"],
                "ret": ret,
                "state": state,
                "default_exp": default_exp,
            }
        )

    if not out:
        raise RuntimeError("No usable ticker data")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="v6 state-augmented local wealth scan")
    parser.add_argument("--label", required=True)
    parser.add_argument("--rerisk-grid", default="0.15,0.2,0.25")
    parser.add_argument("--hold-grid", default="0.15,0.2,0.25,0.3")
    parser.add_argument("--cap-grid", default="0.03,0.05")
    parser.add_argument("--derisk-grid", default="0.0,0.03,0.05,0.08")
    parser.add_argument("--persistence-grid", default="1,2,3")
    parser.add_argument("--candidate-grid", default="-0.2,-0.15,-0.1,-0.05,0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4")
    parser.add_argument("--min-maxdd", type=float, default=-0.12)
    parser.add_argument("--dd-penalty", type=float, default=80.0)
    parser.add_argument("--turnover-penalty", type=float, default=2.0)
    parser.add_argument("--complexity-penalty", type=float, default=0.001)
    parser.add_argument("--max-states", type=int, default=36)
    parser.add_argument("--max-iters", type=int, default=6)
    parser.add_argument("--exclude", default="^VIX,^VVIX")
    parser.add_argument("--out-prefix", default="reports/regime_optimizer_v6_state_aug")
    args = parser.parse_args()

    rerisk_grid = parse_grid(args.rerisk_grid)
    hold_grid = parse_grid(args.hold_grid)
    cap_grid = parse_grid(args.cap_grid)
    derisk_grid = parse_grid(args.derisk_grid)
    persistence_grid = parse_grid(args.persistence_grid, cast=int)
    candidate_grid = parse_grid(args.candidate_grid)
    exclude_tickers = {x.strip() for x in args.exclude.split(",") if x.strip()}

    combos = list(itertools.product(rerisk_grid, hold_grid, cap_grid, derisk_grid, persistence_grid))
    rows = []
    ovr_rows = []

    for idx, (rerisk, hold, cap, derisk, persistence) in enumerate(combos, start=1):
        if idx % 20 == 0 or idx == len(combos):
            print(f"progress {idx}/{len(combos)}")

        td = load_augmented_ticker_data(
            args.label,
            exclude_tickers=exclude_tickers,
            rerisk=rerisk,
            hold=hold,
            cap=cap,
            derisk=derisk,
            persistence_days=persistence,
        )

        all_dates = pd.concat([df["date"] for df in td.values()], ignore_index=True)
        start = pd.to_datetime(all_dates.min())
        end = pd.to_datetime(all_dates.max())

        overrides = optimize_overrides_train(
            td,
            train_start=start,
            train_end=end,
            candidate_grid=candidate_grid,
            hold_fill=hold,
            min_maxdd=args.min_maxdd,
            dd_penalty=args.dd_penalty,
            turnover_penalty=args.turnover_penalty,
            complexity_penalty=args.complexity_penalty,
            max_states=args.max_states,
            max_iters=args.max_iters,
        )

        base = perf_from_sim(simulate_returns(td, {}, hold, start, None))
        opt = perf_from_sim(simulate_returns(td, overrides, hold, start, None))

        rows.append(
            {
                "label": args.label,
                "rerisk": rerisk,
                "hold": hold,
                "cap": cap,
                "derisk": derisk,
                "persistence_days": persistence,
                "overrides_count": len(overrides),
                "base_terminal_wealth": base.terminal_wealth,
                "opt_terminal_wealth": opt.terminal_wealth,
                "delta_terminal_wealth": opt.terminal_wealth - base.terminal_wealth,
                "base_sharpe": base.sharpe,
                "opt_sharpe": opt.sharpe,
                "base_maxdd": base.max_drawdown,
                "opt_maxdd": opt.max_drawdown,
            }
        )

        for st, ov in overrides.items():
            ovr_rows.append(
                {
                    "rerisk": rerisk,
                    "hold": hold,
                    "cap": cap,
                    "derisk": derisk,
                    "persistence_days": persistence,
                    "state": st,
                    "override_exposure": ov,
                }
            )

    res = pd.DataFrame(rows)
    res = res.sort_values(["opt_terminal_wealth", "opt_sharpe"], ascending=[False, False]).reset_index(drop=True)
    res.insert(0, "rank", range(1, len(res) + 1))
    ovr = pd.DataFrame(ovr_rows)

    prefix = ROOT / args.out_prefix
    out_csv = Path(str(prefix) + "_scan.csv")
    out_ovr = Path(str(prefix) + "_scan_overrides.csv")
    out_best_ovr = Path(str(prefix) + "_best_overrides.csv")
    out_md = Path(str(prefix) + "_scan.md")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv, index=False)
    ovr.to_csv(out_ovr, index=False)

    best = res.iloc[0]
    best_mask = (
        (ovr["rerisk"] == best["rerisk"])
        & (ovr["hold"] == best["hold"])
        & (ovr["cap"] == best["cap"])
        & (ovr["derisk"] == best["derisk"])
        & (ovr["persistence_days"] == best["persistence_days"])
    )
    ovr[best_mask][["state", "override_exposure"]].to_csv(out_best_ovr, index=False)

    lines = [
        "# Regime Optimizer v6 State-Augmented Scan",
        "",
        f"rows: {len(res)}",
        "",
        "## Best row",
        best.to_string(),
        "",
        "## Top 20",
        res.head(20).to_string(index=False),
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("saved", out_csv)
    print("saved", out_ovr)
    print("saved", out_best_ovr)
    print("saved", out_md)
    print("best_opt_terminal", float(best["opt_terminal_wealth"]))
    print(
        "best_params",
        {
            "rerisk": float(best["rerisk"]),
            "hold": float(best["hold"]),
            "cap": float(best["cap"]),
            "derisk": float(best["derisk"]),
            "persistence_days": int(best["persistence_days"]),
        },
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
