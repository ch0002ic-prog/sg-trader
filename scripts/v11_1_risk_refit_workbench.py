from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.regime_portfolio_optimizer import load_regime_frames, perf_from_returns, prepare_ticker_data
except ModuleNotFoundError:
    from regime_portfolio_optimizer import load_regime_frames, perf_from_returns, prepare_ticker_data

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def normalize_profile(payload: dict, mode: str = "aggressive") -> dict:
    if all(k in payload for k in ["rerisk", "hold", "cap", "derisk"]):
        out = dict(payload)
        out.setdefault("persistence_days", 1)
        out.setdefault("state_schema", "auto")
        return out
    if mode in payload and isinstance(payload[mode], dict):
        p = payload[mode]
        return {
            "profile": payload.get("profile", mode),
            "label": payload["label"],
            "rerisk": float(p["rerisk"]),
            "hold": float(p["hold"]),
            "cap": float(p["cap"]),
            "derisk": float(p["derisk"]),
            "persistence_days": int(p.get("persistence_days", 1)),
            "state_schema": "auto",
        }
    raise KeyError("Profile does not contain usable exposure keys")


def parse_grid(text: str, cast=float) -> list:
    vals = [cast(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty grid")
    return vals


def build_folds(td: dict[str, pd.DataFrame], n_folds: int) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    dates = pd.Index(sorted(pd.to_datetime(pd.concat([df["date"] for df in td.values()], ignore_index=True).dropna().unique())))
    n = len(dates)
    if n < 500:
        return []

    train_frac = 0.66
    val_frac = 0.10
    step_frac = 0.05

    train_n = max(260, int(n * train_frac))
    val_n = max(90, int(n * val_frac))
    step_n = max(45, int(n * step_frac))

    folds = []
    start_idx = 0
    while len(folds) < n_folds:
        train_end_i = start_idx + train_n
        val_end_i = train_end_i + val_n
        if val_end_i >= n:
            break
        folds.append((dates[start_idx], dates[train_end_i], dates[val_end_i]))
        start_idx += step_n

    return folds


def load_td(profile: dict, overrides: dict[str, float], persistence_days: int) -> dict[str, pd.DataFrame]:
    frames = load_regime_frames(profile["label"])
    return prepare_ticker_data(
        frames,
        rerisk=float(profile["rerisk"]),
        hold=float(profile["hold"]),
        cap=float(profile["cap"]),
        derisk=float(profile["derisk"]),
        exclude_tickers={"^VIX", "^VVIX"},
        state_schema=str(profile.get("state_schema", "auto")),
        persistence_days=int(persistence_days),
        override_keys=set(overrides.keys()),
    )


def simulate_returns_with_cost(
    td: dict[str, pd.DataFrame],
    overrides: dict[str, float],
    hold_fill: float,
    cost_bps: float,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.Series:
    parts = []
    c = float(cost_bps) / 10000.0

    for _, df in td.items():
        w = df.copy()
        if start is not None:
            w = w[w["date"] >= start]
        if end is not None:
            w = w[w["date"] < end]
        if w.empty:
            continue

        exp = w["default_exp"].astype(float).copy()
        if overrides:
            mask = w["state"].isin(overrides.keys())
            if mask.any():
                exp.loc[mask] = w.loc[mask, "state"].map(overrides).astype(float)

        exp_lag = exp.shift(1).fillna(float(hold_fill))
        gross = pd.to_numeric(w["ret"], errors="coerce").fillna(0.0) * exp_lag
        turnover = exp_lag.diff().abs().fillna(0.0)
        net = gross - turnover * c
        parts.append(pd.DataFrame({"date": w["date"], "strategy_ret": net}))

    if not parts:
        return pd.Series(dtype=float)

    merged = pd.concat(parts, ignore_index=True)
    return merged.groupby("date", as_index=True)["strategy_ret"].mean().sort_index()


def evaluate(
    td: dict[str, pd.DataFrame],
    overrides: dict[str, float],
    hold: float,
    cost_grid: list[float],
    folds: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]],
    stress_start: pd.Timestamp,
    stress_end: pd.Timestamp,
    base_cache: dict,
) -> dict:
    rows = []
    worst_fold_maxdd = np.inf
    worst_stress_maxdd = np.inf
    full_maxdd_worst = np.inf

    for cost in cost_grid:
        fold_deltas = []
        fold_sh_deltas = []

        full_ret = simulate_returns_with_cost(td, overrides, hold, cost)
        full_perf = perf_from_returns(full_ret)
        full_maxdd_worst = min(full_maxdd_worst, float(full_perf.max_drawdown))

        stress_ret = simulate_returns_with_cost(td, overrides, hold, cost, start=stress_start, end=stress_end)
        stress_perf = perf_from_returns(stress_ret)
        worst_stress_maxdd = min(worst_stress_maxdd, float(stress_perf.max_drawdown))

        for i, (_, train_end, val_end) in enumerate(folds):
            cand_ret = simulate_returns_with_cost(td, overrides, hold, cost, start=train_end, end=val_end)
            cand_perf = perf_from_returns(cand_ret)
            worst_fold_maxdd = min(worst_fold_maxdd, float(cand_perf.max_drawdown))

            base_fold = base_cache[float(cost)]["folds"][i]
            fold_deltas.append(float(cand_perf.terminal_wealth - base_fold["terminal"]))
            fold_sh_deltas.append(float(cand_perf.sharpe - base_fold["sharpe"]))

        rows.append(
            {
                "cost": float(cost),
                "median_delta_terminal": float(pd.Series(fold_deltas).median()),
                "median_delta_sharpe": float(pd.Series(fold_sh_deltas).median()),
                "pass_rate_terminal": float((pd.Series(fold_deltas) > 0).mean()),
            }
        )

    r = pd.DataFrame(rows)
    return {
        "min_median_delta_terminal": float(r["median_delta_terminal"].min()),
        "min_pass_rate_terminal": float(r["pass_rate_terminal"].min()),
        "avg_median_delta_terminal": float(r["median_delta_terminal"].mean()),
        "avg_median_delta_sharpe": float(r["median_delta_sharpe"].mean()),
        "worst_fold_maxdd": float(worst_fold_maxdd),
        "worst_stress_maxdd": float(worst_stress_maxdd),
        "full_candidate_maxdd": float(full_maxdd_worst),
    }


def build_base_cache(
    td: dict[str, pd.DataFrame],
    overrides: dict[str, float],
    hold: float,
    cost_grid: list[float],
    folds: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]],
) -> dict:
    cache: dict = {}
    for cost in cost_grid:
        frows = []
        for _, train_end, val_end in folds:
            ret = simulate_returns_with_cost(td, overrides, hold, cost, start=train_end, end=val_end)
            perf = perf_from_returns(ret)
            frows.append({"terminal": float(perf.terminal_wealth), "sharpe": float(perf.sharpe)})
        cache[float(cost)] = {"folds": frows}
    return cache


def main() -> int:
    parser = argparse.ArgumentParser(description="V11.1 risk-first override refit (per-day regimes)")
    parser.add_argument("--base-profile", default="reports/v10_tuned_profile.json")
    parser.add_argument("--base-overrides", default="reports/v10_tuned_overrides.csv")
    parser.add_argument("--candidate-grid", default="-0.2,-0.1,0.0,0.1,0.2,0.3,0.4")
    parser.add_argument("--max-states", type=int, default=8)
    parser.add_argument("--max-iters", type=int, default=2)
    parser.add_argument("--cost-grid", default="2,4,6")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--persistence-days", type=int, default=1)
    parser.add_argument("--stress-start", default="2008-05-05")
    parser.add_argument("--stress-end", default="2009-09-30")
    parser.add_argument("--min-pass-rate", type=float, default=0.67)
    parser.add_argument("--fold-maxdd-cap", type=float, default=-0.22)
    parser.add_argument("--stress-maxdd-cap", type=float, default=-0.266)
    parser.add_argument("--full-maxdd-cap", type=float, default=-0.271)
    parser.add_argument("--out-prefix", default="reports/v11_1_risk_refit_run1")
    args = parser.parse_args()

    base_profile = normalize_profile(json.loads((ROOT / args.base_profile).read_text(encoding="utf-8")))
    base_ovr_df = pd.read_csv(ROOT / args.base_overrides)
    base_overrides = {str(r.state): float(r.override_exposure) for r in base_ovr_df.itertuples(index=False)}

    td = load_td(base_profile, base_overrides, persistence_days=int(args.persistence_days))
    folds = build_folds(td, n_folds=int(args.folds))
    if not folds:
        raise RuntimeError("Could not construct folds")

    state_counts = pd.concat([d["state"].dropna().astype(str) for d in td.values()], ignore_index=True).value_counts()
    states = state_counts.head(int(args.max_states)).index.tolist()

    candidate_grid = parse_grid(args.candidate_grid)
    cost_grid = parse_grid(args.cost_grid)
    hold = float(base_profile["hold"])
    stress_start = pd.Timestamp(args.stress_start)
    stress_end = pd.Timestamp(args.stress_end)

    base_cache = build_base_cache(td, base_overrides, hold, cost_grid, folds)

    current = dict(base_overrides)
    current_eval = evaluate(td, current, hold, cost_grid, folds, stress_start, stress_end, base_cache)

    history = []

    def feasible(m: dict) -> bool:
        return (
            float(m["min_pass_rate_terminal"]) >= float(args.min_pass_rate)
            and float(m["worst_fold_maxdd"]) >= float(args.fold_maxdd_cap)
            and float(m["worst_stress_maxdd"]) >= float(args.stress_maxdd_cap)
            and float(m["full_candidate_maxdd"]) >= float(args.full_maxdd_cap)
        )

    current_score = float(current_eval["min_median_delta_terminal"])

    for it in range(1, int(args.max_iters) + 1):
        improved = False
        for st in states:
            best_val = current.get(st, None)
            best_eval = current_eval
            best_score = current_score
            for val in candidate_grid:
                trial = dict(current)
                trial[st] = float(val)
                m = evaluate(td, trial, hold, cost_grid, folds, stress_start, stress_end, base_cache)
                score = float(m["min_median_delta_terminal"])
                if feasible(m) and (score > best_score + 1e-12):
                    best_score = score
                    best_val = float(val)
                    best_eval = m
            if best_val is not None and current.get(st) != best_val:
                current[st] = float(best_val)
                current_eval = best_eval
                current_score = best_score
                improved = True

        history.append(
            {
                "iter": it,
                "improved": improved,
                **current_eval,
                "active_states": len(current),
            }
        )
        if not improved:
            break

    out_prefix = ROOT / args.out_prefix
    out_csv = Path(str(out_prefix) + "_summary.csv")
    out_hist = Path(str(out_prefix) + "_history.csv")
    out_md = Path(str(out_prefix) + "_report.md")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame([
        {
            "profile": base_profile.get("profile", "v10_tuned"),
            "persistence_days": int(args.persistence_days),
            **current_eval,
            "decision": "PROMOTE_V11_1" if feasible(current_eval) and float(current_eval["min_median_delta_terminal"]) > 0 else "KEEP_V10_TUNED",
        }
    ])
    summary.to_csv(out_csv, index=False)
    pd.DataFrame(history).to_csv(out_hist, index=False)

    if feasible(current_eval) and float(current_eval["min_median_delta_terminal"]) > 0:
        promoted_profile = dict(base_profile)
        promoted_profile["profile"] = "v11_1_tuned"
        promoted_profile["persistence_days"] = int(args.persistence_days)
        Path(str(out_prefix) + "_promote_profile.json").write_text(json.dumps(promoted_profile, indent=2) + "\n", encoding="utf-8")
        pd.DataFrame([{"state": k, "override_exposure": v} for k, v in sorted(current.items())]).to_csv(
            Path(str(out_prefix) + "_promote_overrides.csv"), index=False
        )

    lines = [
        "# V11.1 Risk Refit Report",
        "",
        f"- profile: {base_profile.get('profile', 'v10_tuned')}",
        f"- per-day regimes: persistence_days={int(args.persistence_days)}",
        f"- fold_maxdd_cap: {args.fold_maxdd_cap}",
        f"- stress_maxdd_cap: {args.stress_maxdd_cap}",
        f"- full_maxdd_cap: {args.full_maxdd_cap}",
        "",
        "## Summary",
        summary.to_string(index=False),
        "",
        "## History",
        pd.DataFrame(history).to_string(index=False) if history else "No optimization steps",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("saved", out_csv)
    print("saved", out_hist)
    print("saved", out_md)
    if (Path(str(out_prefix) + "_promote_profile.json")).exists():
        print("saved", Path(str(out_prefix) + "_promote_profile.json"))
        print("saved", Path(str(out_prefix) + "_promote_overrides.csv"))
    print(summary.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
