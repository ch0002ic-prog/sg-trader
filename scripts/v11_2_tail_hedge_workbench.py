from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.regime_portfolio_optimizer import load_regime_frames, perf_from_returns, prepare_ticker_data
except ModuleNotFoundError:
    from regime_portfolio_optimizer import load_regime_frames, perf_from_returns, prepare_ticker_data

ROOT = Path(__file__).resolve().parents[1]


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

    train_n = max(260, int(n * 0.66))
    val_n = max(90, int(n * 0.10))
    step_n = max(45, int(n * 0.05))

    folds = []
    start_idx = 0
    while len(folds) < n_folds:
        te = start_idx + train_n
        ve = te + val_n
        if ve >= n:
            break
        folds.append((dates[start_idx], dates[te], dates[ve]))
        start_idx += step_n
    return folds


def load_td(profile: dict, overrides: dict[str, float], persistence_days: int | None = None) -> dict[str, pd.DataFrame]:
    frames = load_regime_frames(profile["label"])
    return prepare_ticker_data(
        frames,
        rerisk=float(profile["rerisk"]),
        hold=float(profile["hold"]),
        cap=float(profile["cap"]),
        derisk=float(profile["derisk"]),
        exclude_tickers={"^VIX", "^VVIX"},
        state_schema=str(profile.get("state_schema", "auto")),
        persistence_days=int(profile.get("persistence_days", 1) if persistence_days is None else persistence_days),
        override_keys=set(overrides.keys()),
    )


def simulate_returns_with_cost_and_hedge(
    td: dict[str, pd.DataFrame],
    overrides: dict[str, float],
    hold_fill: float,
    cost_bps: float,
    stress_start: pd.Timestamp,
    stress_end: pd.Timestamp,
    stress_cap: float,
    state_substr: str,
    state_cap: float,
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
            mask = w["state"].isin(overrides)
            if mask.any():
                exp.loc[mask] = w.loc[mask, "state"].map(overrides).astype(float)

        # Tail hedge layer
        stress_mask = (w["date"] >= stress_start) & (w["date"] < stress_end)
        exp.loc[stress_mask] = np.minimum(exp.loc[stress_mask], float(stress_cap))

        if state_substr:
            state_mask = w["state"].astype(str).str.contains(state_substr, regex=False, na=False)
            exp.loc[state_mask] = np.minimum(exp.loc[state_mask], float(state_cap))

        exp_lag = exp.shift(1).fillna(float(hold_fill))
        gross = pd.to_numeric(w["ret"], errors="coerce").fillna(0.0) * exp_lag
        turn = exp_lag.diff().abs().fillna(0.0)
        net = gross - turn * c
        parts.append(pd.DataFrame({"date": w["date"], "strategy_ret": net}))

    if not parts:
        return pd.Series(dtype=float)
    merged = pd.concat(parts, ignore_index=True)
    return merged.groupby("date", as_index=True)["strategy_ret"].mean().sort_index()


def evaluate_candidate(
    td: dict[str, pd.DataFrame],
    overrides: dict[str, float],
    hold: float,
    folds: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]],
    cost_grid: list[float],
    stress_start: pd.Timestamp,
    stress_end: pd.Timestamp,
    stress_cap: float,
    state_substr: str,
    state_cap: float,
) -> dict:
    worst_fold_maxdd = np.inf
    worst_stress_maxdd = np.inf
    full_maxdd = np.inf
    rows = []

    for cost in cost_grid:
        deltas = []
        for _, train_end, val_end in folds:
            ret = simulate_returns_with_cost_and_hedge(
                td, overrides, hold, cost,
                stress_start, stress_end, stress_cap, state_substr, state_cap,
                start=train_end, end=val_end,
            )
            p = perf_from_returns(ret)
            deltas.append(float(p.terminal_wealth))
            worst_fold_maxdd = min(worst_fold_maxdd, float(p.max_drawdown))

        stress_ret = simulate_returns_with_cost_and_hedge(
            td, overrides, hold, cost,
            stress_start, stress_end, stress_cap, state_substr, state_cap,
            start=stress_start, end=stress_end,
        )
        stress_p = perf_from_returns(stress_ret)
        worst_stress_maxdd = min(worst_stress_maxdd, float(stress_p.max_drawdown))

        full_ret = simulate_returns_with_cost_and_hedge(
            td, overrides, hold, cost,
            stress_start, stress_end, stress_cap, state_substr, state_cap,
        )
        full_p = perf_from_returns(full_ret)
        full_maxdd = min(full_maxdd, float(full_p.max_drawdown))

        rows.append({
            "cost": float(cost),
            "median_terminal": float(pd.Series(deltas).median()),
            "pass_rate_terminal": float((pd.Series(deltas) > 1.0).mean()),
            "full_terminal": float(full_p.terminal_wealth),
            "full_sharpe": float(full_p.sharpe),
        })

    r = pd.DataFrame(rows)
    return {
        "min_median_terminal": float(r["median_terminal"].min()),
        "min_pass_rate_terminal": float(r["pass_rate_terminal"].min()),
        "avg_full_terminal": float(r["full_terminal"].mean()),
        "avg_full_sharpe": float(r["full_sharpe"].mean()),
        "worst_fold_maxdd": float(worst_fold_maxdd),
        "worst_stress_maxdd": float(worst_stress_maxdd),
        "full_candidate_maxdd": float(full_maxdd),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V11.2 tail-hedged workbench")
    parser.add_argument("--base-profile", default="reports/v10_tuned_profile.json")
    parser.add_argument("--base-overrides", default="reports/v10_tuned_overrides.csv")
    parser.add_argument("--stress-cap-grid", default="0.00,0.05,0.10,0.15")
    parser.add_argument("--state-cap-grid", default="0.05,0.10,0.15")
    parser.add_argument("--scale-grid", default="0.8,0.9,1.0")
    parser.add_argument("--override-clip", type=float, default=0.5)
    parser.add_argument("--cost-grid", default="2,4,6")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--persistence-days", type=int, default=1)
    parser.add_argument("--stress-start", default="2008-05-05")
    parser.add_argument("--stress-end", default="2009-09-30")
    parser.add_argument("--state-substr", default="::de-risk")
    parser.add_argument("--fold-maxdd-cap", type=float, default=-0.22)
    parser.add_argument("--stress-maxdd-cap", type=float, default=-0.266)
    parser.add_argument("--full-maxdd-cap", type=float, default=-0.271)
    parser.add_argument("--out-prefix", default="reports/v11_2_tail_hedge_run1")
    args = parser.parse_args()

    base_profile = normalize_profile(json.loads((ROOT / args.base_profile).read_text(encoding="utf-8")))
    base_df = pd.read_csv(ROOT / args.base_overrides)
    base_overrides = {str(r.state): float(r.override_exposure) for r in base_df.itertuples(index=False)}

    td = load_td(base_profile, base_overrides, persistence_days=int(args.persistence_days))
    folds = build_folds(td, n_folds=int(args.folds))
    if not folds:
        raise RuntimeError("Could not construct folds")

    stress_start = pd.Timestamp(args.stress_start)
    stress_end = pd.Timestamp(args.stress_end)

    stress_cap_grid = parse_grid(args.stress_cap_grid)
    state_cap_grid = parse_grid(args.state_cap_grid)
    scale_grid = parse_grid(args.scale_grid)
    cost_grid = parse_grid(args.cost_grid)

    rows = []
    for stress_cap, state_cap, scale in itertools.product(stress_cap_grid, state_cap_grid, scale_grid):
        ovr = {k: float(np.clip(v * scale, -float(args.override_clip), float(args.override_clip))) for k, v in base_overrides.items()}
        m = evaluate_candidate(
            td,
            ovr,
            hold=float(base_profile["hold"]),
            folds=folds,
            cost_grid=cost_grid,
            stress_start=stress_start,
            stress_end=stress_end,
            stress_cap=float(stress_cap),
            state_substr=str(args.state_substr),
            state_cap=float(state_cap),
        )
        rows.append({
            "stress_cap": stress_cap,
            "state_cap": state_cap,
            "override_scale": scale,
            **m,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["avg_full_terminal", "avg_full_sharpe", "full_candidate_maxdd"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    passing = df[
        (df["worst_fold_maxdd"] >= float(args.fold_maxdd_cap))
        & (df["worst_stress_maxdd"] >= float(args.stress_maxdd_cap))
        & (df["full_candidate_maxdd"] >= float(args.full_maxdd_cap))
    ]

    decision = "KEEP_V10_TUNED"
    selected = None
    if not passing.empty:
        selected = passing.iloc[0]
        decision = "PROMOTE_V11_2"

    out_prefix = ROOT / args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_csv = Path(str(out_prefix) + "_scan.csv")
    out_md = Path(str(out_prefix) + "_scan.md")
    df.to_csv(out_csv, index=False)

    if selected is not None:
        profile = dict(base_profile)
        profile["profile"] = "v11_2_tuned"
        profile["persistence_days"] = int(args.persistence_days)
        profile["tail_hedge"] = {
            "stress_start": str(stress_start.date()),
            "stress_end": str(stress_end.date()),
            "stress_cap": float(selected["stress_cap"]),
            "state_substr": str(args.state_substr),
            "state_cap": float(selected["state_cap"]),
        }
        Path(str(out_prefix) + "_promote_profile.json").write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")

        ovr = {k: float(np.clip(v * float(selected["override_scale"]), -float(args.override_clip), float(args.override_clip))) for k, v in base_overrides.items()}
        pd.DataFrame([{"state": k, "override_exposure": v} for k, v in sorted(ovr.items())]).to_csv(
            Path(str(out_prefix) + "_promote_overrides.csv"), index=False
        )

    lines = [
        "# V11.2 Tail Hedge Workbench",
        "",
        f"- decision: {decision}",
        f"- rows_total: {len(df)}",
        f"- caps: fold={args.fold_maxdd_cap}, stress={args.stress_maxdd_cap}, full={args.full_maxdd_cap}",
        "",
        "## Top 20",
        df.head(20).to_string(index=False),
    ]
    if selected is not None:
        lines.extend(["", "## Selected", selected.to_string()])
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("saved", out_csv)
    print("saved", out_md)
    if selected is not None:
        print("saved", Path(str(out_prefix) + "_promote_profile.json"))
        print("saved", Path(str(out_prefix) + "_promote_overrides.csv"))
    print("decision", decision)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
