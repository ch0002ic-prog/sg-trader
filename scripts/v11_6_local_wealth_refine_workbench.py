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
    values = [cast(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError("Empty grid")
    return values


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
        train_end = start_idx + train_n
        val_end = train_end + val_n
        if val_end >= n:
            break
        folds.append((dates[start_idx], dates[train_end], dates[val_end]))
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


def apply_state_specific_scaling(
    base_overrides: dict[str, float],
    overall_scale: float,
    risk_on_scale: float,
    derisk_scale: float,
    panic_scale: float,
    derisk_substr: str,
    panic_substr: str,
    override_clip: float,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for state, value in base_overrides.items():
        state_text = str(state)
        scale = overall_scale
        if panic_substr and panic_substr in state_text:
            scale *= panic_scale
        elif derisk_substr and derisk_substr in state_text:
            scale *= derisk_scale
        else:
            scale *= risk_on_scale
        out[state] = float(np.clip(value * scale, -override_clip, override_clip))
    return out


def simulate_returns_with_cost_and_hedge(
    td: dict[str, pd.DataFrame],
    overrides: dict[str, float],
    hold_fill: float,
    cost_bps: float,
    stress1_start: pd.Timestamp,
    stress1_end: pd.Timestamp,
    stress2_start: pd.Timestamp,
    stress2_end: pd.Timestamp,
    stress_cap: float,
    derisk_substr: str,
    derisk_cap: float,
    panic_substr: str,
    panic_cap: float,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.Series:
    cost = float(cost_bps) / 10000.0
    parts = []

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

        stress_mask = ((w["date"] >= stress1_start) & (w["date"] < stress1_end)) | ((w["date"] >= stress2_start) & (w["date"] < stress2_end))
        exp.loc[stress_mask] = np.minimum(exp.loc[stress_mask], float(stress_cap))

        if derisk_substr:
            derisk_mask = w["state"].astype(str).str.contains(derisk_substr, regex=False, na=False)
            exp.loc[derisk_mask] = np.minimum(exp.loc[derisk_mask], float(derisk_cap))
        if panic_substr:
            panic_mask = w["state"].astype(str).str.contains(panic_substr, regex=False, na=False)
            exp.loc[panic_mask] = np.minimum(exp.loc[panic_mask], float(panic_cap))

        exp_lag = exp.shift(1).fillna(float(hold_fill))
        gross = pd.to_numeric(w["ret"], errors="coerce").fillna(0.0) * exp_lag
        turnover = exp_lag.diff().abs().fillna(0.0)
        net = gross - turnover * cost
        parts.append(pd.DataFrame({"date": w["date"], "strategy_ret": net}))

    if not parts:
        return pd.Series(dtype=float)
    merged = pd.concat(parts, ignore_index=True)
    return merged.groupby("date", as_index=True)["strategy_ret"].mean().sort_index()


def shift_window(start: pd.Timestamp, end: pd.Timestamp, shift_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return start + pd.Timedelta(days=int(shift_days)), end + pd.Timedelta(days=int(shift_days))


def evaluate_candidate(
    td: dict[str, pd.DataFrame],
    overrides: dict[str, float],
    hold: float,
    folds: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]],
    cost_grid: list[float],
    stress1_start: pd.Timestamp,
    stress1_end: pd.Timestamp,
    stress2_start: pd.Timestamp,
    stress2_end: pd.Timestamp,
    jitter_days: list[int],
    stress_cap: float,
    derisk_substr: str,
    derisk_cap: float,
    panic_substr: str,
    panic_cap: float,
) -> dict:
    worst_fold_maxdd = np.inf
    worst_stress_maxdd = np.inf
    full_maxdd = np.inf
    rows = []

    for cost in cost_grid:
        fold_terminals = []
        for _, train_end, val_end in folds:
            ret = simulate_returns_with_cost_and_hedge(
                td,
                overrides,
                hold,
                cost,
                stress1_start,
                stress1_end,
                stress2_start,
                stress2_end,
                stress_cap,
                derisk_substr,
                derisk_cap,
                panic_substr,
                panic_cap,
                start=train_end,
                end=val_end,
            )
            p = perf_from_returns(ret)
            fold_terminals.append(float(p.terminal_wealth))
            worst_fold_maxdd = min(worst_fold_maxdd, float(p.max_drawdown))

        for shift_1 in jitter_days:
            for shift_2 in jitter_days:
                s1_start, s1_end = shift_window(stress1_start, stress1_end, int(shift_1))
                s2_start, s2_end = shift_window(stress2_start, stress2_end, int(shift_2))

                stress1_ret = simulate_returns_with_cost_and_hedge(
                    td,
                    overrides,
                    hold,
                    cost,
                    s1_start,
                    s1_end,
                    s2_start,
                    s2_end,
                    stress_cap,
                    derisk_substr,
                    derisk_cap,
                    panic_substr,
                    panic_cap,
                    start=s1_start,
                    end=s1_end,
                )
                stress1_p = perf_from_returns(stress1_ret)

                stress2_ret = simulate_returns_with_cost_and_hedge(
                    td,
                    overrides,
                    hold,
                    cost,
                    s1_start,
                    s1_end,
                    s2_start,
                    s2_end,
                    stress_cap,
                    derisk_substr,
                    derisk_cap,
                    panic_substr,
                    panic_cap,
                    start=s2_start,
                    end=s2_end,
                )
                stress2_p = perf_from_returns(stress2_ret)

                worst_stress_maxdd = min(worst_stress_maxdd, float(stress1_p.max_drawdown), float(stress2_p.max_drawdown))

        full_ret = simulate_returns_with_cost_and_hedge(
            td,
            overrides,
            hold,
            cost,
            stress1_start,
            stress1_end,
            stress2_start,
            stress2_end,
            stress_cap,
            derisk_substr,
            derisk_cap,
            panic_substr,
            panic_cap,
        )
        full_p = perf_from_returns(full_ret)
        full_maxdd = min(full_maxdd, float(full_p.max_drawdown))

        rows.append(
            {
                "cost": float(cost),
                "median_terminal": float(pd.Series(fold_terminals).median()),
                "pass_rate_terminal": float((pd.Series(fold_terminals) > 1.0).mean()),
                "full_terminal": float(full_p.terminal_wealth),
                "full_sharpe": float(full_p.sharpe),
            }
        )

    metrics = pd.DataFrame(rows)
    return {
        "min_median_terminal": float(metrics["median_terminal"].min()),
        "min_pass_rate_terminal": float(metrics["pass_rate_terminal"].min()),
        "avg_full_terminal": float(metrics["full_terminal"].mean()),
        "avg_full_sharpe": float(metrics["full_sharpe"].mean()),
        "worst_fold_maxdd": float(worst_fold_maxdd),
        "worst_stress_maxdd": float(worst_stress_maxdd),
        "full_candidate_maxdd": float(full_maxdd),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V11.6 local wealth refinement workbench")
    parser.add_argument("--base-profile", default="reports/v11_5_tuned_profile.json")
    parser.add_argument("--base-overrides", default="reports/v11_5_tuned_overrides.csv")
    parser.add_argument("--stress-cap-grid", default="0.14,0.16,0.18")
    parser.add_argument("--derisk-cap-grid", default="0.30,0.34,0.38")
    parser.add_argument("--panic-cap-grid", default="0.06,0.08,0.10,0.12")
    parser.add_argument("--overall-scale-grid", default="0.98,1.00,1.02")
    parser.add_argument("--risk-on-scale-grid", default="0.98,1.00,1.02,1.04")
    parser.add_argument("--derisk-scale-grid", default="0.98,1.00,1.02")
    parser.add_argument("--panic-scale-grid", default="0.80,0.85,0.90")
    parser.add_argument("--override-clip", type=float, default=0.7)
    parser.add_argument("--cost-grid", default="2,4,6,8,10")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--persistence-days", type=int, default=1)
    parser.add_argument("--stress1-start", default="2008-05-05")
    parser.add_argument("--stress1-end", default="2009-09-30")
    parser.add_argument("--stress2-start", default="2020-02-20")
    parser.add_argument("--stress2-end", default="2020-06-30")
    parser.add_argument("--stress-jitter-days", default="-5,0,5")
    parser.add_argument("--derisk-substr", default="::de-risk")
    parser.add_argument("--panic-substr", default="panic")
    parser.add_argument("--fold-maxdd-cap", type=float, default=-0.22)
    parser.add_argument("--stress-maxdd-cap", type=float, default=-0.266)
    parser.add_argument("--full-maxdd-cap", type=float, default=-0.271)
    parser.add_argument("--min-margin-floor", type=float, default=0.003)
    parser.add_argument("--out-prefix", default="reports/v11_6_local_wealth_refine_run1")
    args = parser.parse_args()

    base_profile = normalize_profile(json.loads((ROOT / args.base_profile).read_text(encoding="utf-8")))
    base_df = pd.read_csv(ROOT / args.base_overrides)
    states = base_df["state"].astype(str)
    exposures = pd.to_numeric(base_df["override_exposure"], errors="coerce").fillna(0.0).astype(float)
    base_overrides = dict(zip(states, exposures))

    td = load_td(base_profile, base_overrides, persistence_days=int(args.persistence_days))
    folds = build_folds(td, n_folds=int(args.folds))
    if not folds:
        raise RuntimeError("Could not construct folds")

    stress1_start = pd.Timestamp(args.stress1_start)
    stress1_end = pd.Timestamp(args.stress1_end)
    stress2_start = pd.Timestamp(args.stress2_start)
    stress2_end = pd.Timestamp(args.stress2_end)

    stress_cap_grid = parse_grid(args.stress_cap_grid)
    derisk_cap_grid = parse_grid(args.derisk_cap_grid)
    panic_cap_grid = parse_grid(args.panic_cap_grid)
    overall_scale_grid = parse_grid(args.overall_scale_grid)
    risk_on_scale_grid = parse_grid(args.risk_on_scale_grid)
    derisk_scale_grid = parse_grid(args.derisk_scale_grid)
    panic_scale_grid = parse_grid(args.panic_scale_grid)
    cost_grid = parse_grid(args.cost_grid)
    jitter_days = parse_grid(args.stress_jitter_days, cast=int)

    rows = []
    for stress_cap, derisk_cap, panic_cap, overall_scale, risk_on_scale, derisk_scale, panic_scale in itertools.product(
        stress_cap_grid,
        derisk_cap_grid,
        panic_cap_grid,
        overall_scale_grid,
        risk_on_scale_grid,
        derisk_scale_grid,
        panic_scale_grid,
    ):
        overrides = apply_state_specific_scaling(
            base_overrides=base_overrides,
            overall_scale=float(overall_scale),
            risk_on_scale=float(risk_on_scale),
            derisk_scale=float(derisk_scale),
            panic_scale=float(panic_scale),
            derisk_substr=str(args.derisk_substr),
            panic_substr=str(args.panic_substr),
            override_clip=float(args.override_clip),
        )

        m = evaluate_candidate(
            td=td,
            overrides=overrides,
            hold=float(base_profile["hold"]),
            folds=folds,
            cost_grid=cost_grid,
            stress1_start=stress1_start,
            stress1_end=stress1_end,
            stress2_start=stress2_start,
            stress2_end=stress2_end,
            jitter_days=jitter_days,
            stress_cap=float(stress_cap),
            derisk_substr=str(args.derisk_substr),
            derisk_cap=float(derisk_cap),
            panic_substr=str(args.panic_substr),
            panic_cap=float(panic_cap),
        )

        fold_margin = float(m["worst_fold_maxdd"] - float(args.fold_maxdd_cap))
        stress_margin = float(m["worst_stress_maxdd"] - float(args.stress_maxdd_cap))
        full_margin = float(m["full_candidate_maxdd"] - float(args.full_maxdd_cap))
        min_margin = float(min(fold_margin, stress_margin, full_margin))
        wealth_dd_ratio = float(m["avg_full_terminal"] / max(abs(float(m["full_candidate_maxdd"])), 1e-6))

        rows.append(
            {
                "stress_cap": float(stress_cap),
                "derisk_cap": float(derisk_cap),
                "panic_cap": float(panic_cap),
                "overall_scale": float(overall_scale),
                "risk_on_scale": float(risk_on_scale),
                "derisk_scale": float(derisk_scale),
                "panic_scale": float(panic_scale),
                **m,
                "fold_margin": fold_margin,
                "stress_margin": stress_margin,
                "full_margin": full_margin,
                "min_margin": min_margin,
                "wealth_dd_ratio": wealth_dd_ratio,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["avg_full_terminal", "min_median_terminal", "avg_full_sharpe", "full_candidate_maxdd"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    caps_passing = df[
        (df["worst_fold_maxdd"] >= float(args.fold_maxdd_cap))
        & (df["worst_stress_maxdd"] >= float(args.stress_maxdd_cap))
        & (df["full_candidate_maxdd"] >= float(args.full_maxdd_cap))
    ]
    buffer_passing = caps_passing[caps_passing["min_margin"] >= float(args.min_margin_floor)]

    selected = None
    decision = "KEEP_V11_5_TUNED"
    selection_mode = "none"
    if not buffer_passing.empty:
        selected = buffer_passing.iloc[0]
        decision = "PROMOTE_V11_6"
        selection_mode = "buffer-passing"
    elif not caps_passing.empty:
        selected = caps_passing.iloc[0]
        decision = "PROMOTE_V11_6"
        selection_mode = "caps-passing"

    out_prefix = ROOT / args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_csv = Path(str(out_prefix) + "_scan.csv")
    out_md = Path(str(out_prefix) + "_scan.md")
    df.to_csv(out_csv, index=False)

    if selected is not None:
        profile = dict(base_profile)
        profile["profile"] = "v11_6_tuned"
        profile["persistence_days"] = int(args.persistence_days)
        profile["tail_hedge"] = {
            "stress1_start": str(stress1_start.date()),
            "stress1_end": str(stress1_end.date()),
            "stress2_start": str(stress2_start.date()),
            "stress2_end": str(stress2_end.date()),
            "stress_cap": float(selected["stress_cap"]),
            "state_substr": str(args.derisk_substr),
            "state_cap": float(selected["derisk_cap"]),
            "panic_substr": str(args.panic_substr),
            "panic_cap": float(selected["panic_cap"]),
            "stress_jitter_days": [int(x) for x in jitter_days],
            "selection_mode": selection_mode,
            "min_margin_floor": float(args.min_margin_floor),
            "state_scaling": {
                "overall_scale": float(selected["overall_scale"]),
                "risk_on_scale": float(selected["risk_on_scale"]),
                "derisk_scale": float(selected["derisk_scale"]),
                "panic_scale": float(selected["panic_scale"]),
            },
        }
        Path(str(out_prefix) + "_promote_profile.json").write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")

        selected_overrides = apply_state_specific_scaling(
            base_overrides=base_overrides,
            overall_scale=float(selected["overall_scale"]),
            risk_on_scale=float(selected["risk_on_scale"]),
            derisk_scale=float(selected["derisk_scale"]),
            panic_scale=float(selected["panic_scale"]),
            derisk_substr=str(args.derisk_substr),
            panic_substr=str(args.panic_substr),
            override_clip=float(args.override_clip),
        )
        pd.DataFrame(
            [{"state": key, "override_exposure": value} for key, value in sorted(selected_overrides.items())]
        ).to_csv(Path(str(out_prefix) + "_promote_overrides.csv"), index=False)

    lines = [
        "# V11.6 Local Wealth Refinement Workbench",
        "",
        f"- decision: {decision}",
        f"- selection_mode: {selection_mode}",
        f"- rows_total: {len(df)}",
        f"- rows_caps_passing: {len(caps_passing)}",
        f"- rows_buffer_passing: {len(buffer_passing)}",
        f"- caps: fold={args.fold_maxdd_cap}, stress={args.stress_maxdd_cap}, full={args.full_maxdd_cap}",
        f"- min_margin_floor: {args.min_margin_floor}",
        f"- stress_jitter_days: {jitter_days}",
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
