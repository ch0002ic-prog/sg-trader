from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.regime_portfolio_optimizer import (
        load_regime_frames,
        perf_from_returns,
        prepare_ticker_data,
        simulate_returns,
    )
except ModuleNotFoundError:
    from regime_portfolio_optimizer import (
        load_regime_frames,
        perf_from_returns,
        prepare_ticker_data,
        simulate_returns,
    )

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FOLD_MAXDD_CAP = -0.22
STRESS_MAXDD_CAP = -0.266
FULL_MAXDD_CAP = -0.271


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


def load_profile(profile_path: Path, mode: str = "aggressive") -> dict:
    return normalize_profile(json.loads(profile_path.read_text(encoding="utf-8")), mode=mode)


def load_overrides(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    states = df["state"].astype(str)
    exposures = pd.to_numeric(df["override_exposure"], errors="coerce").fillna(0.0).astype(float)
    return dict(zip(states, exposures))


def build_ticker_data(profile: dict, overrides: dict[str, float]) -> dict[str, pd.DataFrame]:
    frames = load_regime_frames(profile["label"])
    return prepare_ticker_data(
        frames,
        rerisk=float(profile["rerisk"]),
        hold=float(profile["hold"]),
        cap=float(profile["cap"]),
        derisk=float(profile["derisk"]),
        exclude_tickers={"^VIX", "^VVIX"},
        state_schema=str(profile.get("state_schema", "auto")),
        persistence_days=int(profile.get("persistence_days", 1)),
        override_keys=set(overrides.keys()),
    )


def _tail_hedge_params(profile: dict) -> tuple[pd.Timestamp | None, pd.Timestamp | None, pd.Timestamp | None, pd.Timestamp | None, str, float | None, float | None, str, float | None]:
    hedge = profile.get("tail_hedge") if isinstance(profile.get("tail_hedge"), dict) else {}
    if not hedge:
        return None, None, None, None, "", None, None, "", None

    s1_start = hedge.get("stress1_start", hedge.get("stress_start"))
    s1_end = hedge.get("stress1_end", hedge.get("stress_end"))
    s2_start = hedge.get("stress2_start")
    s2_end = hedge.get("stress2_end")
    state_substr = str(hedge.get("state_substr", ""))
    stress_cap = float(hedge["stress_cap"]) if hedge.get("stress_cap") is not None else None
    state_cap = float(hedge["state_cap"]) if hedge.get("state_cap") is not None else None
    panic_substr = str(hedge.get("panic_substr", ""))
    panic_cap = float(hedge["panic_cap"]) if hedge.get("panic_cap") is not None else None

    return (
        pd.Timestamp(s1_start) if s1_start else None,
        pd.Timestamp(s1_end) if s1_end else None,
        pd.Timestamp(s2_start) if s2_start else None,
        pd.Timestamp(s2_end) if s2_end else None,
        state_substr,
        stress_cap,
        state_cap,
        panic_substr,
        panic_cap,
    )


def simulate_returns_profile_aware(td: dict[str, pd.DataFrame], overrides: dict[str, float], hold_fill: float, profile: dict) -> pd.Series:
    s1_start, s1_end, s2_start, s2_end, state_substr, stress_cap, state_cap, panic_substr, panic_cap = _tail_hedge_params(profile)
    use_tail_hedge = any(x is not None for x in (s1_start, s1_end, s2_start, s2_end, stress_cap, state_cap, panic_cap)) or bool(state_substr) or bool(panic_substr)
    if not use_tail_hedge:
        return simulate_returns(td, overrides, float(hold_fill))

    parts = []
    for _, df in td.items():
        w = df.copy()
        if w.empty:
            continue

        exp = w["default_exp"].astype(float).copy()
        if overrides:
            mask = w["state"].isin(overrides)
            if mask.any():
                exp.loc[mask] = w.loc[mask, "state"].map(overrides).astype(float)

        if stress_cap is not None:
            stress_mask = pd.Series(False, index=w.index)
            if s1_start is not None and s1_end is not None:
                stress_mask = stress_mask | ((w["date"] >= s1_start) & (w["date"] < s1_end))
            if s2_start is not None and s2_end is not None:
                stress_mask = stress_mask | ((w["date"] >= s2_start) & (w["date"] < s2_end))
            exp.loc[stress_mask] = np.minimum(exp.loc[stress_mask], float(stress_cap))

        if state_substr and state_cap is not None:
            state_mask = w["state"].astype(str).str.contains(state_substr, regex=False, na=False)
            exp.loc[state_mask] = np.minimum(exp.loc[state_mask], float(state_cap))

        if panic_substr and panic_cap is not None:
            panic_mask = w["state"].astype(str).str.contains(panic_substr, regex=False, na=False)
            exp.loc[panic_mask] = np.minimum(exp.loc[panic_mask], float(panic_cap))

        exp_lag = exp.shift(1).fillna(float(hold_fill))
        ret = pd.to_numeric(w["ret"], errors="coerce").fillna(0.0) * exp_lag
        parts.append(pd.DataFrame({"date": w["date"], "strategy_ret": ret}))

    if not parts:
        return pd.Series(dtype=float)
    merged = pd.concat(parts, ignore_index=True)
    return merged.groupby("date", as_index=True)["strategy_ret"].mean().sort_index()


def run_perf(td: dict[str, pd.DataFrame], overrides: dict[str, float], hold: float, profile: dict) -> dict:
    p = perf_from_returns(simulate_returns_profile_aware(td, overrides, float(hold), profile))
    return {
        "terminal_wealth": float(p.terminal_wealth),
        "sharpe": float(p.sharpe),
        "maxdd": float(p.max_drawdown),
    }


def run_perf_legacy_reference(td: dict[str, pd.DataFrame], overrides: dict[str, float], hold: float) -> dict:
    p = perf_from_returns(simulate_returns(td, overrides, float(hold)))
    return {
        "terminal_wealth": float(p.terminal_wealth),
        "sharpe": float(p.sharpe),
        "maxdd": float(p.max_drawdown),
    }


def run_smoke(label: str, profile: dict, overrides: dict[str, float]) -> pd.DataFrame:
    frames = load_regime_frames(label)
    base_td = prepare_ticker_data(
        frames,
        rerisk=float(profile["rerisk"]),
        hold=float(profile["hold"]),
        cap=float(profile["cap"]),
        derisk=float(profile["derisk"]),
        exclude_tickers={"^VIX", "^VVIX"},
    )
    auto_td = prepare_ticker_data(
        frames,
        rerisk=float(profile["rerisk"]),
        hold=float(profile["hold"]),
        cap=float(profile["cap"]),
        derisk=float(profile["derisk"]),
        exclude_tickers={"^VIX", "^VVIX"},
        state_schema="auto",
        persistence_days=int(profile.get("persistence_days", 1)),
        override_keys=set(overrides.keys()),
    )
    p_base = perf_from_returns(simulate_returns_profile_aware(base_td, {}, float(profile["hold"]), profile))
    p_auto = perf_from_returns(simulate_returns_profile_aware(auto_td, overrides, float(profile["hold"]), profile))
    return pd.DataFrame(
        [
            {
                "mode": "base_standard",
                "terminal_wealth": float(p_base.terminal_wealth),
                "sharpe": float(p_base.sharpe),
                "maxdd": float(p_base.max_drawdown),
            },
            {
                "mode": "with_overrides_auto",
                "terminal_wealth": float(p_auto.terminal_wealth),
                "sharpe": float(p_auto.sharpe),
                "maxdd": float(p_auto.max_drawdown),
            },
        ]
    )


def pick_oot_row(oot_df: pd.DataFrame) -> pd.DataFrame:
    if oot_df.empty:
        return oot_df.head(1).copy()

    required = {"worst_fold_maxdd", "worst_stress_maxdd", "full_candidate_maxdd"}
    if required.issubset(set(oot_df.columns)):
        passing = oot_df[
            (oot_df["worst_fold_maxdd"] >= FOLD_MAXDD_CAP)
            & (oot_df["worst_stress_maxdd"] >= STRESS_MAXDD_CAP)
            & (oot_df["full_candidate_maxdd"] >= FULL_MAXDD_CAP)
        ]
        if not passing.empty:
            if "robust_score" in passing.columns:
                return passing.sort_values(["robust_score", "avg_full_terminal"], ascending=[False, False]).head(1).copy()
            return passing.head(1).copy()

    return oot_df.head(1).copy()


def main() -> int:
    has_v11_5 = (REPORTS / "v11_5_tuned_profile.json").exists() and (REPORTS / "v11_5_tuned_overrides.csv").exists()
    has_v11_4 = (REPORTS / "v11_4_tuned_profile.json").exists() and (REPORTS / "v11_4_tuned_overrides.csv").exists()
    has_v11_3 = (REPORTS / "v11_3_tuned_profile.json").exists() and (REPORTS / "v11_3_tuned_overrides.csv").exists()
    has_v11_2 = (REPORTS / "v11_2_tuned_profile.json").exists() and (REPORTS / "v11_2_tuned_overrides.csv").exists()
    has_v10 = (REPORTS / "v10_tuned_profile.json").exists() and (REPORTS / "v10_tuned_overrides.csv").exists()
    has_v9 = (REPORTS / "v9_tuned_profile.json").exists() and (REPORTS / "v9_tuned_overrides.csv").exists()
    has_v8 = (REPORTS / "v8_tuned_profile.json").exists() and (REPORTS / "v8_tuned_overrides.csv").exists()
    has_v7 = (REPORTS / "v7_tuned_profile.json").exists() and (REPORTS / "v7_tuned_overrides.csv").exists()

    if has_v11_5 and has_v11_3:
        baseline_name = "v11_3_tuned"
        baseline_profile = load_profile(REPORTS / "v11_3_tuned_profile.json", mode="aggressive")
        baseline_overrides = load_overrides(REPORTS / "v11_3_tuned_overrides.csv")

        current_name = "v11_5_tuned"
        current_profile = load_profile(REPORTS / "v11_5_tuned_profile.json", mode="aggressive")
        current_overrides = load_overrides(REPORTS / "v11_5_tuned_overrides.csv")

        oot_path = REPORTS / "v11_5_wealth_unlock_run1_scan.csv"
        keep_decision = "KEEP_V11_3_TUNED"
        promote_decision = "PROMOTED_V11_5_TUNED"
    elif has_v11_4 and has_v11_3:
        baseline_name = "v11_3_tuned"
        baseline_profile = load_profile(REPORTS / "v11_3_tuned_profile.json", mode="aggressive")
        baseline_overrides = load_overrides(REPORTS / "v11_3_tuned_overrides.csv")

        current_name = "v11_4_tuned"
        current_profile = load_profile(REPORTS / "v11_4_tuned_profile.json", mode="aggressive")
        current_overrides = load_overrides(REPORTS / "v11_4_tuned_overrides.csv")

        oot_path = REPORTS / "v11_4_robust_frontier_run1_scan.csv"
        keep_decision = "KEEP_V11_3_TUNED"
        promote_decision = "PROMOTED_V11_4_TUNED"
    elif has_v11_3 and has_v11_2:
        baseline_name = "v11_2_tuned"
        baseline_profile = load_profile(REPORTS / "v11_2_tuned_profile.json", mode="aggressive")
        baseline_overrides = load_overrides(REPORTS / "v11_2_tuned_overrides.csv")

        current_name = "v11_3_tuned"
        current_profile = load_profile(REPORTS / "v11_3_tuned_profile.json", mode="aggressive")
        current_overrides = load_overrides(REPORTS / "v11_3_tuned_overrides.csv")

        oot_path = REPORTS / "v11_3_tail_hedge_run1_scan.csv"
        keep_decision = "KEEP_V11_2_TUNED"
        promote_decision = "PROMOTED_V11_3_TUNED"
    elif has_v11_2 and has_v10:
        baseline_name = "v10_tuned"
        baseline_profile = load_profile(REPORTS / "v10_tuned_profile.json", mode="aggressive")
        baseline_overrides = load_overrides(REPORTS / "v10_tuned_overrides.csv")

        current_name = "v11_2_tuned"
        current_profile = load_profile(REPORTS / "v11_2_tuned_profile.json", mode="aggressive")
        current_overrides = load_overrides(REPORTS / "v11_2_tuned_overrides.csv")

        oot_path = REPORTS / "v11_2_tail_hedge_run1_scan.csv"
        keep_decision = "KEEP_V10_TUNED"
        promote_decision = "PROMOTED_V11_2_TUNED"
    elif has_v10 and has_v9:
        baseline_name = "v9_tuned"
        baseline_profile = load_profile(REPORTS / "v9_tuned_profile.json", mode="aggressive")
        baseline_overrides = load_overrides(REPORTS / "v9_tuned_overrides.csv")

        current_name = "v10_tuned"
        current_profile = load_profile(REPORTS / "v10_tuned_profile.json", mode="aggressive")
        current_overrides = load_overrides(REPORTS / "v10_tuned_overrides.csv")

        oot_path = REPORTS / "v10_promotion_workbench_run1_scan.csv"
        keep_decision = "KEEP_V9_TUNED"
        promote_decision = "PROMOTED_V10_TUNED"
    elif has_v9 and has_v8:
        baseline_name = "v8_tuned"
        baseline_profile = load_profile(REPORTS / "v8_tuned_profile.json", mode="aggressive")
        baseline_overrides = load_overrides(REPORTS / "v8_tuned_overrides.csv")

        current_name = "v9_tuned"
        current_profile = load_profile(REPORTS / "v9_tuned_profile.json", mode="aggressive")
        current_overrides = load_overrides(REPORTS / "v9_tuned_overrides.csv")

        oot_path = REPORTS / "v9_promotion_workbench_run1_scan.csv"
        keep_decision = "KEEP_V8_TUNED"
        promote_decision = "PROMOTED_V9_TUNED"
    elif has_v8 and has_v7:
        baseline_name = "v7_tuned"
        baseline_profile = load_profile(REPORTS / "v7_tuned_profile.json", mode="aggressive")
        baseline_overrides = load_overrides(REPORTS / "v7_tuned_overrides.csv")

        current_name = "v8_tuned"
        current_profile = load_profile(REPORTS / "v8_tuned_profile.json", mode="aggressive")
        current_overrides = load_overrides(REPORTS / "v8_tuned_overrides.csv")

        oot_path = REPORTS / "v8_promotion_workbench_stress_scan.csv"
        keep_decision = "KEEP_V7_TUNED"
        promote_decision = "PROMOTED_V8_TUNED"
    elif has_v7:
        baseline_name = "v6_native_tuned"
        baseline_profile = load_profile(REPORTS / "v6_tuned_profile.json", mode="aggressive")
        baseline_overrides = load_overrides(REPORTS / "v6_tuned_overrides.csv")

        current_name = "v7_tuned"
        current_profile = load_profile(REPORTS / "v7_tuned_profile.json", mode="aggressive")
        current_overrides = load_overrides(REPORTS / "v7_tuned_overrides.csv")

        oot_path = REPORTS / "v7_promotion_workbench_run1_scan.csv"
        keep_decision = "KEEP_V6_TUNED"
        promote_decision = "PROMOTED_V7_TUNED"
    else:
        baseline_name = "v5_aggressive"
        baseline_profile = load_profile(REPORTS / "v5_aggressive_profile.json", mode="aggressive")
        baseline_overrides = load_overrides(REPORTS / "v5_aggressive_overrides.csv")

        current_name = "v6_native_tuned"
        current_profile = load_profile(REPORTS / "v6_tuned_profile.json", mode="aggressive")
        current_overrides = load_overrides(REPORTS / "v6_tuned_overrides.csv")

        oot_path = REPORTS / "v6_promotion_workbench_run2_scan.csv"
        keep_decision = "KEEP_V5"
        promote_decision = "PROMOTED_V6_TUNED"

    baseline_td = build_ticker_data(baseline_profile, baseline_overrides)
    current_td = build_ticker_data(current_profile, current_overrides)

    baseline_base = run_perf(baseline_td, {}, baseline_profile["hold"], baseline_profile)
    baseline_opt = run_perf(baseline_td, baseline_overrides, baseline_profile["hold"], baseline_profile)
    current_base = run_perf(current_td, {}, current_profile["hold"], current_profile)
    current_opt = run_perf(current_td, current_overrides, current_profile["hold"], current_profile)
    baseline_opt_legacy = run_perf_legacy_reference(baseline_td, baseline_overrides, baseline_profile["hold"])
    current_opt_legacy = run_perf_legacy_reference(current_td, current_overrides, current_profile["hold"])

    full_df = pd.DataFrame(
        [
            {"profile": baseline_name, "mode": "base", **baseline_base},
            {"profile": baseline_name, "mode": "with_overrides", **baseline_opt},
            {"profile": current_name, "mode": "base", **current_base},
            {"profile": current_name, "mode": "with_overrides", **current_opt},
        ]
    )

    oot_df = pd.read_csv(oot_path) if oot_path.exists() else pd.DataFrame()
    oot_top = pick_oot_row(oot_df)

    active_label = (REPORTS / "active_profile_label.txt").read_text(encoding="utf-8").strip()
    active_profile = normalize_profile(json.loads((REPORTS / "active_profile_exposures.json").read_text(encoding="utf-8")))

    smoke_df = run_smoke(active_label, active_profile, load_overrides(REPORTS / "active_policy_overrides.csv"))

    delta_terminal = float(current_opt["terminal_wealth"] - baseline_opt["terminal_wealth"])
    oot_delta_col = "median_delta_terminal" if "median_delta_terminal" in oot_top.columns else (
        "min_median_delta_terminal" if "min_median_delta_terminal" in oot_top.columns else (
            "min_median_terminal" if "min_median_terminal" in oot_top.columns else None
        )
    )
    oot_pass_col = "pass_rate_terminal" if "pass_rate_terminal" in oot_top.columns else (
        "min_pass_rate_terminal" if "min_pass_rate_terminal" in oot_top.columns else None
    )

    oot_delta_val = float(oot_top.iloc[0][oot_delta_col]) if (not oot_top.empty and oot_delta_col is not None) else float("nan")
    oot_pass_val = float(oot_top.iloc[0][oot_pass_col]) if (not oot_top.empty and oot_pass_col is not None) else float("nan")

    promote_flag = bool(delta_terminal > 0 and not oot_top.empty and np.isfinite(oot_delta_val) and oot_delta_val > 0)
    decision = promote_decision if promote_flag else keep_decision

    out_csv = REPORTS / "promotion_dashboard_summary.csv"
    simulation_mode = "tail_hedge_aware"
    pd.DataFrame(
        [
            {
                "simulation_mode": simulation_mode,
                "active_profile": active_profile.get("profile", "unknown"),
                "active_label": active_label,
                "baseline_profile": baseline_name,
                "current_profile": current_name,
                "baseline_with_overrides_terminal": float(baseline_opt["terminal_wealth"]),
                "baseline_with_overrides_terminal_legacy_reference": float(baseline_opt_legacy["terminal_wealth"]),
                "current_with_overrides_terminal": float(current_opt["terminal_wealth"]),
                "current_with_overrides_terminal_legacy_reference": float(current_opt_legacy["terminal_wealth"]),
                "delta_terminal_current_minus_baseline": delta_terminal,
                "oot_median_delta_terminal": oot_delta_val,
                "oot_pass_rate_terminal": oot_pass_val,
                "decision": decision,
            }
        ]
    ).to_csv(out_csv, index=False)

    out_md = REPORTS / "promotion_dashboard.md"
    lines = [
        "# Promotion Dashboard",
        "",
        "## Promotion Changelog",
        f"- {baseline_name} -> {current_name}: delta_terminal={delta_terminal:.6f}, decision={decision}",
        "",
        "## Full-Sample Side-by-Side",
        full_df.to_string(index=False),
        "",
        "## OOT Selection (Top Row)",
        f"- source: {oot_path.name if oot_path.exists() else 'missing'}",
        oot_top.to_string(index=False) if not oot_top.empty else "No OOT rows found",
        "",
        "## Simulation Mode",
        f"- simulation_mode: {simulation_mode}",
        f"- baseline_with_overrides_terminal_legacy_reference: {float(baseline_opt_legacy['terminal_wealth']):.6f}",
        f"- current_with_overrides_terminal_legacy_reference: {float(current_opt_legacy['terminal_wealth']):.6f}",
        "",
        "## Active Smoke",
        smoke_df.to_string(index=False),
        "",
        "## Active State",
        f"- active_profile: {active_profile.get('profile', 'unknown')}",
        f"- active_label: {active_label}",
        "",
        "## Decision",
        f"- {decision}",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    changelog_path = REPORTS / "promotion_changelog.md"
    stamp = datetime.now().isoformat(timespec="seconds")
    if not changelog_path.exists():
        changelog_path.write_text("# Promotion Changelog\n\n", encoding="utf-8")
    with changelog_path.open("a", encoding="utf-8") as fh:
        fh.write(
            f"- {stamp} | {baseline_name} -> {current_name} | "
            f"delta_terminal={delta_terminal:.6f} | decision={decision} | "
            f"mode={simulation_mode}\n"
        )

    print("saved", out_csv)
    print("saved", out_md)
    print("saved", changelog_path)
    print("decision", decision)
    print(full_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
