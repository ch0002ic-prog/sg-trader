from __future__ import annotations

import json
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


def normalize_active_profile(payload: dict) -> dict:
    if all(k in payload for k in ["rerisk", "hold", "cap", "derisk"]):
        out = dict(payload)
        out.setdefault("persistence_days", 1)
        out.setdefault("state_schema", "auto")
        return out
    for key in ["aggressive", "safe"]:
        if key in payload and isinstance(payload[key], dict):
            p = payload[key]
            return {
                "profile": payload.get("profile", key),
                "label": payload["label"],
                "rerisk": float(p["rerisk"]),
                "hold": float(p["hold"]),
                "cap": float(p["cap"]),
                "derisk": float(p["derisk"]),
                "persistence_days": int(p.get("persistence_days", 1)),
                "state_schema": "auto",
            }
    raise KeyError("active_profile_exposures.json does not contain usable exposure keys")


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


def main() -> int:
    active_label = (REPORTS / "active_profile_label.txt").read_text(encoding="utf-8").strip()
    active_profile = normalize_active_profile(
        json.loads((REPORTS / "active_profile_exposures.json").read_text(encoding="utf-8"))
    )
    overrides_df = pd.read_csv(REPORTS / "active_policy_overrides.csv")
    states = overrides_df["state"].astype(str)
    exposures = pd.to_numeric(overrides_df["override_exposure"], errors="coerce").fillna(0.0).astype(float)
    overrides = dict(zip(states, exposures))

    frames = load_regime_frames(active_label)

    td_standard = prepare_ticker_data(
        frames,
        rerisk=float(active_profile["rerisk"]),
        hold=float(active_profile["hold"]),
        cap=float(active_profile["cap"]),
        derisk=float(active_profile["derisk"]),
        exclude_tickers={"^VIX", "^VVIX"},
    )
    td_auto = prepare_ticker_data(
        frames,
        rerisk=float(active_profile["rerisk"]),
        hold=float(active_profile["hold"]),
        cap=float(active_profile["cap"]),
        derisk=float(active_profile["derisk"]),
        exclude_tickers={"^VIX", "^VVIX"},
        state_schema="auto",
        persistence_days=int(active_profile.get("persistence_days", 1)),
        override_keys=set(overrides.keys()),
    )

    p_base = perf_from_returns(simulate_returns_profile_aware(td_standard, {}, float(active_profile["hold"]), active_profile))
    p_opt = perf_from_returns(simulate_returns_profile_aware(td_auto, overrides, float(active_profile["hold"]), active_profile))

    rows = [
        {
            "mode": "base_standard",
            "terminal_wealth": float(p_base.terminal_wealth),
            "sharpe": float(p_base.sharpe),
            "maxdd": float(p_base.max_drawdown),
        },
        {
            "mode": "with_overrides_auto",
            "terminal_wealth": float(p_opt.terminal_wealth),
            "sharpe": float(p_opt.sharpe),
            "maxdd": float(p_opt.max_drawdown),
        },
    ]
    out = pd.DataFrame(rows)

    out_csv = REPORTS / "active_profile_smoke.csv"
    out_md = REPORTS / "active_profile_smoke.md"
    out.to_csv(out_csv, index=False)

    smoke_status = "PASS" if float(p_opt.terminal_wealth) > 1 else "CHECK"
    lines = [
        "# Active Profile Smoke",
        "",
        f"- active_label: {active_label}",
        f"- profile: {active_profile.get('profile', 'unknown')}",
        f"- exposures: rerisk={active_profile['rerisk']}, hold={active_profile['hold']}, cap={active_profile['cap']}, derisk={active_profile['derisk']}, persistence_days={active_profile.get('persistence_days', 1)}",
        f"- override_states: {len(overrides)}",
        "",
        "## Results",
        out.to_string(index=False),
        "",
        "## Decision",
        f"- smoke_status: {smoke_status}",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("saved", out_csv)
    print("saved", out_md)
    print(out.to_string(index=False))
    print("smoke_status", smoke_status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
