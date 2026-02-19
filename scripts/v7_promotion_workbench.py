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

    train_frac = 0.70
    val_frac = 0.10
    step_frac = 0.07

    train_n = max(240, int(n * train_frac))
    val_n = max(80, int(n * val_frac))
    step_n = max(40, int(n * step_frac))

    folds = []
    start_idx = 0
    while len(folds) < n_folds:
        train_end_i = start_idx + train_n
        val_end_i = train_end_i + val_n
        if val_end_i >= n:
            break
        folds.append((dates[start_idx], dates[train_end_i], dates[val_end_i]))
        start_idx += step_n

    if len(folds) < n_folds:
        folds = []
        anchors = np.linspace(max(0, n - train_n - val_n * n_folds), n - train_n - val_n, n_folds, dtype=int)
        for anchor in anchors:
            a = int(anchor)
            te = a + train_n
            ve = te + val_n
            if ve < n:
                folds.append((dates[a], dates[te], dates[ve]))

    return folds


def load_td(profile: dict, overrides: dict[str, float]) -> dict[str, pd.DataFrame]:
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


def simulate_returns_with_cost(
    ticker_data: dict[str, pd.DataFrame],
    overrides: dict[str, float],
    hold_fill: float,
    cost_bps: float,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.Series:
    parts = []
    cost_rate = float(cost_bps) / 10000.0
    for _, df in ticker_data.items():
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
        net = gross - turnover * cost_rate
        parts.append(pd.DataFrame({"date": w["date"], "strategy_ret": net}))

    if not parts:
        return pd.Series(dtype=float)

    merged = pd.concat(parts, ignore_index=True)
    return merged.groupby("date", as_index=True)["strategy_ret"].mean().sort_index()


def evaluate_candidate(
    cand_profile: dict,
    cand_overrides: dict[str, float],
    base_profile: dict,
    base_overrides: dict[str, float],
    folds: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]],
    cost_bps: float,
) -> dict:
    cand_td = load_td(cand_profile, cand_overrides)
    base_td = load_td(base_profile, base_overrides)

    dtw = []
    dsh = []
    ddd = []

    for _, train_end, val_end in folds:
        cand_ret = simulate_returns_with_cost(cand_td, cand_overrides, float(cand_profile["hold"]), cost_bps, start=train_end, end=val_end)
        base_ret = simulate_returns_with_cost(base_td, base_overrides, float(base_profile["hold"]), cost_bps, start=train_end, end=val_end)
        pc = perf_from_returns(cand_ret)
        pb = perf_from_returns(base_ret)
        dtw.append(float(pc.terminal_wealth - pb.terminal_wealth))
        dsh.append(float(pc.sharpe - pb.sharpe))
        ddd.append(float(pc.max_drawdown - pb.max_drawdown))

    cand_full = perf_from_returns(simulate_returns_with_cost(cand_td, cand_overrides, float(cand_profile["hold"]), cost_bps))
    base_full = perf_from_returns(simulate_returns_with_cost(base_td, base_overrides, float(base_profile["hold"]), cost_bps))

    return {
        "median_delta_terminal": float(pd.Series(dtw).median()),
        "median_delta_sharpe": float(pd.Series(dsh).median()),
        "median_delta_maxdd": float(pd.Series(ddd).median()),
        "pass_rate_terminal": float((pd.Series(dtw) > 0).mean()),
        "full_delta_terminal": float(cand_full.terminal_wealth - base_full.terminal_wealth),
        "full_candidate_terminal": float(cand_full.terminal_wealth),
        "full_candidate_sharpe": float(cand_full.sharpe),
        "full_candidate_maxdd": float(cand_full.max_drawdown),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V7 promotion workbench (candidate vs active v6 tuned)")
    parser.add_argument("--base-profile", default="reports/v6_tuned_profile.json")
    parser.add_argument("--base-overrides", default="reports/v6_tuned_overrides.csv")
    parser.add_argument("--hold-grid", default="0.35,0.4,0.45")
    parser.add_argument("--rerisk-grid", default="0.1,0.15,0.2")
    parser.add_argument("--cap-grid", default="0.03,0.05")
    parser.add_argument("--derisk-grid", default="0.08,0.1,0.12")
    parser.add_argument("--scale-grid", default="1.0,1.1,1.2,1.25")
    parser.add_argument("--cost-bps", type=float, default=2.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-pass-rate", type=float, default=0.7)
    parser.add_argument("--dd-tol", type=float, default=-0.015)
    parser.add_argument("--out-prefix", default="reports/v7_promotion_workbench")
    args = parser.parse_args()

    base_profile = normalize_profile(json.loads((ROOT / args.base_profile).read_text(encoding="utf-8")))
    base_ovr_df = pd.read_csv(ROOT / args.base_overrides)
    base_overrides = {str(r.state): float(r.override_exposure) for r in base_ovr_df.itertuples(index=False)}

    base_td = load_td(base_profile, base_overrides)
    folds = build_folds(base_td, n_folds=int(args.folds))
    if not folds:
        raise RuntimeError("Could not construct OOT folds for v7 workbench")

    hold_grid = parse_grid(args.hold_grid)
    rerisk_grid = parse_grid(args.rerisk_grid)
    cap_grid = parse_grid(args.cap_grid)
    derisk_grid = parse_grid(args.derisk_grid)
    scale_grid = parse_grid(args.scale_grid)

    rows = []
    for rerisk, hold, cap, derisk, scale in itertools.product(rerisk_grid, hold_grid, cap_grid, derisk_grid, scale_grid):
        cand_profile = dict(base_profile)
        cand_profile["rerisk"] = float(rerisk)
        cand_profile["hold"] = float(hold)
        cand_profile["cap"] = float(cap)
        cand_profile["derisk"] = float(derisk)
        cand_profile["profile"] = "v7_candidate"

        cand_overrides = {k: float(v * scale) for k, v in base_overrides.items()}
        metrics = evaluate_candidate(cand_profile, cand_overrides, base_profile, base_overrides, folds, cost_bps=float(args.cost_bps))
        rows.append(
            {
                "rerisk": rerisk,
                "hold": hold,
                "cap": cap,
                "derisk": derisk,
                "override_scale": scale,
                **metrics,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["median_delta_terminal", "pass_rate_terminal", "median_delta_maxdd", "full_delta_terminal"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    passing = df[
        (df["pass_rate_terminal"] >= float(args.min_pass_rate))
        & (df["median_delta_maxdd"] >= float(args.dd_tol))
        & (df["median_delta_terminal"] > 0)
    ]

    decision = "KEEP_V6_TUNED"
    selected = None
    out_prefix = ROOT / args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    out_csv = Path(str(out_prefix) + "_scan.csv")
    out_md = Path(str(out_prefix) + "_scan.md")
    df.to_csv(out_csv, index=False)

    if not passing.empty:
        selected = passing.iloc[0]
        decision = "PROMOTE_V7"
        v7_profile = dict(base_profile)
        v7_profile["profile"] = "v7_tuned"
        v7_profile["rerisk"] = float(selected["rerisk"])
        v7_profile["hold"] = float(selected["hold"])
        v7_profile["cap"] = float(selected["cap"])
        v7_profile["derisk"] = float(selected["derisk"])

        v7_overrides = {k: float(v * float(selected["override_scale"])) for k, v in base_overrides.items()}

        Path(str(out_prefix) + "_promote_profile.json").write_text(json.dumps(v7_profile, indent=2) + "\n", encoding="utf-8")
        pd.DataFrame([{"state": k, "override_exposure": v} for k, v in sorted(v7_overrides.items())]).to_csv(
            Path(str(out_prefix) + "_promote_overrides.csv"), index=False
        )

    lines = [
        "# V7 Promotion Workbench",
        "",
        f"- base_profile: {base_profile.get('profile', 'v6_tuned')}",
        f"- folds: {len(folds)}",
        f"- cost_bps: {args.cost_bps}",
        f"- min_pass_rate: {args.min_pass_rate}",
        f"- dd_tol (median delta maxdd): {args.dd_tol}",
        "",
        "## Top 20",
        df.head(20).to_string(index=False),
        "",
        "## Decision",
        f"- {decision}",
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
    if selected is not None:
        print("selected", selected.to_dict())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
