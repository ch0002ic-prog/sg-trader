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


def parse_windows(text: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for part in str(text).split(","):
        token = part.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid window token: {token}")
        left, right = token.split(":", 1)
        start = pd.Timestamp(left.strip())
        end = pd.Timestamp(right.strip())
        if end <= start:
            raise ValueError(f"Window end must be after start: {token}")
        windows.append((start, end))
    if not windows:
        raise ValueError("No stress windows provided")
    return windows


def build_folds(td: dict[str, pd.DataFrame], n_folds: int) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    dates = pd.Index(sorted(pd.to_datetime(pd.concat([df["date"] for df in td.values()], ignore_index=True).dropna().unique())))
    n = len(dates)
    if n < 500:
        return []

    train_frac = 0.64
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


def evaluate_candidate(
    cand_profile: dict,
    cand_overrides: dict[str, float],
    base_profile: dict,
    base_overrides: dict[str, float],
    folds: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]],
    cost_grid: list[float],
    persistence_scenarios: list[int],
    stress_windows: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> dict:
    scenario_rows = []
    worst_fold_maxdd = np.inf

    for persistence in persistence_scenarios:
        cand_td = load_td(cand_profile, cand_overrides, persistence_days=persistence)
        base_td = load_td(base_profile, base_overrides, persistence_days=persistence)

        for cost_bps in cost_grid:
            deltas_tw = []
            deltas_sh = []
            deltas_dd = []
            fold_maxdds = []

            for _, train_end, val_end in folds:
                cand_ret = simulate_returns_with_cost(cand_td, cand_overrides, float(cand_profile["hold"]), cost_bps, start=train_end, end=val_end)
                base_ret = simulate_returns_with_cost(base_td, base_overrides, float(base_profile["hold"]), cost_bps, start=train_end, end=val_end)
                pc = perf_from_returns(cand_ret)
                pb = perf_from_returns(base_ret)
                deltas_tw.append(float(pc.terminal_wealth - pb.terminal_wealth))
                deltas_sh.append(float(pc.sharpe - pb.sharpe))
                deltas_dd.append(float(pc.max_drawdown - pb.max_drawdown))
                fold_maxdds.append(float(pc.max_drawdown))

            worst_fold_maxdd = min(worst_fold_maxdd, float(pd.Series(fold_maxdds).min()))
            scenario_rows.append(
                {
                    "persistence_days": persistence,
                    "cost_bps": float(cost_bps),
                    "median_delta_terminal": float(pd.Series(deltas_tw).median()),
                    "median_delta_sharpe": float(pd.Series(deltas_sh).median()),
                    "median_delta_maxdd": float(pd.Series(deltas_dd).median()),
                    "pass_rate_terminal": float((pd.Series(deltas_tw) > 0).mean()),
                }
            )

    scen = pd.DataFrame(scenario_rows)

    cand_full_td = load_td(cand_profile, cand_overrides, persistence_days=int(cand_profile.get("persistence_days", 1)))
    base_full_td = load_td(base_profile, base_overrides, persistence_days=int(base_profile.get("persistence_days", 1)))
    low_cost = min(cost_grid)
    cand_full = perf_from_returns(simulate_returns_with_cost(cand_full_td, cand_overrides, float(cand_profile["hold"]), low_cost))
    base_full = perf_from_returns(simulate_returns_with_cost(base_full_td, base_overrides, float(base_profile["hold"]), low_cost))

    stress_dds = []
    for ws, we in stress_windows:
        cand_stress = perf_from_returns(
            simulate_returns_with_cost(cand_full_td, cand_overrides, float(cand_profile["hold"]), low_cost, start=ws, end=we)
        )
        stress_dds.append(float(cand_stress.max_drawdown))
    worst_stress_maxdd = float(pd.Series(stress_dds).min()) if stress_dds else np.nan

    return {
        "min_median_delta_terminal": float(scen["median_delta_terminal"].min()),
        "min_pass_rate_terminal": float(scen["pass_rate_terminal"].min()),
        "worst_median_delta_maxdd": float(scen["median_delta_maxdd"].min()),
        "worst_fold_maxdd": float(worst_fold_maxdd),
        "avg_median_delta_terminal": float(scen["median_delta_terminal"].mean()),
        "avg_median_delta_sharpe": float(scen["median_delta_sharpe"].mean()),
        "worst_stress_maxdd": worst_stress_maxdd,
        "full_delta_terminal": float(cand_full.terminal_wealth - base_full.terminal_wealth),
        "full_candidate_terminal": float(cand_full.terminal_wealth),
        "full_candidate_sharpe": float(cand_full.sharpe),
        "full_candidate_maxdd": float(cand_full.max_drawdown),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V11 promotion workbench (hard fold + full DD envelope vs v10)")
    parser.add_argument("--base-profile", default="reports/v10_tuned_profile.json")
    parser.add_argument("--base-overrides", default="reports/v10_tuned_overrides.csv")
    parser.add_argument("--rerisk-grid", default="0.02,0.04,0.06")
    parser.add_argument("--hold-grid", default="0.40,0.42,0.44")
    parser.add_argument("--cap-grid", default="0.03")
    parser.add_argument("--derisk-grid", default="0.16,0.18,0.2")
    parser.add_argument("--scale-grid", default="1.05,1.1,1.15,1.2")
    parser.add_argument("--override-clip", type=float, default=0.45)
    parser.add_argument("--cost-grid", default="2,4,6,8")
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--persistence-shifts", default="-1,0,1")
    parser.add_argument("--min-pass-rate", type=float, default=0.67)
    parser.add_argument("--dd-tol", type=float, default=-0.01)
    parser.add_argument("--maxdd-cap", type=float, default=-0.22, help="Hard cap: worst fold maxdd must be >= this value")
    parser.add_argument("--full-maxdd-cap", type=float, default=-0.26, help="Hard cap: full candidate maxdd must be >= this value")
    parser.add_argument(
        "--stress-windows",
        default="2007-10-01:2009-09-30,2020-02-01:2020-06-30",
        help="Comma-separated stress windows start:end",
    )
    parser.add_argument("--stress-maxdd-cap", type=float, default=-0.26, help="Hard cap: worst stress-window maxdd")
    parser.add_argument("--min-avg-sharpe-delta", type=float, default=0.0)
    parser.add_argument("--out-prefix", default="reports/v11_promotion_workbench")
    args = parser.parse_args()

    base_profile = normalize_profile(json.loads((ROOT / args.base_profile).read_text(encoding="utf-8")))
    base_ovr_df = pd.read_csv(ROOT / args.base_overrides)
    base_overrides_raw = {str(r.state): float(r.override_exposure) for r in base_ovr_df.itertuples(index=False)}

    base_td = load_td(base_profile, base_overrides_raw, persistence_days=int(base_profile.get("persistence_days", 1)))
    folds = build_folds(base_td, n_folds=int(args.folds))
    if not folds:
        raise RuntimeError("Could not construct OOT folds for v11 workbench")

    rerisk_grid = parse_grid(args.rerisk_grid)
    hold_grid = parse_grid(args.hold_grid)
    cap_grid = parse_grid(args.cap_grid)
    derisk_grid = parse_grid(args.derisk_grid)
    scale_grid = parse_grid(args.scale_grid)
    cost_grid = parse_grid(args.cost_grid)
    shift_grid = parse_grid(args.persistence_shifts, cast=int)

    base_p = int(base_profile.get("persistence_days", 1))
    persistence_scenarios = sorted({max(1, base_p + int(s)) for s in shift_grid})
    stress_windows = parse_windows(args.stress_windows)

    rows = []
    for rerisk, hold, cap, derisk, scale in itertools.product(rerisk_grid, hold_grid, cap_grid, derisk_grid, scale_grid):
        cand_profile = dict(base_profile)
        cand_profile["profile"] = "v11_candidate"
        cand_profile["rerisk"] = float(rerisk)
        cand_profile["hold"] = float(hold)
        cand_profile["cap"] = float(cap)
        cand_profile["derisk"] = float(derisk)

        cand_overrides = {
            k: float(np.clip(v * float(scale), -float(args.override_clip), float(args.override_clip)))
            for k, v in base_overrides_raw.items()
        }

        metrics = evaluate_candidate(
            cand_profile,
            cand_overrides,
            base_profile,
            base_overrides_raw,
            folds,
            cost_grid,
            persistence_scenarios,
            stress_windows,
        )
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
        [
            "min_median_delta_terminal",
            "min_pass_rate_terminal",
            "worst_median_delta_maxdd",
            "worst_fold_maxdd",
            "worst_stress_maxdd",
            "avg_median_delta_sharpe",
            "full_delta_terminal",
        ],
        ascending=[False, False, False, False, False, False, False],
    ).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    passing = df[
        (df["min_median_delta_terminal"] > 0)
        & (df["min_pass_rate_terminal"] >= float(args.min_pass_rate))
        & (df["worst_median_delta_maxdd"] >= float(args.dd_tol))
        & (df["worst_fold_maxdd"] >= float(args.maxdd_cap))
        & (df["worst_stress_maxdd"] >= float(args.stress_maxdd_cap))
        & (df["full_candidate_maxdd"] >= float(args.full_maxdd_cap))
        & (df["avg_median_delta_sharpe"] >= float(args.min_avg_sharpe_delta))
    ]

    decision = "KEEP_V10_TUNED"
    selected = None

    out_prefix = ROOT / args.out_prefix
    out_csv = Path(str(out_prefix) + "_scan.csv")
    out_md = Path(str(out_prefix) + "_scan.md")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    if not passing.empty:
        selected = passing.iloc[0]
        decision = "PROMOTE_V11"

        v11_profile = dict(base_profile)
        v11_profile["profile"] = "v11_tuned"
        v11_profile["rerisk"] = float(selected["rerisk"])
        v11_profile["hold"] = float(selected["hold"])
        v11_profile["cap"] = float(selected["cap"])
        v11_profile["derisk"] = float(selected["derisk"])

        v11_overrides = {
            k: float(np.clip(v * float(selected["override_scale"]), -float(args.override_clip), float(args.override_clip)))
            for k, v in base_overrides_raw.items()
        }

        Path(str(out_prefix) + "_promote_profile.json").write_text(json.dumps(v11_profile, indent=2) + "\n", encoding="utf-8")
        pd.DataFrame([{"state": k, "override_exposure": v} for k, v in sorted(v11_overrides.items())]).to_csv(
            Path(str(out_prefix) + "_promote_overrides.csv"), index=False
        )

    lines = [
        "# V11 Promotion Workbench",
        "",
        f"- base_profile: {base_profile.get('profile', 'v10_tuned')}",
        f"- folds: {len(folds)}",
        f"- cost_grid_bps: {cost_grid}",
        f"- persistence_scenarios: {persistence_scenarios}",
        f"- min_pass_rate: {args.min_pass_rate}",
        f"- dd_tol: {args.dd_tol}",
        f"- maxdd_cap: {args.maxdd_cap}",
        f"- full_maxdd_cap: {args.full_maxdd_cap}",
        f"- stress_windows: {args.stress_windows}",
        f"- stress_maxdd_cap: {args.stress_maxdd_cap}",
        f"- min_avg_sharpe_delta: {args.min_avg_sharpe_delta}",
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
