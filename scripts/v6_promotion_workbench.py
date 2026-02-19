from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.regime_portfolio_optimizer import (
        load_regime_frames,
        prepare_ticker_data,
        simulate_returns,
        perf_from_returns,
    )
except ModuleNotFoundError:
    from regime_portfolio_optimizer import (
        load_regime_frames,
        prepare_ticker_data,
        simulate_returns,
        perf_from_returns,
    )

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


def build_folds(td: dict[str, pd.DataFrame], n_folds: int = 3) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    dates = pd.Index(sorted(pd.to_datetime(pd.concat([df["date"] for df in td.values()], ignore_index=True).dropna().unique())))
    n = len(dates)
    if n < 300:
        return []

    train_frac = 0.75
    val_frac = 0.12
    step_frac = 0.10

    train_n = max(180, int(n * train_frac))
    val_n = max(60, int(n * val_frac))
    step_n = max(30, int(n * step_frac))

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


def evaluate_candidate(
    v6_profile: dict,
    v6_overrides: dict[str, float],
    v5_profile: dict,
    v5_overrides: dict[str, float],
    folds: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]],
) -> dict:
    v6_td = load_td(v6_profile, v6_overrides)
    v5_td = load_td(v5_profile, v5_overrides)

    deltas_tw = []
    deltas_sh = []
    deltas_dd = []

    for _, train_end, val_end in folds:
        v6_ret = simulate_returns(v6_td, v6_overrides, float(v6_profile["hold"]), start=train_end, end=val_end)
        v5_ret = simulate_returns(v5_td, v5_overrides, float(v5_profile["hold"]), start=train_end, end=val_end)
        p6 = perf_from_returns(v6_ret)
        p5 = perf_from_returns(v5_ret)
        deltas_tw.append(float(p6.terminal_wealth - p5.terminal_wealth))
        deltas_sh.append(float(p6.sharpe - p5.sharpe))
        deltas_dd.append(float(p6.max_drawdown - p5.max_drawdown))

    # full-sample check
    v6_full = perf_from_returns(simulate_returns(v6_td, v6_overrides, float(v6_profile["hold"])))
    v5_full = perf_from_returns(simulate_returns(v5_td, v5_overrides, float(v5_profile["hold"])))

    return {
        "median_delta_terminal": float(pd.Series(deltas_tw).median()),
        "median_delta_sharpe": float(pd.Series(deltas_sh).median()),
        "median_delta_maxdd": float(pd.Series(deltas_dd).median()),
        "pass_rate_terminal": float((pd.Series(deltas_tw) > 0).mean()),
        "full_delta_terminal": float(v6_full.terminal_wealth - v5_full.terminal_wealth),
        "full_v6_terminal": float(v6_full.terminal_wealth),
        "full_v6_sharpe": float(v6_full.sharpe),
        "full_v6_maxdd": float(v6_full.max_drawdown),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V6 promotion workbench (OOT vs v5)")
    parser.add_argument("--v6-profile", default="reports/v6_native_profile.json")
    parser.add_argument("--v6-overrides", default="reports/v6_native_overrides.csv")
    parser.add_argument("--v5-profile", default="reports/v5_aggressive_profile.json")
    parser.add_argument("--v5-overrides", default="reports/v5_aggressive_overrides.csv")
    parser.add_argument("--hold-grid", default="0.25,0.3,0.35")
    parser.add_argument("--rerisk-grid", default="")
    parser.add_argument("--cap-grid", default="")
    parser.add_argument("--derisk-grid", default="0.03,0.05,0.08")
    parser.add_argument("--scale-grid", default="0.7,0.85,1.0,1.15")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--dd-tol", type=float, default=-0.02, help="Minimum median delta maxdd (v6-v5)")
    parser.add_argument("--min-pass-rate", type=float, default=0.67)
    parser.add_argument("--out-prefix", default="reports/v6_promotion_workbench")
    args = parser.parse_args()

    v6_raw = json.loads((ROOT / args.v6_profile).read_text(encoding="utf-8"))
    v6_base = normalize_profile(v6_raw)
    v6_ovr_df = pd.read_csv(ROOT / args.v6_overrides)
    v6_base_ovr = {str(r.state): float(r.override_exposure) for r in v6_ovr_df.itertuples(index=False)}

    v5_raw = json.loads((ROOT / args.v5_profile).read_text(encoding="utf-8"))
    v5 = normalize_profile(v5_raw, mode="aggressive")
    v5_ovr_df = pd.read_csv(ROOT / args.v5_overrides)
    v5_ovr = {str(r.state): float(r.override_exposure) for r in v5_ovr_df.itertuples(index=False)}

    v6_td_for_folds = load_td(v6_base, v6_base_ovr)
    folds = build_folds(v6_td_for_folds, n_folds=int(args.folds))
    if not folds:
        raise RuntimeError("Could not construct OOT folds")

    hold_grid = parse_grid(args.hold_grid)
    rerisk_grid = parse_grid(args.rerisk_grid) if str(args.rerisk_grid).strip() else [float(v6_base["rerisk"])]
    cap_grid = parse_grid(args.cap_grid) if str(args.cap_grid).strip() else [float(v6_base["cap"])]
    derisk_grid = parse_grid(args.derisk_grid)
    scale_grid = parse_grid(args.scale_grid)

    rows = []
    for rerisk, hold, cap, derisk, scale in itertools.product(rerisk_grid, hold_grid, cap_grid, derisk_grid, scale_grid):
        v6_profile = dict(v6_base)
        v6_profile["rerisk"] = float(rerisk)
        v6_profile["hold"] = float(hold)
        v6_profile["cap"] = float(cap)
        v6_profile["derisk"] = float(derisk)

        v6_ovr = {k: float(v * scale) for k, v in v6_base_ovr.items()}
        metrics = evaluate_candidate(v6_profile, v6_ovr, v5, v5_ovr, folds)
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

    out_prefix = ROOT / args.out_prefix
    out_csv = Path(str(out_prefix) + "_scan.csv")
    out_md = Path(str(out_prefix) + "_scan.md")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    passing = df[(df["pass_rate_terminal"] >= float(args.min_pass_rate)) & (df["median_delta_maxdd"] >= float(args.dd_tol))]
    decision = "KEEP_V5"
    selected = None
    if not passing.empty and float(passing.iloc[0]["median_delta_terminal"]) > 0:
        selected = passing.iloc[0]
        decision = "PROMOTE_V6_TUNED"

        v6_promote = dict(v6_base)
        v6_promote["hold"] = float(selected["hold"])
        v6_promote["derisk"] = float(selected["derisk"])
        v6_promote["profile"] = "v6_native_tuned"
        v6_promote["state_schema"] = "auto"

        promote_ovr = {k: float(v * float(selected["override_scale"])) for k, v in v6_base_ovr.items()}

        Path(str(out_prefix) + "_promote_profile.json").write_text(json.dumps(v6_promote, indent=2) + "\n", encoding="utf-8")
        pd.DataFrame([{"state": k, "override_exposure": v} for k, v in sorted(promote_ovr.items())]).to_csv(
            Path(str(out_prefix) + "_promote_overrides.csv"), index=False
        )

    lines = [
        "# V6 Promotion Workbench",
        "",
        f"- folds: {len(folds)}",
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
        lines.extend(
            [
                "",
                "## Selected",
                selected.to_string(),
            ]
        )

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
