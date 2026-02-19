from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
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
    from scripts.regime_portfolio_optimizer_v2 import perf_from_sim, simulate_returns as simulate_returns_v2
    from scripts.regime_optimizer_v6_state_aug_scan import load_augmented_ticker_data
except ModuleNotFoundError:
    from regime_portfolio_optimizer import (
        load_regime_frames,
        perf_from_returns,
        prepare_ticker_data,
        simulate_returns,
    )
    from regime_portfolio_optimizer_v2 import perf_from_sim, simulate_returns as simulate_returns_v2
    from regime_optimizer_v6_state_aug_scan import load_augmented_ticker_data

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def parse_grid(text: str) -> list[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty grid")
    return vals


def base_key(state: str) -> str:
    return "::".join(str(state).split("::")[:2])


def objective(terminal_wealth: float, maxdd: float, min_maxdd: float, dd_penalty: float) -> float:
    base = float(np.log(max(float(terminal_wealth), 1e-300)))
    dd_shortfall = max(0.0, float(min_maxdd - maxdd))
    return base - dd_penalty * dd_shortfall


def summarize_perf(tag: str, mode: str, perf) -> dict:
    return {
        "tag": tag,
        "mode": mode,
        "terminal_wealth": float(perf.terminal_wealth),
        "sharpe": float(perf.sharpe),
        "maxdd": float(perf.max_drawdown),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve v6 augmented override schema into deployable standard overrides")
    parser.add_argument("--scan-csv", default="reports/regime_optimizer_v6_state_aug_local_scan.csv")
    parser.add_argument("--scan-overrides", default="reports/regime_optimizer_v6_state_aug_local_best_overrides.csv")
    parser.add_argument("--candidate-grid", default="-0.2,-0.15,-0.1,-0.05,0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4")
    parser.add_argument("--min-maxdd", type=float, default=-0.12)
    parser.add_argument("--dd-penalty", type=float, default=80.0)
    parser.add_argument("--dd-guard", type=float, default=None, help="Optional max drawdown floor for wealth-first winner")
    parser.add_argument("--out-prefix", default="reports/v6_resolved")
    args = parser.parse_args()

    scan = pd.read_csv(ROOT / args.scan_csv)
    if scan.empty:
        raise ValueError("Scan CSV is empty")
    best = scan.iloc[0]

    label = str(best["label"])
    rerisk = float(best["rerisk"])
    hold = float(best["hold"])
    cap = float(best["cap"])
    derisk = float(best["derisk"])
    persistence_days = int(best["persistence_days"])

    aug_overrides_df = pd.read_csv(ROOT / args.scan_overrides)
    aug_overrides = {str(r.state): float(r.override_exposure) for r in aug_overrides_df.itertuples(index=False)}

    aug_td = load_augmented_ticker_data(
        label,
        exclude_tickers={"^VIX", "^VVIX"},
        rerisk=rerisk,
        hold=hold,
        cap=cap,
        derisk=derisk,
        persistence_days=persistence_days,
    )

    state_counts = defaultdict(int)
    for df in aug_td.values():
        vc = df["state"].dropna().astype(str).value_counts()
        for s, c in vc.items():
            state_counts[s] += int(c)

    grouped = defaultdict(list)
    for s, v in aug_overrides.items():
        grouped[base_key(s)].append((float(v), int(state_counts.get(s, 1)), s))

    bridge_mean = {}
    bridge_wmean = {}
    bridge_median = {}
    for b, items in grouped.items():
        vals = [x[0] for x in items]
        ws = [x[1] for x in items]
        bridge_mean[b] = float(sum(vals) / len(vals))
        bridge_wmean[b] = float(sum(v * w for v, w, _ in items) / max(1, sum(ws)))
        bridge_median[b] = float(sorted(vals)[len(vals) // 2])

    frames = load_regime_frames(label)
    std_td = prepare_ticker_data(
        frames,
        rerisk=rerisk,
        hold=hold,
        cap=cap,
        derisk=derisk,
        exclude_tickers={"^VIX", "^VVIX"},
    )

    std_states = set()
    for df in std_td.values():
        std_states.update(df["state"].dropna().astype(str).tolist())

    aug_states = set()
    for df in aug_td.values():
        aug_states.update(df["state"].dropna().astype(str).tolist())

    # Baselines
    std_base_perf = perf_from_returns(simulate_returns(std_td, {}, hold))
    aug_base_perf = perf_from_sim(simulate_returns_v2(aug_td, {}, hold, None, None))
    aug_opt_perf = perf_from_sim(simulate_returns_v2(aug_td, aug_overrides, hold, None, None))

    # Bridge evaluations
    bridge_rows = []
    bridge_maps = {
        "mean": bridge_mean,
        "wmean": bridge_wmean,
        "median": bridge_median,
    }

    for name, bmap in bridge_maps.items():
        p = perf_from_returns(simulate_returns(std_td, bmap, hold))
        bridge_rows.append(
            {
                "bridge_method": name,
                "states": len(bmap),
                "terminal_wealth": float(p.terminal_wealth),
                "sharpe": float(p.sharpe),
                "maxdd": float(p.max_drawdown),
                "objective": objective(p.terminal_wealth, p.max_drawdown, args.min_maxdd, args.dd_penalty),
            }
        )

    bridge_df = pd.DataFrame(bridge_rows).sort_values(["objective", "terminal_wealth"], ascending=[False, False]).reset_index(drop=True)

    # Compare vs active v5 if available
    active_comp = None
    active_label_path = REPORTS / "active_profile_label.txt"
    active_exp_path = REPORTS / "active_profile_exposures.json"
    active_ovr_path = REPORTS / "active_policy_overrides.csv"
    if active_label_path.exists() and active_exp_path.exists() and active_ovr_path.exists():
        active_label = active_label_path.read_text(encoding="utf-8").strip()
        active_exp = json.loads(active_exp_path.read_text(encoding="utf-8"))
        active_ovr_df = pd.read_csv(active_ovr_path)
        active_ovr = {str(r.state): float(r.override_exposure) for r in active_ovr_df.itertuples(index=False)}
        active_frames = load_regime_frames(active_label)
        active_td = prepare_ticker_data(
            active_frames,
            rerisk=float(active_exp["rerisk"]),
            hold=float(active_exp["hold"]),
            cap=float(active_exp["cap"]),
            derisk=float(active_exp["derisk"]),
            exclude_tickers={"^VIX", "^VVIX"},
        )
        active_perf = perf_from_returns(simulate_returns(active_td, active_ovr, float(active_exp["hold"])))
        active_comp = {
            "label": active_label,
            "terminal_wealth": float(active_perf.terminal_wealth),
            "sharpe": float(active_perf.sharpe),
            "maxdd": float(active_perf.max_drawdown),
        }

    # Exhaustive refit on deployable key set
    candidate_grid = parse_grid(args.candidate_grid)
    deploy_states = sorted(bridge_wmean.keys())
    dd_guard = args.dd_guard
    if dd_guard is None and active_comp is not None:
        dd_guard = float(active_comp["maxdd"])

    best_map = dict(bridge_wmean)
    best_perf = perf_from_returns(simulate_returns(std_td, best_map, hold))
    best_score = objective(best_perf.terminal_wealth, best_perf.max_drawdown, args.min_maxdd, args.dd_penalty)

    best_wealth_map = dict(best_map)
    best_wealth_perf = best_perf

    best_guard_map = dict(best_map)
    best_guard_perf = None
    if dd_guard is not None and float(best_perf.max_drawdown) >= float(dd_guard):
        best_guard_perf = best_perf

    for values in itertools.product(candidate_grid, repeat=len(deploy_states)):
        trial = {k: float(v) for k, v in zip(deploy_states, values)}
        p = perf_from_returns(simulate_returns(std_td, trial, hold))
        score = objective(p.terminal_wealth, p.max_drawdown, args.min_maxdd, args.dd_penalty)
        if score > best_score + 1e-12:
            best_score = score
            best_map = trial
            best_perf = p
        if float(p.terminal_wealth) > float(best_wealth_perf.terminal_wealth) + 1e-12:
            best_wealth_map = trial
            best_wealth_perf = p
        if dd_guard is not None and float(p.max_drawdown) >= float(dd_guard):
            if best_guard_perf is None or float(p.terminal_wealth) > float(best_guard_perf.terminal_wealth) + 1e-12:
                best_guard_map = trial
                best_guard_perf = p

    # Persist artifacts
    out_prefix = ROOT / args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    coverage = {
        "override_states_aug": len(aug_overrides),
        "std_unique_states": len(std_states),
        "aug_unique_states": len(aug_states),
        "override_intersection_std": len(set(aug_overrides) & std_states),
        "override_intersection_aug": len(set(aug_overrides) & aug_states),
        "deploy_bridge_states": len(deploy_states),
    }

    pd.DataFrame(
        [
            summarize_perf("std", "base", std_base_perf),
            summarize_perf("std", "resolved_objective_best", best_perf),
            summarize_perf("std", "resolved_wealth_best", best_wealth_perf),
            summarize_perf("aug", "base", aug_base_perf),
            summarize_perf("aug", "opt", aug_opt_perf),
        ]
    ).to_csv(Path(str(out_prefix) + "_perf_summary.csv"), index=False)

    bridge_df.to_csv(Path(str(out_prefix) + "_bridge_methods.csv"), index=False)
    pd.DataFrame([{"state": k, "override_exposure": v} for k, v in sorted(best_map.items())]).to_csv(
        Path(str(out_prefix) + "_overrides_objective.csv"), index=False
    )
    pd.DataFrame([{"state": k, "override_exposure": v} for k, v in sorted(best_wealth_map.items())]).to_csv(
        Path(str(out_prefix) + "_overrides_wealth.csv"), index=False
    )
    if best_guard_perf is not None:
        pd.DataFrame([{"state": k, "override_exposure": v} for k, v in sorted(best_guard_map.items())]).to_csv(
            Path(str(out_prefix) + "_overrides_wealth_guard.csv"), index=False
        )

    decision = "KEEP_ACTIVE"
    decision_metric = best_guard_perf if best_guard_perf is not None else best_wealth_perf
    if active_comp is not None and decision_metric.terminal_wealth > active_comp["terminal_wealth"]:
        decision = "PROMOTE_RESOLVED_V6"

    lines = [
        "# V6 Schema Resolution Report",
        "",
        f"- label: {label}",
        f"- exposures: rerisk={rerisk}, hold={hold}, cap={cap}, derisk={derisk}, persistence_days={persistence_days}",
        f"- coverage: {coverage}",
        "",
        "## Bridge method results (standard pipeline)",
        bridge_df.to_string(index=False),
        "",
        "## Best resolved deployable overrides (standard keys)",
        "### Objective-best",
        pd.DataFrame([{"state": k, "override_exposure": v} for k, v in sorted(best_map.items())]).to_string(index=False),
        "",
        "### Wealth-best",
        pd.DataFrame([{"state": k, "override_exposure": v} for k, v in sorted(best_wealth_map.items())]).to_string(index=False),
        "",
        "## Performance summary",
        pd.read_csv(Path(str(out_prefix) + "_perf_summary.csv")).to_string(index=False),
    ]

    if active_comp is not None:
        lines.extend(
            [
                "",
                "## Active comparison",
                f"- active_label: {active_comp['label']}",
                f"- active terminal/sharpe/maxdd: {active_comp['terminal_wealth']:.6f} / {active_comp['sharpe']:.6f} / {active_comp['maxdd']:.6f}",
                f"- resolved_objective terminal/sharpe/maxdd: {best_perf.terminal_wealth:.6f} / {best_perf.sharpe:.6f} / {best_perf.max_drawdown:.6f}",
                f"- resolved_wealth terminal/sharpe/maxdd: {best_wealth_perf.terminal_wealth:.6f} / {best_wealth_perf.sharpe:.6f} / {best_wealth_perf.max_drawdown:.6f}",
            ]
        )
        if best_guard_perf is not None:
            lines.extend(
                [
                    f"- dd_guard used: {float(dd_guard):.6f}",
                    f"- resolved_wealth_guard terminal/sharpe/maxdd: {best_guard_perf.terminal_wealth:.6f} / {best_guard_perf.sharpe:.6f} / {best_guard_perf.max_drawdown:.6f}",
                    f"- terminal delta (guard - active): {best_guard_perf.terminal_wealth - active_comp['terminal_wealth']:.6f}",
                ]
            )
        else:
            lines.append("- no candidate satisfied dd_guard")

    lines.extend(["", "## Decision", f"- {decision}"])

    Path(str(out_prefix) + "_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("saved", Path(str(out_prefix) + "_perf_summary.csv"))
    print("saved", Path(str(out_prefix) + "_bridge_methods.csv"))
    print("saved", Path(str(out_prefix) + "_overrides_objective.csv"))
    print("saved", Path(str(out_prefix) + "_overrides_wealth.csv"))
    if best_guard_perf is not None:
        print("saved", Path(str(out_prefix) + "_overrides_wealth_guard.csv"))
    print("saved", Path(str(out_prefix) + "_report.md"))
    print("coverage", coverage)
    print("resolved_objective_terminal", float(best_perf.terminal_wealth))
    print("resolved_wealth_terminal", float(best_wealth_perf.terminal_wealth))
    if best_guard_perf is not None:
        print("resolved_guard_terminal", float(best_guard_perf.terminal_wealth))
    print("decision", decision)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
