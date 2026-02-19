#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _parse_mapping(items: list[str], *, label: str) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Invalid {label} mapping '{item}'. Expected key=path.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"Invalid {label} mapping '{item}'. Empty key.")
        mapping[key] = Path(value)
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build monthly strategy decision summary from walk-forward and allocation outputs.",
    )
    parser.add_argument(
        "--scan-csv",
        action="append",
        required=True,
        help="Lookback to CSV mapping (for example: 63=/tmp/walkforward_lb63.csv).",
    )
    parser.add_argument(
        "--allocation",
        action="append",
        required=True,
        help="Profile to allocation JSON mapping (for example: aggressive=/tmp/agg.json).",
    )
    parser.add_argument(
        "--switch-primary-lookbacks",
        default="63,126",
        help="Comma-separated lookbacks used for switch rule evaluation.",
    )
    parser.add_argument("--switch-candidate", default="defensive")
    parser.add_argument("--current-default", default="aggressive")
    parser.add_argument("--top3-threshold", type=float, default=0.70)
    parser.add_argument("--effective-n-threshold", type=float, default=4.50)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    return parser.parse_args()


def _read_scan_rows(scan_paths: dict[str, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lookback_str, path in scan_paths.items():
        lookback = int(lookback_str)
        with path.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "lookback": lookback,
                        "profile": str(row["profile"]),
                        "avg": float(row["avg_forward_return"]),
                        "median": float(row["median_forward_return"]),
                        "win_rate": float(row["win_rate"]),
                    }
                )
    return rows


def _rank_scan_rows(scan_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    ranking_rows: list[dict[str, Any]] = []
    stability_points: dict[str, int] = {}
    lookbacks = sorted({int(item["lookback"]) for item in scan_rows})
    for lookback in lookbacks:
        subset = sorted(
            [item for item in scan_rows if int(item["lookback"]) == lookback],
            key=lambda item: float(item["avg"]),
            reverse=True,
        )
        for rank, item in enumerate(subset, start=1):
            pts = {1: 3, 2: 2, 3: 1}.get(rank, 0)
            profile = str(item["profile"])
            stability_points[profile] = stability_points.get(profile, 0) + pts
            ranking_rows.append(
                {
                    "lookback": lookback,
                    "rank": rank,
                    "profile": profile,
                    "avg_forward": float(item["avg"]),
                    "median_forward": float(item["median"]),
                    "win_rate": float(item["win_rate"]),
                }
            )
    return ranking_rows, stability_points


def _concentration_rows(
    allocation_paths: dict[str, Path],
    *,
    top3_threshold: float,
    effective_n_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile in sorted(allocation_paths.keys()):
        payload = json.loads(allocation_paths[profile].read_text(encoding="utf-8"))
        weights = [
            max(float(item.get("weight", 0.0)), 0.0)
            for item in payload.get("allocations", [])
        ]
        weights_sorted = sorted(weights, reverse=True)
        top3 = float(sum(weights_sorted[:3])) if weights_sorted else 0.0
        denom = float(sum(value * value for value in weights_sorted))
        effective_n = float(1.0 / denom) if denom > 0 else 0.0
        breach = bool(top3 > top3_threshold or effective_n < effective_n_threshold)
        rows.append(
            {
                "profile": profile,
                "selected_count": len(weights_sorted),
                "top3_concentration": top3,
                "effective_n": effective_n,
                "guardrail": "BREACH" if breach else "OK",
            }
        )
    return rows


def _switch_rule_met(
    ranking_rows: list[dict[str, Any]],
    *,
    switch_candidate: str,
    primary_lookbacks: set[int],
) -> bool:
    by_lookback: dict[int, str] = {}
    for row in ranking_rows:
        lookback = int(row["lookback"])
        if int(row["rank"]) == 1:
            by_lookback[lookback] = str(row["profile"])
    return all(by_lookback.get(lb) == switch_candidate for lb in primary_lookbacks)


def _render_markdown(
    *,
    ranking_rows: list[dict[str, Any]],
    stability_points: dict[str, int],
    concentration: list[dict[str, Any]],
    decision: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# Monthly Strategy Decision Pack")
    lines.append("")
    lines.append("## Walk-forward ranking")
    lines.append("")
    lines.append("lookback | rank | profile | avg_forward | median_forward | win_rate")
    lines.append("--- | --- | --- | --- | --- | ---")
    for row in sorted(ranking_rows, key=lambda item: (item["lookback"], item["rank"])):
        lines.append(
            f"{row['lookback']} | {row['rank']} | {row['profile']} | "
            f"{row['avg_forward']:.6f} | {row['median_forward']:.6f} | {row['win_rate']:.2%}"
        )
    lines.append("")
    lines.append("Stability points (3/2/1 by avg rank):")
    for profile, points in sorted(stability_points.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- {profile}: {points}")
    lines.append("")
    lines.append("## Concentration guardrails")
    lines.append("")
    lines.append("profile | selected_count | top3_concentration | effective_n | guardrail")
    lines.append("--- | --- | --- | --- | ---")
    for row in concentration:
        lines.append(
            f"{row['profile']} | {row['selected_count']} | "
            f"{row['top3_concentration']:.4f} | {row['effective_n']:.2f} | {row['guardrail']}"
        )
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- Keep default profile: {decision['keep_default']}")
    lines.append(f"- Suggested switch candidate: {decision['switch_candidate']}")
    lines.append(f"- Switch rule met this cycle: {decision['switch_rule_met']}")
    lines.append(f"- Any concentration guardrail breach: {decision['any_guardrail_breach']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    scan_paths = _parse_mapping(args.scan_csv, label="scan")
    allocation_paths = _parse_mapping(args.allocation, label="allocation")
    primary_lookbacks = {
        int(item.strip())
        for item in args.switch_primary_lookbacks.split(",")
        if item.strip()
    }

    scan_rows = _read_scan_rows(scan_paths)
    ranking_rows, stability_points = _rank_scan_rows(scan_rows)
    concentration = _concentration_rows(
        allocation_paths,
        top3_threshold=float(args.top3_threshold),
        effective_n_threshold=float(args.effective_n_threshold),
    )

    any_guardrail_breach = any(row["guardrail"] == "BREACH" for row in concentration)
    switch_rule_met = _switch_rule_met(
        ranking_rows,
        switch_candidate=str(args.switch_candidate),
        primary_lookbacks=primary_lookbacks,
    )

    decision = {
        "keep_default": str(args.current_default),
        "switch_candidate": str(args.switch_candidate),
        "switch_rule_met": switch_rule_met,
        "any_guardrail_breach": any_guardrail_breach,
    }

    payload = {
        "ranking_rows": ranking_rows,
        "stability_points": stability_points,
        "concentration": concentration,
        "decision": decision,
    }

    markdown = _render_markdown(
        ranking_rows=ranking_rows,
        stability_points=stability_points,
        concentration=concentration,
        decision=decision,
    )
    print(markdown)

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(markdown, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
