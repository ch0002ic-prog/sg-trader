from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import AppConfig
from . import backtest as bt


DEFAULT_MIN_TRADES = 40
DEFAULT_MAX_DRAWDOWN: float | None = 0.25
PRIMARY_ALPHA_GRID = "3,5,7"
PRIMARY_VVIX_GRID = "90,110,130"
PRIMARY_VVIX_QUANTILE = 0.75


@dataclass
class ValidationResult:
    summary: dict[str, Any]
    scenarios: list[dict[str, Any]]


def _parse_grid(value: str) -> list[float]:
    items = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            items.append(float(raw))
        except ValueError:
            continue
    return items


def _compute_stats(
    spx: pd.Series,
    entries: pd.Series,
    min_trades: int,
) -> dict[str, float]:
    try:
        import vectorbt as vbt
    except ImportError as exc:
        raise RuntimeError(
            "vectorbt is required for backtest validation. Install it with pip install vectorbt."
        ) from exc

    exits = ~entries
    pf = vbt.Portfolio.from_signals(
        spx,
        entries=entries,
        exits=exits,
        freq="1D",
        direction="longonly",
        size=1.0,
        size_type="percent",
        init_cash=1.0,
    )

    value_series = pf.value().dropna()
    if value_series.empty:
        total_return = 0.0
        annualized_return = 0.0
        annualized_vol = 0.0
        max_drawdown = 0.0
    else:
        total_return = float(value_series.iloc[-1] / value_series.iloc[0] - 1)
        years = len(value_series) / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
        returns_series = value_series.pct_change().dropna()
        annualized_vol = float(returns_series.std() * np.sqrt(252))
        running_max = value_series.cummax()
        drawdown = (value_series / running_max) - 1.0
        max_drawdown = float(abs(drawdown.min()))

    entry_count = float(entries.sum())
    trade_count = float(pf.trades.count())

    score = 0.0
    if annualized_vol > 0:
        score = annualized_return / annualized_vol
    robust_score = score
    if trade_count > 0:
        scale = min(1.0, trade_count / float(min_trades))
        robust_score = score * np.sqrt(scale)
    if score >= 0:
        penalized_score = score * (1.0 - max_drawdown)
        penalized_robust = robust_score * (1.0 - max_drawdown)
    else:
        penalized_score = score * (1.0 + max_drawdown)
        penalized_robust = robust_score * (1.0 + max_drawdown)
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "return_to_vol": score,
        "return_to_vol_robust": robust_score,
        "return_to_vol_penalized": penalized_score,
        "return_to_vol_robust_penalized": penalized_robust,
        "max_drawdown": max_drawdown,
        "entry_count": entry_count,
        "trade_count": trade_count,
    }


def _apply_mask(series: pd.Series, mask: pd.Series | None) -> pd.Series:
    if mask is None:
        return series
    aligned = series.loc[mask.index]
    return aligned[mask]


def _rank_scenarios(
    scenarios: list[dict[str, Any]],
    min_trades: int,
    max_drawdown: float | None = None,
    dedupe: bool = False,
) -> list[dict[str, Any]]:
    filtered = []
    for row in scenarios:
        stats = row.get("stats", {})
        trades = stats.get("trade_count", 0.0)
        drawdown = stats.get("max_drawdown", 0.0)
        if trades < min_trades:
            continue
        if max_drawdown is not None and drawdown > max_drawdown:
            continue
        filtered.append(row)

    def _rank_value(row: dict[str, Any]) -> float:
        stats = row.get("stats", {})
        return float(
            stats.get(
                "return_to_vol_robust_penalized",
                stats.get(
                    "return_to_vol_penalized",
                    stats.get(
                        "return_to_vol_robust",
                        stats.get("return_to_vol", 0.0),
                    ),
                ),
            )
        )

    if not dedupe:
        return sorted(filtered, key=_rank_value, reverse=True)

    best_by_params: dict[tuple[float, float], dict[str, Any]] = {}
    for row in filtered:
        key = (
            float(row.get("alpha_spread_threshold", 0.0)),
            float(row.get("vvix_safe_threshold", 0.0)),
        )
        score = _rank_value(row)
        if key not in best_by_params or score > _rank_value(best_by_params[key]):
            best_by_params[key] = row

    return sorted(best_by_params.values(), key=_rank_value, reverse=True)


def build_validation_report(
    config: AppConfig,
    alpha_grid: Iterable[float],
    vvix_grid: Iterable[float],
    vix_quantile: float,
    vvix_quantile: float,
    min_trades: int,
    max_drawdown: float | None,
) -> ValidationResult:
    spx = bt._fetch_series(config.ticker)
    vix = bt._fetch_series(config.vix_ticker)
    vvix = bt._fetch_series(config.vvix_ticker)
    if spx.empty or vix.empty or vvix.empty:
        raise RuntimeError("Backtest validation fetch returned empty series")
    spx, vix, vvix = bt._align_series(spx, vix, vvix)
    if spx.empty or vix.empty or vvix.empty:
        raise RuntimeError("Backtest validation data alignment returned empty series")

    log_returns = np.log(spx / spx.shift(1))
    rv = log_returns.rolling(window=30).std() * np.sqrt(252) * 100
    rv = rv.dropna()
    if rv.empty:
        raise RuntimeError("Backtest validation RV series is empty")
    vix = vix.loc[rv.index]
    vvix = vvix.loc[rv.index]
    spx = spx.loc[rv.index]

    spread = vix - rv

    vix_threshold = float(vix.quantile(vix_quantile))
    vvix_threshold = float(vvix.quantile(vvix_quantile))

    regimes = {
        "full": None,
        "high_vix": vix >= vix_threshold,
        "high_vvix": vvix >= vvix_threshold,
    }

    scenarios: list[dict[str, Any]] = []
    for alpha_threshold in alpha_grid:
        for vvix_threshold_value in vvix_grid:
            entries = (spread > alpha_threshold) & (vvix < vvix_threshold_value)
            for regime_name, mask in regimes.items():
                spx_slice = _apply_mask(spx, mask)
                entries_slice = _apply_mask(entries, mask)
                if spx_slice.empty or entries_slice.empty:
                    continue
                stats = _compute_stats(spx_slice, entries_slice, min_trades)
                scenarios.append(
                    {
                        "alpha_spread_threshold": alpha_threshold,
                        "vvix_safe_threshold": vvix_threshold_value,
                        "regime": regime_name,
                        "stats": stats,
                    }
                )

    best = None
    worst = None
    if scenarios:
        best = max(scenarios, key=lambda row: row["stats"]["annualized_return"])
        worst = min(scenarios, key=lambda row: row["stats"]["annualized_return"])

    summary = {
        "as_of": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scenario_count": len(scenarios),
        "vix_quantile": vix_quantile,
        "vvix_quantile": vvix_quantile,
        "best": best,
        "worst": worst,
        "filters": {
            "min_trades": min_trades,
            "max_drawdown": max_drawdown,
        },
    }

    return ValidationResult(summary=summary, scenarios=scenarios)


def write_validation_report(
    config: AppConfig,
    output_dir: str | Path,
    alpha_grid: str,
    vvix_grid: str,
    vix_quantile: float,
    vvix_quantile: float,
    min_trades: int = DEFAULT_MIN_TRADES,
    max_drawdown: float | None = DEFAULT_MAX_DRAWDOWN,
) -> Path:
    alpha_values = _parse_grid(alpha_grid)
    vvix_values = _parse_grid(vvix_grid)
    if not alpha_values or not vvix_values:
        raise ValueError("alpha_grid and vvix_grid must contain numeric values")

    report = build_validation_report(
        config,
        alpha_grid=alpha_values,
        vvix_grid=vvix_values,
        vix_quantile=vix_quantile,
        vvix_quantile=vvix_quantile,
        min_trades=min_trades,
        max_drawdown=max_drawdown,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "backtest_validation.json"
    md_path = output_dir / "backtest_validation.md"
    def _rank_value(row: dict[str, Any]) -> float:
        stats = row.get("stats", {})
        return float(
            stats.get(
                "return_to_vol_robust_penalized",
                stats.get(
                    "return_to_vol_penalized",
                    stats.get(
                        "return_to_vol_robust",
                        stats.get("return_to_vol", 0.0),
                    ),
                ),
            )
        )

    ranked = sorted(report.scenarios, key=_rank_value, reverse=True)
    ranked_filtered = _rank_scenarios(
        report.scenarios,
        min_trades,
        max_drawdown=max_drawdown,
    )
    ranked_unique = _rank_scenarios(
        report.scenarios,
        min_trades,
        max_drawdown=max_drawdown,
        dedupe=True,
    )
    payload = {
        "summary": report.summary,
        "scenarios": report.scenarios,
        "ranked": ranked[:10],
        "ranked_filtered": ranked_filtered[:10],
        "ranked_unique": ranked_unique[:10],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Backtest Validation",
        "",
        f"As of: {report.summary['as_of']}",
        f"Scenarios: {report.summary['scenario_count']}",
        f"Min trades filter: {min_trades}",
        "",
        "## Top 10 Return/Vol (All)",
        "",
        "alpha | vvix | regime | ann_return | ann_vol | r/vol | r/vol* | r/vol_pen | r/vol*_pen | dd | trades",
        "--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---",
    ]
    for row in ranked[:10]:
        stats = row["stats"]
        lines.append(
            f"{row['alpha_spread_threshold']} | {row['vvix_safe_threshold']} | "
            f"{row['regime']} | {stats['annualized_return']:.4f} | "
            f"{stats['annualized_volatility']:.4f} | "
            f"{stats['return_to_vol']:.3f} | {stats['return_to_vol_robust']:.3f} | "
            f"{stats['return_to_vol_penalized']:.3f} | "
            f"{stats['return_to_vol_robust_penalized']:.3f} | "
            f"{stats['max_drawdown']:.3f} | {stats['trade_count']:.0f}"
        )
    lines.extend(
        [
            "",
            "## Top 10 Return/Vol (Trades Filtered)",
            "",
            "alpha | vvix | regime | ann_return | ann_vol | r/vol | r/vol* | r/vol_pen | r/vol*_pen | dd | trades",
            "--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---",
        ]
    )
    for row in ranked_filtered[:10]:
        stats = row["stats"]
        lines.append(
            f"{row['alpha_spread_threshold']} | {row['vvix_safe_threshold']} | "
            f"{row['regime']} | {stats['annualized_return']:.4f} | "
            f"{stats['annualized_volatility']:.4f} | "
            f"{stats['return_to_vol']:.3f} | {stats['return_to_vol_robust']:.3f} | "
            f"{stats['return_to_vol_penalized']:.3f} | "
            f"{stats['return_to_vol_robust_penalized']:.3f} | "
            f"{stats['max_drawdown']:.3f} | {stats['trade_count']:.0f}"
        )
    lines.extend(
        [
            "",
            "## Top 10 Unique Params (Trades Filtered)",
            "",
            "alpha | vvix | regime | ann_return | ann_vol | r/vol | r/vol* | r/vol_pen | r/vol*_pen | dd | trades",
            "--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---",
        ]
    )
    for row in ranked_unique[:10]:
        stats = row["stats"]
        lines.append(
            f"{row['alpha_spread_threshold']} | {row['vvix_safe_threshold']} | "
            f"{row['regime']} | {stats['annualized_return']:.4f} | "
            f"{stats['annualized_volatility']:.4f} | "
            f"{stats['return_to_vol']:.3f} | {stats['return_to_vol_robust']:.3f} | "
            f"{stats['return_to_vol_penalized']:.3f} | "
            f"{stats['return_to_vol_robust_penalized']:.3f} | "
            f"{stats['max_drawdown']:.3f} | {stats['trade_count']:.0f}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    top_csv = output_dir / "backtest_top_unique.csv"
    top_md = output_dir / "backtest_top_unique.md"
    if ranked_unique:
        import csv

        with top_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "q",
                    "alpha",
                    "vvix",
                    "regime",
                    "trades",
                        "score",
                        "robust",
                        "score_penalized",
                        "robust_penalized",
                        "max_drawdown",
                    "ret",
                    "vol",
                ],
            )
            writer.writeheader()
            for row in ranked_unique[:10]:
                stats = row["stats"]
                writer.writerow(
                    {
                        "q": report.summary["vvix_quantile"],
                        "alpha": row["alpha_spread_threshold"],
                        "vvix": row["vvix_safe_threshold"],
                        "regime": row["regime"],
                        "trades": stats["trade_count"],
                        "score": stats["return_to_vol"],
                        "robust": stats["return_to_vol_robust"],
                        "score_penalized": stats["return_to_vol_penalized"],
                        "robust_penalized": stats["return_to_vol_robust_penalized"],
                        "max_drawdown": stats["max_drawdown"],
                        "ret": stats["annualized_return"],
                        "vol": stats["annualized_volatility"],
                    }
                )

        top_lines = [
            "# Top 10 Unique Params (Trades Filtered)",
            "",
            "q | alpha | vvix | regime | trades | score | robust | score_pen | robust_pen | dd | ret | vol",
            "--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---",
        ]
        for row in ranked_unique[:10]:
            stats = row["stats"]
            top_lines.append(
                f"{report.summary['vvix_quantile']} | {row['alpha_spread_threshold']} | "
                f"{row['vvix_safe_threshold']} | {row['regime']} | "
                f"{stats['trade_count']:.0f} | {stats['return_to_vol']:.3f} | "
                f"{stats['return_to_vol_robust']:.3f} | {stats['return_to_vol_penalized']:.3f} | "
                f"{stats['return_to_vol_robust_penalized']:.3f} | {stats['max_drawdown']:.3f} | "
                f"{stats['annualized_return']:.4f} | {stats['annualized_volatility']:.4f}"
            )
        top_md.write_text("\n".join(top_lines) + "\n", encoding="utf-8")
    return path
