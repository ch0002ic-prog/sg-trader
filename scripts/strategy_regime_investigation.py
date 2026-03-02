from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import importlib.util
import json
import re

import numpy as np
import pandas as pd

from sg_trader.cache_utils import read_series_cache
from sg_trader.config import load_config
from sg_trader.pnl_dashboard import (
    _compute_performance_metrics,
    _parse_date,
    _parse_timestamp,
)

REPORTS = {
    "zscore_throttle_tuned": "reports/pnl_backfill_zscore_throttle_tuned_2013-01-02_2026-02-06.json",
    "zscore_throttle_cut50": "reports/pnl_backfill_zscore_throttle_cut50_2013-01-02_2026-02-06.json",
    "regime_blend_ief20": "reports/pnl_backfill_regime_blend_ief20_2013-01-02_2026-02-06.json",
    "regime_blend_vix29": "reports/pnl_backfill_regime_blend_vix29_2013-01-02_2026-02-06.json",
}

TRADING_DAYS = 252


@dataclass
class StrategyData:
    name: str
    daily: list[dict]
    normalized: pd.Series
    returns: pd.Series


def _build_daily(entries: list[dict]) -> list[dict]:
    daily_realized: dict[str, float] = {}
    daily_unrealized: dict[str, dict[str, float]] = {}
    daily_last_pnl_ts: dict[str, dict[str, object]] = {}

    for entry in entries:
        action = entry.get("action")
        if action not in {"PAPER_PNL", "PAPER_REALIZED"}:
            continue
        timestamp = entry.get("timestamp", "")
        parsed = _parse_timestamp(timestamp)
        if not parsed:
            continue
        day_key = parsed.date().strftime("%Y-%m-%d")
        details = entry.get("details", {})
        symbol = entry.get("ticker", "Unknown")
        if action == "PAPER_REALIZED":
            pnl = details.get("realized_pnl")
            if isinstance(pnl, (int, float)):
                daily_realized[day_key] = daily_realized.get(day_key, 0.0) + float(pnl)
        else:
            pnl = details.get("unrealized_pnl")
            if not isinstance(pnl, (int, float)):
                continue
            daily_unrealized.setdefault(day_key, {})
            daily_last_pnl_ts.setdefault(day_key, {})
            prior_ts = daily_last_pnl_ts[day_key].get(symbol)
            if prior_ts is None or parsed >= prior_ts:
                daily_unrealized[day_key][symbol] = float(pnl)
                daily_last_pnl_ts[day_key][symbol] = parsed

    all_days = sorted(set(daily_realized.keys()) | set(daily_unrealized.keys()))
    daily: list[dict] = []
    cumulative_realized = 0.0
    for day_key in all_days:
        realized = daily_realized.get(day_key, 0.0)
        unrealized = sum(daily_unrealized.get(day_key, {}).values())
        cumulative_realized += realized
        day_dt = _parse_date(day_key)
        week_key = f"{day_dt.isocalendar().year}-W{day_dt.isocalendar().week:02d}"
        month_key = day_dt.strftime("%Y-%m")
        equity = cumulative_realized + unrealized
        daily.append(
            {
                "date": day_key,
                "week": week_key,
                "month": month_key,
                "realized": realized,
                "unrealized": unrealized,
                "cumulative_realized": cumulative_realized,
                "equity": equity,
            }
        )
    return daily


def _normalized_series(daily: list[dict], initial_capital: float) -> pd.Series:
    dates = [row["date"] for row in daily]
    values = [1.0 + float(row["equity"]) / initial_capital for row in daily]
    idx = pd.to_datetime(dates, utc=True).tz_convert(None)
    return pd.Series(values, index=idx).sort_index()


def _returns_series(normalized: pd.Series) -> pd.Series:
    returns = normalized.pct_change().dropna()
    returns.index = pd.to_datetime(returns.index, utc=True).tz_convert(None)
    return returns


def _filter_daily(daily: list[dict], dates: pd.Index) -> list[dict]:
    date_set = set(date.strftime("%Y-%m-%d") for date in dates)
    return [row for row in daily if row["date"] in date_set]


def _metrics_for_dates(daily: list[dict], dates: pd.Index, config) -> dict:
    filtered = _filter_daily(daily, dates)
    return _compute_performance_metrics(
        filtered,
        config.paper_initial_capital,
        config.risk_free_rate,
        config.pnl_downside_min_days,
    )


def _fmt_num(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value * 100:.{digits}f}%"


def _fmt_bps(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value * 10000:.{digits}f} bps"


def _daily_date_index(daily: list[dict[str, object]]) -> pd.Index:
    if not daily:
        return pd.Index([])
    dates = [row.get("date") for row in daily]
    idx = pd.to_datetime(dates, utc=True).tz_convert(None)
    return idx.dropna()


def _metrics_for_range(
    daily: list[dict[str, object]],
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    config,
) -> dict[str, float]:
    idx = _daily_date_index(daily)
    if idx.empty:
        return {}
    mask = (idx >= start_dt) & (idx <= end_dt)
    dates = idx[mask]
    if dates.empty:
        return {}
    return _metrics_for_dates(daily, dates, config)


def _oos_split_defs(daily: list[dict[str, object]]) -> list[dict[str, object]]:
    idx = _daily_date_index(daily)
    if idx.empty:
        return []
    start_dt = idx.min()
    end_dt = idx.max()
    splits = []

    split_a_train_end = min(end_dt, pd.Timestamp("2022-12-31"))
    split_a_test_start = pd.Timestamp("2023-01-01")
    splits.append(
        {
            "label": "2017-10-30..2022-12-31 / 2023-01-01+",
            "train_start": start_dt,
            "train_end": split_a_train_end,
            "test_start": split_a_test_start,
            "test_end": end_dt,
        }
    )

    split_b_train_end = min(end_dt, pd.Timestamp("2020-12-31"))
    split_b_test_start = pd.Timestamp("2021-01-01")
    splits.append(
        {
            "label": "2017-10-30..2020-12-31 / 2021-01-01+",
            "train_start": start_dt,
            "train_end": split_b_train_end,
            "test_start": split_b_test_start,
            "test_end": end_dt,
        }
    )

    split_c_test_start = end_dt - pd.DateOffset(years=3)
    split_c_train_end = split_c_test_start - pd.Timedelta(days=1)
    splits.append(
        {
            "label": "last 3Y test",
            "train_start": start_dt,
            "train_end": split_c_train_end,
            "test_start": split_c_test_start,
            "test_end": end_dt,
        }
    )

    return splits


def _append_additional_checks(
    report_lines: list[str],
    label: str,
    strategy: StrategyData,
    entries: list[dict[str, object]] | None,
    config,
    tracking_error: dict[str, float] | None = None,
    collector: dict[str, list[dict[str, object]]] | None = None,
) -> None:
    report_lines.append(f"### {label}")
    report_lines.append("")

    report_lines.append("#### Out-of-sample splits")
    splits = _oos_split_defs(strategy.daily)
    if splits:
        report_lines.append(
            "| Split | Train Sharpe | Train CAGR | Train Max DD | Test Sharpe | Test CAGR | Test Max DD |"
        )
        report_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for split in splits:
            train_metrics = _metrics_for_range(
                strategy.daily,
                split["train_start"],
                split["train_end"],
                config,
            )
            test_metrics = _metrics_for_range(
                strategy.daily,
                split["test_start"],
                split["test_end"],
                config,
            )
            report_lines.append(
                "| {label} | {train_sharpe} | {train_cagr} | {train_dd} | {test_sharpe} | {test_cagr} | {test_dd} |".format(
                    label=split["label"],
                    train_sharpe=_fmt_num(train_metrics.get("sharpe_ratio")),
                    train_cagr=_fmt_pct(train_metrics.get("cagr")),
                    train_dd=_fmt_pct(train_metrics.get("max_drawdown_pct")),
                    test_sharpe=_fmt_num(test_metrics.get("sharpe_ratio")),
                    test_cagr=_fmt_pct(test_metrics.get("cagr")),
                    test_dd=_fmt_pct(test_metrics.get("max_drawdown_pct")),
                )
            )
            if collector is not None:
                collector["oos"].append(
                    {
                        "strategy": strategy.name,
                        "split": split["label"],
                        "train_sharpe": train_metrics.get("sharpe_ratio"),
                        "train_cagr": train_metrics.get("cagr"),
                        "train_max_dd": train_metrics.get("max_drawdown_pct"),
                        "test_sharpe": test_metrics.get("sharpe_ratio"),
                        "test_cagr": test_metrics.get("cagr"),
                        "test_max_dd": test_metrics.get("max_drawdown_pct"),
                    }
                )
    else:
        report_lines.append("- Out-of-sample splits unavailable.")

    report_lines.append("")
    report_lines.append("#### Cost stress test (5/10/20 bps)")
    if entries:
        turnover = _compute_turnover_series(entries, config.paper_initial_capital)
    else:
        turnover = pd.Series(dtype=float)
    if entries and not turnover.empty:
        report_lines.append(
            "| Cost (bps) | Sharpe | Final Equity | CAGR | Vol | Max DD | Calmar |"
        )
        report_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for cost_bps in (5.0, 10.0, 20.0):
            metrics = _apply_cost_stress_test(
                strategy.daily,
                turnover,
                config,
                cost_bps=cost_bps,
            )
            report_lines.append(
                "| {bps} | {sharpe} | {end} | {cagr} | {vol} | {dd} | {calmar} |".format(
                    bps=_fmt_num(metrics.get("cost_bps"), 0),
                    sharpe=_fmt_num(metrics.get("sharpe_ratio")),
                    end=_fmt_num(metrics.get("normalized_end"), 4),
                    cagr=_fmt_pct(metrics.get("cagr")),
                    vol=_fmt_pct(metrics.get("annualized_volatility")),
                    dd=_fmt_pct(metrics.get("max_drawdown_pct")),
                    calmar=_fmt_num(metrics.get("calmar_ratio")),
                )
            )
            if collector is not None:
                collector["cost"].append(
                    {
                        "strategy": strategy.name,
                        "cost_bps": metrics.get("cost_bps"),
                        "sharpe": metrics.get("sharpe_ratio"),
                        "final_equity": metrics.get("normalized_end"),
                        "cagr": metrics.get("cagr"),
                        "vol": metrics.get("annualized_volatility"),
                        "max_dd": metrics.get("max_drawdown_pct"),
                        "calmar": metrics.get("calmar_ratio"),
                    }
                )
    else:
        report_lines.append("- Cost stress test unavailable.")

    report_lines.append("")
    report_lines.append("#### Regime breakdown")
    regimes = _regime_breakdown_single(strategy.normalized, strategy.daily, config)
    if regimes:
        report_lines.append(
            "| Regime | Sharpe | Final Equity | CAGR | Vol | Max DD | Calmar | Points |"
        )
        report_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for regime, metrics in regimes.items():
            report_lines.append(
                "| {regime} | {sharpe} | {end} | {cagr} | {vol} | {dd} | {calmar} | {points} |".format(
                    regime=regime,
                    sharpe=_fmt_num(metrics.get("sharpe_ratio")),
                    end=_fmt_num(metrics.get("normalized_end"), 4),
                    cagr=_fmt_pct(metrics.get("cagr")),
                    vol=_fmt_pct(metrics.get("annualized_volatility")),
                    dd=_fmt_pct(metrics.get("max_drawdown_pct")),
                    calmar=_fmt_num(metrics.get("calmar_ratio")),
                    points=metrics.get("data_points", 0),
                )
            )
            if collector is not None:
                collector["regime"].append(
                    {
                        "strategy": strategy.name,
                        "regime": regime,
                        "sharpe": metrics.get("sharpe_ratio"),
                        "final_equity": metrics.get("normalized_end"),
                        "cagr": metrics.get("cagr"),
                        "vol": metrics.get("annualized_volatility"),
                        "max_dd": metrics.get("max_drawdown_pct"),
                        "calmar": metrics.get("calmar_ratio"),
                        "points": metrics.get("data_points", 0),
                    }
                )
    else:
        report_lines.append("- Regime breakdown unavailable.")

    report_lines.append("")
    report_lines.append("#### Rolling-window stability")
    for window_years in (3, 5):
        summary = _rolling_summary({strategy.name: strategy}, config, window_years)
        stats = summary.get(strategy.name, {})
        report_lines.append(
            f"- {window_years}Y: sharpe_mean={_fmt_num(stats.get('sharpe_mean'))}, sharpe_min={_fmt_num(stats.get('sharpe_min'))}, cagr_mean={_fmt_pct(stats.get('cagr_mean'))}, cagr_min={_fmt_pct(stats.get('cagr_min'))}, windows={stats.get('windows', 0)}"
        )
        if collector is not None:
            collector["rolling"].append(
                {
                    "strategy": strategy.name,
                    "window_years": window_years,
                    "sharpe_mean": stats.get("sharpe_mean"),
                    "sharpe_min": stats.get("sharpe_min"),
                    "cagr_mean": stats.get("cagr_mean"),
                    "cagr_min": stats.get("cagr_min"),
                    "windows": stats.get("windows", 0),
                }
            )

    report_lines.append("")
    report_lines.append("#### Tracking error vs synthetic")
    if tracking_error:
        report_lines.append("| Metric | Value |")
        report_lines.append("| --- | --- |")
        report_lines.append(
            f"| Tracking error (annualized) | {_fmt_pct(tracking_error.get('tracking_error'))} |"
        )
        report_lines.append(
            f"| Return correlation | {_fmt_num(tracking_error.get('return_corr'), 3)} |"
        )
        report_lines.append(
            f"| Mean abs daily return diff | {_fmt_pct(tracking_error.get('mean_abs_return_diff'), 3)} |"
        )
        report_lines.append(
            f"| Max equity gap (pct) | {_fmt_pct(tracking_error.get('max_equity_gap_pct'))} |"
        )
        if collector is not None:
            collector["tracking"].append(
                {
                    "strategy": strategy.name,
                    "tracking_error": tracking_error.get("tracking_error"),
                    "return_corr": tracking_error.get("return_corr"),
                    "mean_abs_return_diff": tracking_error.get("mean_abs_return_diff"),
                    "max_equity_gap_pct": tracking_error.get("max_equity_gap_pct"),
                }
            )
    else:
        report_lines.append("- Tracking error check unavailable.")
    report_lines.append("")


def _init_checks_collector() -> dict[str, list[dict[str, object]]]:
    return {"oos": [], "cost": [], "regime": [], "rolling": [], "tracking": []}


def _export_checks_csv(
    output_dir: Path, slug: str, collector: dict[str, list[dict[str, object]]]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "oos": "oos_splits",
        "cost": "cost_stress",
        "regime": "regime_breakdown",
        "rolling": "rolling",
        "tracking": "tracking_error",
    }
    for key, stem in mapping.items():
        rows = collector.get(key, [])
        if not rows:
            continue
        path = output_dir / f"{stem}_{slug}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)


def _metrics_from_normalized(normalized: pd.Series, config) -> dict[str, float]:
    if normalized.empty:
        return {}
    daily = [
        {
            "date": idx.strftime("%Y-%m-%d"),
            "equity": (value - 1.0) * config.paper_initial_capital,
        }
        for idx, value in normalized.items()
    ]
    return _compute_performance_metrics(
        daily,
        config.paper_initial_capital,
        config.risk_free_rate,
        config.pnl_downside_min_days,
    )


def _append_synth_vs_positions_table(
    report_lines: list[str],
    label: str,
    items: list[dict[str, object]],
    config,
    top_n: int = 2,
) -> None:
    candidates: list[dict[str, object]] = []
    for item in items:
        strategy = item.get("strategy")
        normalized = item.get("synthetic_normalized")
        if not isinstance(strategy, StrategyData):
            continue
        if not isinstance(normalized, pd.Series) or normalized.empty:
            continue
        candidates.append(
            {
                "strategy": strategy,
                "synthetic_normalized": normalized,
                "score": item.get("composite_score"),
            }
        )
    if not candidates:
        return
    sorted_candidates = sorted(
        candidates,
        key=lambda row: row.get("score") if row.get("score") is not None else -999,
        reverse=True,
    )[:top_n]
    report_lines.append(f"### Synthetic vs positions (top composite): {label}")
    for candidate in sorted_candidates:
        strategy = candidate["strategy"]
        synth_metrics = _metrics_from_normalized(
            candidate["synthetic_normalized"],
            config,
        )
        pos_metrics = _compute_performance_metrics(
            strategy.daily,
            config.paper_initial_capital,
            config.risk_free_rate,
            config.pnl_downside_min_days,
        )
        report_lines.append(f"#### {strategy.name}")
        report_lines.append(
            "| Variant | Sharpe | Final Equity | CAGR | Vol | Max DD | Calmar |"
        )
        report_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        report_lines.append(
            "| synthetic | {sharpe} | {end} | {cagr} | {vol} | {dd} | {calmar} |".format(
                sharpe=_fmt_num(synth_metrics.get("sharpe_ratio")),
                end=_fmt_num(synth_metrics.get("normalized_end"), 4),
                cagr=_fmt_pct(synth_metrics.get("cagr")),
                vol=_fmt_pct(synth_metrics.get("annualized_volatility")),
                dd=_fmt_pct(synth_metrics.get("max_drawdown_pct")),
                calmar=_fmt_num(synth_metrics.get("calmar_ratio")),
            )
        )
        report_lines.append(
            "| positions | {sharpe} | {end} | {cagr} | {vol} | {dd} | {calmar} |".format(
                sharpe=_fmt_num(pos_metrics.get("sharpe_ratio")),
                end=_fmt_num(pos_metrics.get("normalized_end"), 4),
                cagr=_fmt_pct(pos_metrics.get("cagr")),
                vol=_fmt_pct(pos_metrics.get("annualized_volatility")),
                dd=_fmt_pct(pos_metrics.get("max_drawdown_pct")),
                calmar=_fmt_num(pos_metrics.get("calmar_ratio")),
            )
        )
        report_lines.append(
            "| delta (pos - synth) | {sharpe} | {end} | {cagr} | {vol} | {dd} | {calmar} |".format(
                sharpe=_fmt_num(
                    (
                        pos_metrics.get("sharpe_ratio")
                        - synth_metrics.get("sharpe_ratio")
                    )
                    if pos_metrics.get("sharpe_ratio") is not None
                    and synth_metrics.get("sharpe_ratio") is not None
                    else None
                ),
                end=_fmt_num(
                    (
                        pos_metrics.get("normalized_end")
                        - synth_metrics.get("normalized_end")
                    )
                    if pos_metrics.get("normalized_end") is not None
                    and synth_metrics.get("normalized_end") is not None
                    else None,
                    4,
                ),
                cagr=_fmt_pct(
                    (pos_metrics.get("cagr") - synth_metrics.get("cagr"))
                    if pos_metrics.get("cagr") is not None
                    and synth_metrics.get("cagr") is not None
                    else None
                ),
                vol=_fmt_pct(
                    (
                        pos_metrics.get("annualized_volatility")
                        - synth_metrics.get("annualized_volatility")
                    )
                    if pos_metrics.get("annualized_volatility") is not None
                    and synth_metrics.get("annualized_volatility") is not None
                    else None
                ),
                dd=_fmt_pct(
                    (
                        pos_metrics.get("max_drawdown_pct")
                        - synth_metrics.get("max_drawdown_pct")
                    )
                    if pos_metrics.get("max_drawdown_pct") is not None
                    and synth_metrics.get("max_drawdown_pct") is not None
                    else None
                ),
                calmar=_fmt_num(
                    (
                        pos_metrics.get("calmar_ratio")
                        - synth_metrics.get("calmar_ratio")
                    )
                    if pos_metrics.get("calmar_ratio") is not None
                    and synth_metrics.get("calmar_ratio") is not None
                    else None
                ),
            )
        )
    report_lines.append("")


def _best_row(rows: list[dict[str, object]], key: str) -> dict[str, object] | None:
    best = None
    best_value = None
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        if (
            best is None
            or (best_value is not None and value > best_value)
            or best_value is None
        ):
            best = row
            best_value = value
    return best


def _format_dyn_params(row: dict[str, object]) -> str:
    return "cut={cut} min_lev={min_lev} vix_cap={vix_cap} boost={boost} vix_th={threshold} lag={lag}".format(
        cut=_fmt_num(row.get("cut"), 2),
        min_lev=_fmt_num(row.get("min_leverage"), 2),
        vix_cap=_fmt_num(row.get("vix_cap"), 0),
        boost=_fmt_num(row.get("ief_boost"), 2),
        threshold=_fmt_num(row.get("vix_threshold"), 0),
        lag=_fmt_num(row.get("lag"), 0),
    )


def _append_scoreboard(
    report_lines: list[str],
    synthetic_rows: list[dict[str, object]],
    positions_rows: list[dict[str, object]],
    output_dir: Path | None = None,
    slug: str | None = None,
) -> None:
    if not synthetic_rows and not positions_rows:
        return
    scoreboard_rows: list[dict[str, object]] = []
    report_lines.append("### Scoreboard: Best Sharpe / Returns")
    report_lines.append(
        "| Source | Criterion | Strategy/Params | Sharpe | Final Equity | CAGR | Max DD |"
    )
    report_lines.append("| --- | --- | --- | --- | --- | --- | --- |")

    def _append_row(
        source: str, criterion: str, row: dict[str, object], label: str
    ) -> None:
        scoreboard_rows.append(
            {
                "source": source,
                "criterion": criterion,
                "strategy_or_params": label,
                "sharpe": row.get("sharpe"),
                "final_equity": row.get("final_equity"),
                "cagr": row.get("cagr"),
                "max_dd": row.get("max_dd"),
            }
        )
        report_lines.append(
            "| {source} | {criterion} | {label} | {sharpe} | {end} | {cagr} | {dd} |".format(
                source=source,
                criterion=criterion,
                label=label,
                sharpe=_fmt_num(row.get("sharpe")),
                end=_fmt_num(row.get("final_equity"), 4),
                cagr=_fmt_pct(row.get("cagr")),
                dd=_fmt_pct(row.get("max_dd")),
            )
        )

    best_synth_sharpe = _best_row(synthetic_rows, "sharpe")
    best_synth_return = _best_row(synthetic_rows, "final_equity")
    if best_synth_sharpe is not None:
        _append_row(
            "synthetic grid",
            "best Sharpe",
            best_synth_sharpe,
            _format_dyn_params(best_synth_sharpe),
        )
    if best_synth_return is not None:
        _append_row(
            "synthetic grid",
            "best returns",
            best_synth_return,
            _format_dyn_params(best_synth_return),
        )

    best_pos_sharpe = _best_row(positions_rows, "sharpe")
    best_pos_return = _best_row(positions_rows, "final_equity")
    if best_pos_sharpe is not None:
        _append_row(
            "positions (common window)",
            "best Sharpe",
            best_pos_sharpe,
            str(best_pos_sharpe.get("name")),
        )
    if best_pos_return is not None:
        _append_row(
            "positions (common window)",
            "best returns",
            best_pos_return,
            str(best_pos_return.get("name")),
        )
    report_lines.append("")
    if output_dir is not None and slug:
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(scoreboard_rows).to_csv(
            output_dir / f"scoreboard_{slug}.csv",
            index=False,
        )


def _composite_row(
    strategy: StrategyData,
    entries: list[dict[str, object]] | None,
    config,
) -> dict[str, object]:
    splits = _oos_split_defs(strategy.daily)
    test_metrics: dict[str, float] = {}
    if splits:
        split = splits[0]
        test_metrics = _metrics_for_range(
            strategy.daily,
            split["test_start"],
            split["test_end"],
            config,
        )
    cost_metrics: dict[str, float] = {}
    if entries:
        turnover = _compute_turnover_series(entries, config.paper_initial_capital)
        if not turnover.empty:
            cost_metrics = _apply_cost_stress_test(
                strategy.daily,
                turnover,
                config,
                cost_bps=10.0,
            )
    test_sharpe = test_metrics.get("sharpe_ratio")
    cost_sharpe = cost_metrics.get("sharpe_ratio")
    test_dd = test_metrics.get("max_drawdown_pct")
    score = None
    if test_sharpe is not None:
        score = test_sharpe
        if cost_sharpe is not None:
            score = 0.7 * test_sharpe + 0.3 * cost_sharpe
        if test_dd is not None:
            score -= 0.25 * abs(test_dd)
    return {
        "strategy": strategy.name,
        "test_sharpe": test_sharpe,
        "test_cagr": test_metrics.get("cagr"),
        "test_max_dd": test_dd,
        "cost10_sharpe": cost_sharpe,
        "score": score,
    }


def _append_composite_table(
    report_lines: list[str],
    label: str,
    rows: list[dict[str, object]],
    output_dir: Path,
    slug: str,
) -> None:
    if not rows:
        return
    report_lines.append(f"### Composite ranking: {label}")
    report_lines.append(
        "Score = 0.7 * test_sharpe + 0.3 * cost10_sharpe - 0.25 * abs(test_max_dd) (2023+ test)."
    )
    report_lines.append(
        "| Rank | Strategy | Test Sharpe | Test CAGR | Test Max DD | Cost10 Sharpe | Score |"
    )
    report_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    sorted_rows = sorted(
        rows,
        key=lambda row: row.get("score") if row.get("score") is not None else -999,
        reverse=True,
    )
    for idx, row in enumerate(sorted_rows, start=1):
        report_lines.append(
            "| {rank} | {strategy} | {test_sharpe} | {test_cagr} | {test_dd} | {cost_sharpe} | {score} |".format(
                rank=idx,
                strategy=row.get("strategy"),
                test_sharpe=_fmt_num(row.get("test_sharpe")),
                test_cagr=_fmt_pct(row.get("test_cagr")),
                test_dd=_fmt_pct(row.get("test_max_dd")),
                cost_sharpe=_fmt_num(row.get("cost10_sharpe")),
                score=_fmt_num(row.get("score")),
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sorted_rows).to_csv(output_dir / f"composite_{slug}.csv", index=False)
    report_lines.append("")


def _append_deep_diagnostics(
    report_lines: list[str],
    label: str,
    items: list[dict[str, object]],
    config,
    top_n: int = 2,
) -> None:
    if not items:
        return
    report_lines.append(f"### Deep diagnostics: {label}")
    for idx, item in enumerate(items[:top_n], start=1):
        strategy = item["strategy"]
        entries = item.get("entries")
        report_lines.append(f"#### {strategy.name} (rank {idx})")
        if not isinstance(entries, list):
            report_lines.append("- Diagnostics unavailable (missing entries).")
            continue
        exposure = _compute_exposure_stats(entries, config.paper_initial_capital)
        report_lines.append("| Exposure Metric | Value |")
        report_lines.append("| --- | --- |")
        report_lines.append(
            f"| Avg leverage | {_fmt_num(exposure.get('avg_leverage'))} |"
        )
        report_lines.append(
            f"| Median leverage | {_fmt_num(exposure.get('median_leverage'))} |"
        )
        report_lines.append(
            f"| % days leverage < 1 | {_fmt_pct(exposure.get('pct_leverage_lt_1'))} |"
        )
        report_lines.append(
            f"| Avg SPY weight | {_fmt_pct(exposure.get('avg_spy_weight'))} |"
        )
        report_lines.append(
            f"| Avg IEF weight | {_fmt_pct(exposure.get('avg_ief_weight'))} |"
        )

        turnover = _compute_turnover_series(entries, config.paper_initial_capital)
        report_lines.append("| Turnover Metric | Value |")
        report_lines.append("| --- | --- |")
        report_lines.append(f"| Avg daily turnover | {_fmt_pct(turnover.mean())} |")
        report_lines.append(
            f"| Median daily turnover | {_fmt_pct(turnover.median())} |"
        )
        report_lines.append(
            f"| 95th pct daily turnover | {_fmt_pct(turnover.quantile(0.95))} |"
        )
        report_lines.append(
            f"| Days > 10% turnover | {_fmt_pct((turnover > 0.10).mean())} |"
        )

        buckets = _compute_leverage_buckets(entries, config.paper_initial_capital)
        report_lines.append("| Leverage Bucket | Share |")
        report_lines.append("| --- | --- |")
        report_lines.append(f"| <0.5 | {_fmt_pct(buckets.get('lt_0_5'))} |")
        report_lines.append(f"| 0.5-0.8 | {_fmt_pct(buckets.get('0_5_0_8'))} |")
        report_lines.append(f"| 0.8-1.0 | {_fmt_pct(buckets.get('0_8_1_0'))} |")
        report_lines.append(f"| 1.0-1.2 | {_fmt_pct(buckets.get('1_0_1_2'))} |")
        report_lines.append(f"| >=1.2 | {_fmt_pct(buckets.get('ge_1_2'))} |")

        concentration = _concentration_stats(entries)
        report_lines.append("| Concentration Metric | Value |")
        report_lines.append("| --- | --- |")
        report_lines.append(
            f"| HHI mean | {_fmt_num(concentration.get('hhi_mean'), 3)} |"
        )
        report_lines.append(
            f"| HHI p95 | {_fmt_num(concentration.get('hhi_p95'), 3)} |"
        )
        report_lines.append(
            f"| Top-1 weight mean | {_fmt_pct(concentration.get('top1_mean'))} |"
        )
        report_lines.append(
            f"| Effective N mean | {_fmt_num(concentration.get('effective_n_mean'), 2)} |"
        )
        report_lines.append(
            f"| Days top-1 > 50% | {_fmt_pct(concentration.get('pct_top1_gt_50'))} |"
        )

        tracking_by_regime = item.get("tracking_by_regime")
        if isinstance(tracking_by_regime, dict) and tracking_by_regime:
            report_lines.append(
                "| Regime | Tracking error | Return corr | Mean abs diff | Points |"
            )
            report_lines.append("| --- | --- | --- | --- | --- |")
            for regime, metrics in tracking_by_regime.items():
                report_lines.append(
                    "| {regime} | {te} | {corr} | {diff} | {count} |".format(
                        regime=regime,
                        te=_fmt_pct(metrics.get("tracking_error")),
                        corr=_fmt_num(metrics.get("return_corr"), 3),
                        diff=_fmt_pct(metrics.get("mean_abs_return_diff"), 3),
                        count=metrics.get("count", 0),
                    )
                )
        report_lines.append("")


def _load_backfill_module() -> object:
    path = Path("scripts/backfill_paper_pnl_from_strategy.py").resolve()
    spec = importlib.util.spec_from_file_location("backfill", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load backfill module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _series_long_cache_key(ticker: str) -> str:
    return "market_long_" + re.sub(r"[^a-zA-Z0-9_-]", "_", ticker)


def _prepare_strategy_data(config) -> dict[str, StrategyData]:
    results: dict[str, StrategyData] = {}
    for name, rel_path in REPORTS.items():
        entries = json.loads(Path(rel_path).read_text(encoding="utf-8"))
        daily = _build_daily(entries)
        normalized = _normalized_series(daily, config.paper_initial_capital)
        returns = _returns_series(normalized)
        results[name] = StrategyData(
            name=name,
            daily=daily,
            normalized=normalized,
            returns=returns,
        )
    return results


def _strategy_from_entries(
    name: str, entries: list[dict], initial_capital: float
) -> StrategyData:
    daily = _build_daily(entries)
    normalized = _normalized_series(daily, initial_capital)
    returns = _returns_series(normalized)
    return StrategyData(name=name, daily=daily, normalized=normalized, returns=returns)


def _common_dates(strategy_data: dict[str, StrategyData]) -> pd.Index:
    common: pd.Index | None = None
    for data in strategy_data.values():
        idx = data.normalized.index
        common = idx if common is None else common.intersection(idx)
    return common if common is not None else pd.Index([])


def _load_market_series(common_dates: pd.Index) -> dict[str, pd.Series]:
    backfill = _load_backfill_module()
    start = common_dates.min().strftime("%Y-%m-%d")
    end = (common_dates.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    prices = backfill._download_prices(["SPY", "^VIX"], start, end)
    spy = prices["SPY"]
    vix = prices["^VIX"]
    spy.index = pd.to_datetime(spy.index, utc=True).tz_convert(None)
    vix.index = pd.to_datetime(vix.index, utc=True).tz_convert(None)
    spy = spy.reindex(common_dates).dropna()
    vix = vix.reindex(common_dates).dropna()
    return {"spy": spy, "vix": vix}


def _load_cached_prices_long(cache_dir: Path) -> dict[str, pd.Series] | None:
    spy = read_series_cache(
        cache_dir, _series_long_cache_key("SPY"), max_age_hours=10**9
    )
    ief = read_series_cache(
        cache_dir, _series_long_cache_key("IEF"), max_age_hours=10**9
    )
    vix = read_series_cache(
        cache_dir, _series_long_cache_key("^VIX"), max_age_hours=10**9
    )
    if spy is None or ief is None or vix is None:
        return None
    spy.index = pd.to_datetime(spy.index, utc=True).tz_convert(None).normalize()
    ief.index = pd.to_datetime(ief.index, utc=True).tz_convert(None).normalize()
    vix.index = pd.to_datetime(vix.index, utc=True).tz_convert(None).normalize()
    spy = spy.groupby(spy.index).last()
    ief = ief.groupby(ief.index).last()
    vix = vix.groupby(vix.index).last()
    common = spy.index.intersection(ief.index).intersection(vix.index)
    if common.empty:
        return None
    spy = spy.reindex(common).dropna()
    ief = ief.reindex(common).dropna()
    vix = vix.reindex(common).dropna()
    if spy.empty or ief.empty or vix.empty:
        return None
    return {"SPY": spy, "IEF": ief, "^VIX": vix}


def _regime_breakdown(
    strategy_data: dict[str, StrategyData], config
) -> tuple[list[str], dict[str, dict[str, dict]]]:
    common_dates = _common_dates(strategy_data)
    if common_dates.empty:
        return [], {}

    market = _load_market_series(common_dates)
    spy = market["spy"]
    vix = market["vix"]
    vix_q1 = vix.quantile(0.33)
    vix_q2 = vix.quantile(0.66)

    sma200 = spy.rolling(window=200, min_periods=200).mean()
    trend_on = spy >= sma200

    regimes = {
        "vix_low": vix <= vix_q1,
        "vix_mid": (vix > vix_q1) & (vix <= vix_q2),
        "vix_high": vix > vix_q2,
        "trend_on": trend_on,
        "trend_off": ~trend_on,
    }

    breakdown: dict[str, dict[str, dict]] = {}
    for name, data in strategy_data.items():
        breakdown[name] = {}
        normalized = data.normalized.reindex(common_dates).dropna()
        peak = normalized.cummax()
        drawdown = normalized / peak - 1.0
        dd_0_5 = drawdown >= -0.05
        dd_5_15 = (drawdown < -0.05) & (drawdown >= -0.15)
        dd_15_plus = drawdown < -0.15
        dd_regimes = {
            "dd_0_5": dd_0_5,
            "dd_5_15": dd_5_15,
            "dd_15_plus": dd_15_plus,
        }

        for regime_name, mask in {**regimes, **dd_regimes}.items():
            dates = common_dates[mask.reindex(common_dates, fill_value=False)]
            metrics = _metrics_for_dates(data.daily, dates, config)
            breakdown[name][regime_name] = metrics

    return list(regimes.keys()) + list(dd_regimes.keys()), breakdown


def _rolling_summary(
    strategy_data: dict[str, StrategyData], config, window_years: int
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    window_days = window_years * TRADING_DAYS
    rf_annual = float(config.risk_free_rate) / 100.0
    rf_daily = (1.0 + rf_annual) ** (1.0 / TRADING_DAYS) - 1.0

    for name, data in strategy_data.items():
        returns = data.returns.dropna()
        normalized = data.normalized.reindex(returns.index).dropna()
        if len(returns) <= window_days:
            summary[name] = {
                "sharpe_mean": float("nan"),
                "sharpe_min": float("nan"),
                "cagr_mean": float("nan"),
                "cagr_min": float("nan"),
                "windows": 0,
            }
            continue

        sharpe_values = []
        cagr_values = []
        dates = returns.index
        for idx in range(window_days, len(returns)):
            window_returns = returns.iloc[idx - window_days : idx]
            excess = window_returns - rf_daily
            std = excess.std()
            if std > 0:
                sharpe = (excess.mean() / std) * np.sqrt(TRADING_DAYS)
                sharpe_values.append(float(sharpe))
            start_value = normalized.iloc[idx - window_days]
            end_value = normalized.iloc[idx]
            if start_value > 0 and end_value > 0:
                years = window_days / TRADING_DAYS
                cagr = (end_value / start_value) ** (1.0 / years) - 1.0
                cagr_values.append(float(cagr))

        summary[name] = {
            "sharpe_mean": float(np.mean(sharpe_values))
            if sharpe_values
            else float("nan"),
            "sharpe_min": float(np.min(sharpe_values))
            if sharpe_values
            else float("nan"),
            "cagr_mean": float(np.mean(cagr_values)) if cagr_values else float("nan"),
            "cagr_min": float(np.min(cagr_values)) if cagr_values else float("nan"),
            "windows": len(sharpe_values),
        }
    return summary


def _blend_returns(
    returns_a: pd.Series, returns_b: pd.Series, weight_a: float
) -> pd.Series:
    common = returns_a.index.intersection(returns_b.index)
    ra = returns_a.reindex(common).fillna(0.0)
    rb = returns_b.reindex(common).fillna(0.0)
    return ra * weight_a + rb * (1.0 - weight_a)


def _blend_label(weight_a: float) -> str:
    primary = int(round(weight_a * 100))
    secondary = max(0, 100 - primary)
    return f"{primary}/{secondary}"


def _blend_metrics(returns: pd.Series, config) -> dict:
    normalized = (1.0 + returns).cumprod()
    daily = [
        {
            "date": idx.strftime("%Y-%m-%d"),
            "equity": (value - 1.0) * config.paper_initial_capital,
        }
        for idx, value in normalized.items()
    ]
    return _compute_performance_metrics(
        daily,
        config.paper_initial_capital,
        config.risk_free_rate,
        config.pnl_downside_min_days,
    )


def _sweep_zscore(config, start: str, end: str) -> list[dict[str, float]]:
    backfill = _load_backfill_module()
    end_dt = pd.to_datetime(end) + pd.Timedelta(days=1)
    prices = backfill._download_prices(
        ["^SPX", "SPY", "QQQ", "^VIX", "^VVIX", "^TNX", "CLR.SI", "IEF", "RSP"],
        start,
        end_dt.strftime("%Y-%m-%d"),
    )

    cuts = [0.4, 0.5, 0.6]
    min_leverages = [0.85, 0.9, 0.95]
    vix_caps = [30, 32, 34]

    results: list[dict[str, float]] = []
    for cut in cuts:
        for min_leverage in min_leverages:
            for vix_cap in vix_caps:
                backfill.ZSCORE_CUT = cut
                backfill.ZSCORE_TREND_ON_MIN_LEVERAGE = min_leverage
                backfill.ZSCORE_TREND_OFF_VIX = vix_cap
                entries = backfill._zscore_throttle_strategy(
                    prices,
                    config.paper_initial_capital,
                    config.growth_ticker,
                )
                daily = _build_daily(entries)
                metrics = _compute_performance_metrics(
                    daily,
                    config.paper_initial_capital,
                    config.risk_free_rate,
                    config.pnl_downside_min_days,
                )
                results.append(
                    {
                        "cut": cut,
                        "min_leverage": min_leverage,
                        "vix_cap": vix_cap,
                        "sharpe": metrics.get("sharpe_ratio"),
                        "final_equity": metrics.get("normalized_end"),
                        "cagr": metrics.get("cagr"),
                        "max_dd": metrics.get("max_drawdown_pct"),
                    }
                )
    return results


def _sweep_zscore_fine(config, prices: dict[str, pd.Series]) -> list[dict[str, float]]:
    backfill = _load_backfill_module()
    cuts = [0.45, 0.48, 0.5, 0.52, 0.55]
    min_leverages = [0.85, 0.9, 0.93, 0.95]
    vix_caps = [32, 33, 34, 35, 36]
    results: list[dict[str, float]] = []
    for cut in cuts:
        for min_leverage in min_leverages:
            for vix_cap in vix_caps:
                backfill.ZSCORE_CUT = cut
                backfill.ZSCORE_TREND_ON_MIN_LEVERAGE = min_leverage
                backfill.ZSCORE_TREND_OFF_VIX = vix_cap
                entries = backfill._zscore_throttle_strategy(
                    prices,
                    config.paper_initial_capital,
                    config.growth_ticker,
                )
                daily = _build_daily(entries)
                metrics = _compute_performance_metrics(
                    daily,
                    config.paper_initial_capital,
                    config.risk_free_rate,
                    config.pnl_downside_min_days,
                )
                results.append(
                    {
                        "cut": cut,
                        "min_leverage": min_leverage,
                        "vix_cap": vix_cap,
                        "sharpe": metrics.get("sharpe_ratio"),
                        "final_equity": metrics.get("normalized_end"),
                        "cagr": metrics.get("cagr"),
                        "max_dd": metrics.get("max_drawdown_pct"),
                    }
                )
    return results


def _sweep_zscore_expanded(
    config, prices: dict[str, pd.Series]
) -> list[dict[str, float]]:
    backfill = _load_backfill_module()
    cuts = [0.46, 0.48, 0.5, 0.52, 0.54]
    min_leverages = [0.9, 0.93, 0.95]
    vix_caps = [38, 39, 40]
    results: list[dict[str, float]] = []
    for cut in cuts:
        for min_leverage in min_leverages:
            for vix_cap in vix_caps:
                backfill.ZSCORE_CUT = cut
                backfill.ZSCORE_TREND_ON_MIN_LEVERAGE = min_leverage
                backfill.ZSCORE_TREND_OFF_VIX = vix_cap
                entries = backfill._zscore_throttle_strategy(
                    prices,
                    config.paper_initial_capital,
                    config.growth_ticker,
                )
                daily = _build_daily(entries)
                metrics = _compute_performance_metrics(
                    daily,
                    config.paper_initial_capital,
                    config.risk_free_rate,
                    config.pnl_downside_min_days,
                )
                results.append(
                    {
                        "cut": cut,
                        "min_leverage": min_leverage,
                        "vix_cap": vix_cap,
                        "sharpe": metrics.get("sharpe_ratio"),
                        "final_equity": metrics.get("normalized_end"),
                        "cagr": metrics.get("cagr"),
                        "max_dd": metrics.get("max_drawdown_pct"),
                    }
                )
    return results


def _sweep_zscore_cached_long(config) -> tuple[list[dict[str, float]], str | None]:
    cache_dir = Path(config.cache_dir)
    prices = _load_cached_prices_long(cache_dir)
    if prices is None:
        return [], "Long-history cache missing for SPY/IEF/^VIX; cached sweep skipped."
    return _sweep_zscore_fine(config, prices), None


def _top_sweep_row(rows: list[dict[str, float]]) -> dict[str, float] | None:
    if not rows:
        return None
    return sorted(
        rows,
        key=lambda row: (
            row.get("sharpe") if row.get("sharpe") is not None else -999,
            row.get("final_equity") if row.get("final_equity") is not None else -999,
        ),
        reverse=True,
    )[0]


def _load_entries(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _daily_positions(entries: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for entry in entries:
        if entry.get("action") != "PAPER_PNL":
            continue
        timestamp = entry.get("timestamp")
        if not isinstance(timestamp, str):
            continue
        date_key = timestamp.split(" ", 1)[0]
        details = entry.get("details", {})
        if not isinstance(details, dict):
            continue
        rows.append(
            {
                "date": date_key,
                "ticker": entry.get("ticker"),
                "quantity": details.get("quantity"),
                "mark_price": details.get("mark_price"),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["date", "ticker", "quantity", "mark_price"])
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None).dt.normalize()
    df["quantity"] = df["quantity"].astype(float)
    df["mark_price"] = df["mark_price"].astype(float)
    df = df.sort_values(["date", "ticker"]).groupby(["date", "ticker"]).last()
    df = df.reset_index()
    return df


def _equity_series(daily: list[dict[str, object]]) -> pd.Series:
    if not daily:
        return pd.Series(dtype=float)
    dates = [row["date"] for row in daily]
    values = [row["equity"] for row in daily]
    idx = pd.to_datetime(dates, utc=True).tz_convert(None)
    return pd.Series(values, index=idx).sort_index()


def _compute_turnover_series(
    entries: list[dict[str, object]],
    initial_capital: float,
) -> pd.Series:
    df = _daily_positions(entries)
    if df.empty:
        return pd.Series(dtype=float)
    df["notional"] = df["quantity"] * df["mark_price"]
    pivot = (
        df.pivot(index="date", columns="ticker", values="notional")
        .fillna(0.0)
        .sort_index()
    )
    daily = _build_daily(entries)
    equity = _equity_series(daily)
    if equity.empty:
        equity = pd.Series(
            initial_capital,
            index=pivot.index,
            dtype=float,
        )
    equity = equity.reindex(pivot.index).ffill()
    turnover = pivot.diff().abs().sum(axis=1) / equity.shift(1).replace(0.0, np.nan)
    return turnover.dropna()


def _compute_turnover_by_ticker(
    entries: list[dict[str, object]],
    initial_capital: float,
) -> pd.DataFrame:
    df = _daily_positions(entries)
    if df.empty:
        return pd.DataFrame()
    df["notional"] = df["quantity"] * df["mark_price"]
    pivot = (
        df.pivot(index="date", columns="ticker", values="notional")
        .fillna(0.0)
        .sort_index()
    )
    daily = _build_daily(entries)
    equity = _equity_series(daily)
    if equity.empty:
        equity = pd.Series(
            initial_capital,
            index=pivot.index,
            dtype=float,
        )
    equity = equity.reindex(pivot.index).ffill()
    turnover = pivot.diff().abs().div(equity.shift(1).replace(0.0, np.nan), axis=0)
    return turnover.dropna(how="all")


def _compute_weight_turnover(
    spy_weights: pd.Series, ief_weights: pd.Series
) -> pd.Series:
    if spy_weights.empty or ief_weights.empty:
        return pd.Series(dtype=float)
    common = spy_weights.index.intersection(ief_weights.index)
    spy = spy_weights.reindex(common).fillna(0.0)
    ief = ief_weights.reindex(common).fillna(0.0)
    turnover = (spy.diff().abs() + ief.diff().abs()) * 0.5
    return turnover.dropna()


def _tracking_error_metrics(
    synthetic: pd.Series,
    positions: pd.Series,
) -> dict[str, float]:
    if synthetic.empty or positions.empty:
        return {}
    common = synthetic.index.intersection(positions.index)
    syn = synthetic.reindex(common).dropna()
    pos = positions.reindex(common).dropna()
    if len(syn) < 2 or len(pos) < 2:
        return {}
    syn_ret = syn.pct_change().dropna()
    pos_ret = pos.pct_change().dropna()
    common_ret = syn_ret.index.intersection(pos_ret.index)
    syn_ret = syn_ret.reindex(common_ret)
    pos_ret = pos_ret.reindex(common_ret)
    diff = pos_ret - syn_ret
    gap = (pos - syn).abs()
    return {
        "tracking_error": float(diff.std() * np.sqrt(TRADING_DAYS)),
        "return_corr": float(pos_ret.corr(syn_ret)),
        "mean_abs_return_diff": float(diff.abs().mean()),
        "max_equity_gap": float(gap.max()),
        "max_equity_gap_pct": float((gap / syn.replace(0.0, np.nan)).max()),
    }


def _tracking_error_by_regime(
    synthetic: pd.Series,
    positions: pd.Series,
    config,
) -> dict[str, dict[str, float]]:
    if synthetic.empty or positions.empty:
        return {}
    common = synthetic.index.intersection(positions.index)
    syn = synthetic.reindex(common).dropna()
    pos = positions.reindex(common).dropna()
    if syn.empty or pos.empty:
        return {}
    spy, vix = _load_regime_market(common, config)
    common = (
        syn.index.intersection(pos.index)
        .intersection(spy.index)
        .intersection(vix.index)
    )
    if common.empty:
        return {}
    syn = syn.reindex(common)
    pos = pos.reindex(common)
    spy = spy.reindex(common)
    vix = vix.reindex(common)

    vix_q1 = vix.quantile(0.33)
    vix_q2 = vix.quantile(0.66)
    sma200 = spy.rolling(window=200, min_periods=200).mean()
    trend_on = spy >= sma200

    regimes = {
        "vix_low": vix <= vix_q1,
        "vix_mid": (vix > vix_q1) & (vix <= vix_q2),
        "vix_high": vix > vix_q2,
        "trend_on": trend_on,
        "trend_off": ~trend_on,
    }

    syn_ret = syn.pct_change().dropna()
    pos_ret = pos.pct_change().dropna()
    common_ret = syn_ret.index.intersection(pos_ret.index)
    syn_ret = syn_ret.reindex(common_ret)
    pos_ret = pos_ret.reindex(common_ret)
    diff = pos_ret - syn_ret

    results: dict[str, dict[str, float]] = {}
    for name, mask in regimes.items():
        mask_ret = mask.reindex(common_ret, fill_value=False)
        if not mask_ret.any():
            continue
        diff_reg = diff[mask_ret]
        syn_reg = syn_ret[mask_ret]
        pos_reg = pos_ret[mask_ret]
        results[name] = {
            "tracking_error": float(diff_reg.std() * np.sqrt(TRADING_DAYS)),
            "return_corr": float(pos_reg.corr(syn_reg)),
            "mean_abs_return_diff": float(diff_reg.abs().mean()),
            "count": int(mask_ret.sum()),
        }
    return results


def _top_n_rows(rows: list[dict[str, float]], n: int) -> list[dict[str, float]]:
    return sorted(
        rows,
        key=lambda row: (
            row.get("sharpe") if row.get("sharpe") is not None else -999,
            row.get("final_equity") if row.get("final_equity") is not None else -999,
        ),
        reverse=True,
    )[:n]


def _top_n_rows_by_key(
    rows: list[dict[str, float]],
    key: str,
    n: int,
) -> list[dict[str, float]]:
    return sorted(
        rows,
        key=lambda row: row.get(key) if row.get(key) is not None else -999,
        reverse=True,
    )[:n]


def _neighbor_values(values: list[float], current: float) -> list[tuple[str, float]]:
    uniq = sorted(set(values))
    if current not in uniq:
        return []
    idx = uniq.index(current)
    neighbors: list[tuple[str, float]] = []
    if idx > 0:
        neighbors.append(("prev", uniq[idx - 1]))
    if idx + 1 < len(uniq):
        neighbors.append(("next", uniq[idx + 1]))
    return neighbors


def _grid_lookup(
    rows: list[dict[str, float]],
    keys: list[str],
) -> dict[tuple[float, ...], dict[str, float]]:
    lookup: dict[tuple[float, ...], dict[str, float]] = {}
    for row in rows:
        lookup[tuple(row.get(key) for key in keys)] = row
    return lookup


def _local_neighborhood_rows(
    base: dict[str, float],
    keys: list[str],
    values_by_key: dict[str, list[float]],
    lookup: dict[tuple[float, ...], dict[str, float]],
) -> list[tuple[str, dict[str, float]]]:
    variants: list[tuple[str, dict[str, float]]] = []
    base_key = tuple(base.get(key) for key in keys)
    base_row = lookup.get(base_key, base)
    variants.append(("base", base_row))
    for key_idx, key in enumerate(keys):
        current = base.get(key)
        for direction, value in _neighbor_values(values_by_key[key], float(current)):
            candidate = list(base_key)
            candidate[key_idx] = value
            row = lookup.get(tuple(candidate))
            if row is None:
                continue
            variants.append((f"{key}:{direction}", row))
    seen = set()
    deduped: list[tuple[str, dict[str, float]]] = []
    for label, row in variants:
        key = tuple(row.get(k) for k in keys)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((label, row))
    return deduped


def _sweep_dynamic_ief_grid(
    config,
    prices: dict[str, pd.Series],
    cuts: list[float],
    min_leverages: list[float],
    vix_caps: list[float],
    boosts: list[float],
    thresholds: list[float],
    lags: list[int],
) -> list[dict[str, float]]:
    spy = prices["SPY"].copy()
    ief = prices["IEF"].copy()
    spy.index = pd.to_datetime(spy.index, utc=True).tz_convert(None)
    ief.index = pd.to_datetime(ief.index, utc=True).tz_convert(None)
    results: list[dict[str, float]] = []
    for cut in cuts:
        for min_lev in min_leverages:
            for vix_cap in vix_caps:
                for boost in boosts:
                    for threshold in thresholds:
                        normalized, spy_weights, ief_weights = _dynamic_ief_series(
                            prices,
                            cut=cut,
                            min_leverage=min_lev,
                            vix_cap=vix_cap,
                            ief_boost=boost,
                            vix_threshold=threshold,
                        )
                        if normalized.empty:
                            continue
                        for lag in lags:
                            if lag == 0:
                                lagged = normalized
                            else:
                                lagged = _simulate_weighted_normalized(
                                    spy,
                                    ief,
                                    spy_weights,
                                    ief_weights,
                                    lag=lag,
                                )
                            if lagged.empty:
                                continue
                            daily = [
                                {
                                    "date": idx.strftime("%Y-%m-%d"),
                                    "equity": (value - 1.0)
                                    * config.paper_initial_capital,
                                }
                                for idx, value in lagged.items()
                            ]
                            metrics = _compute_performance_metrics(
                                daily,
                                config.paper_initial_capital,
                                config.risk_free_rate,
                                config.pnl_downside_min_days,
                            )
                            sharpe = metrics.get("sharpe_ratio")
                            max_dd = metrics.get("max_drawdown_pct")
                            score = None
                            if sharpe is not None:
                                score = sharpe
                                if max_dd is not None:
                                    score -= 0.25 * abs(max_dd)
                            results.append(
                                {
                                    "cut": cut,
                                    "min_leverage": min_lev,
                                    "vix_cap": vix_cap,
                                    "ief_boost": boost,
                                    "vix_threshold": threshold,
                                    "lag": float(lag),
                                    "sharpe": sharpe,
                                    "final_equity": metrics.get("normalized_end"),
                                    "cagr": metrics.get("cagr"),
                                    "vol": metrics.get("annualized_volatility"),
                                    "max_dd": max_dd,
                                    "score": score,
                                }
                            )
    return results


def _sweep_dynamic_ief_positions_grid(
    config,
    prices: dict[str, pd.Series],
    cuts: list[float],
    min_leverages: list[float],
    vix_caps: list[float],
    boosts: list[float],
    thresholds: list[float],
    lags: list[int],
    tag_prefix: str,
) -> list[dict[str, float]]:
    backfill = _load_backfill_module()
    results: list[dict[str, float]] = []
    for cut in cuts:
        for min_lev in min_leverages:
            for vix_cap in vix_caps:
                for boost in boosts:
                    for threshold in thresholds:
                        backfill.DYNAMIC_IEF_CUT = float(cut)
                        backfill.DYNAMIC_IEF_MIN_LEVERAGE = float(min_lev)
                        backfill.DYNAMIC_IEF_VIX_CAP = float(vix_cap)
                        backfill.DYNAMIC_IEF_BOOST = float(boost)
                        backfill.DYNAMIC_IEF_VIX_THRESHOLD = float(threshold)
                        for lag in lags:
                            entries = backfill._dynamic_ief_strategy(
                                prices,
                                config.paper_initial_capital,
                                config.growth_ticker,
                                lag_days=int(lag),
                                tag_suffix=f"{tag_prefix}_{lag}",
                            )
                            daily = _build_daily(entries)
                            metrics = _compute_performance_metrics(
                                daily,
                                config.paper_initial_capital,
                                config.risk_free_rate,
                                config.pnl_downside_min_days,
                            )
                            results.append(
                                {
                                    "cut": cut,
                                    "min_leverage": min_lev,
                                    "vix_cap": vix_cap,
                                    "ief_boost": boost,
                                    "vix_threshold": threshold,
                                    "lag": float(lag),
                                    "sharpe": metrics.get("sharpe_ratio"),
                                    "final_equity": metrics.get("normalized_end"),
                                    "cagr": metrics.get("cagr"),
                                    "vol": metrics.get("annualized_volatility"),
                                    "max_dd": metrics.get("max_drawdown_pct"),
                                }
                            )
    return results


def _sweep_zscore_grid(
    config,
    prices: dict[str, pd.Series],
    cuts: list[float],
    min_leverages: list[float],
    vix_caps: list[float],
    max_offs: list[float],
) -> list[dict[str, float]]:
    backfill = _load_backfill_module()
    results: list[dict[str, float]] = []
    for cut in cuts:
        for min_lev in min_leverages:
            for vix_cap in vix_caps:
                for max_off in max_offs:
                    backfill.ZSCORE_CUT = cut
                    backfill.ZSCORE_TREND_ON_MIN_LEVERAGE = min_lev
                    backfill.ZSCORE_TREND_OFF_VIX = vix_cap
                    backfill.ZSCORE_TREND_OFF_MAX_LEVERAGE = max_off
                    entries = backfill._zscore_throttle_strategy(
                        prices,
                        config.paper_initial_capital,
                        config.growth_ticker,
                    )
                    daily = _build_daily(entries)
                    metrics = _compute_performance_metrics(
                        daily,
                        config.paper_initial_capital,
                        config.risk_free_rate,
                        config.pnl_downside_min_days,
                    )
                    results.append(
                        {
                            "cut": cut,
                            "min_leverage": min_lev,
                            "vix_cap": vix_cap,
                            "max_off": max_off,
                            "sharpe": metrics.get("sharpe_ratio"),
                            "final_equity": metrics.get("normalized_end"),
                            "cagr": metrics.get("cagr"),
                            "vol": metrics.get("annualized_volatility"),
                            "max_dd": metrics.get("max_drawdown_pct"),
                        }
                    )
    return results


def _positions_weights_series(
    entries: list[dict[str, object]],
) -> tuple[pd.Series, pd.Series]:
    df = _daily_positions(entries)
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    df["notional"] = df["quantity"] * df["mark_price"]
    pivot = (
        df.pivot(index="date", columns="ticker", values="notional")
        .fillna(0.0)
        .sort_index()
    )
    total = pivot.abs().sum(axis=1).replace(0.0, np.nan)
    spy = pivot.get("SPY", 0.0) / total
    ief = pivot.get("IEF", 0.0) / total
    return spy.dropna(), ief.dropna()


def _positions_weights_frame(entries: list[dict[str, object]]) -> pd.DataFrame:
    df = _daily_positions(entries)
    if df.empty:
        return pd.DataFrame()
    df["notional"] = df["quantity"] * df["mark_price"]
    pivot = (
        df.pivot(index="date", columns="ticker", values="notional")
        .fillna(0.0)
        .sort_index()
    )
    total = pivot.abs().sum(axis=1).replace(0.0, np.nan)
    weights = pivot.div(total, axis=0)
    return weights.dropna(how="all")


def _concentration_stats(entries: list[dict[str, object]]) -> dict[str, float]:
    weights = _positions_weights_frame(entries)
    if weights.empty:
        return {}
    abs_weights = weights.abs()
    hhi = (abs_weights**2).sum(axis=1)
    top1 = abs_weights.max(axis=1)
    top2 = abs_weights.apply(lambda row: row.nlargest(2).sum(), axis=1)
    effective_n = 1.0 / hhi.replace(0.0, np.nan)
    return {
        "hhi_mean": float(hhi.mean()),
        "hhi_median": float(hhi.median()),
        "hhi_p95": float(hhi.quantile(0.95)),
        "top1_mean": float(top1.mean()),
        "top1_p95": float(top1.quantile(0.95)),
        "top2_mean": float(top2.mean()),
        "effective_n_mean": float(effective_n.mean()),
        "effective_n_median": float(effective_n.median()),
        "pct_top1_gt_50": float((top1 > 0.5).mean()),
    }


def _equity_gap_by_regime(
    synthetic: pd.Series,
    positions: pd.Series,
    config,
) -> dict[str, dict[str, float]]:
    if synthetic.empty or positions.empty:
        return {}
    common = synthetic.index.intersection(positions.index)
    syn = synthetic.reindex(common).dropna()
    pos = positions.reindex(common).dropna()
    if syn.empty or pos.empty:
        return {}
    spy, vix = _load_regime_market(common, config)
    common = (
        syn.index.intersection(pos.index)
        .intersection(spy.index)
        .intersection(vix.index)
    )
    if common.empty:
        return {}
    syn = syn.reindex(common)
    pos = pos.reindex(common)
    spy = spy.reindex(common)
    vix = vix.reindex(common)

    vix_q1 = vix.quantile(0.33)
    vix_q2 = vix.quantile(0.66)
    sma200 = spy.rolling(window=200, min_periods=200).mean()
    trend_on = spy >= sma200
    regimes = {
        "vix_low": vix <= vix_q1,
        "vix_mid": (vix > vix_q1) & (vix <= vix_q2),
        "vix_high": vix > vix_q2,
        "trend_on": trend_on,
        "trend_off": ~trend_on,
    }

    gap_pct = (pos - syn) / syn.replace(0.0, np.nan)
    results: dict[str, dict[str, float]] = {}
    for name, mask in regimes.items():
        mask_gap = mask.reindex(common, fill_value=False)
        series = gap_pct[mask_gap].dropna()
        if series.empty:
            continue
        abs_gap = series.abs()
        results[name] = {
            "mean_abs_gap": float(abs_gap.mean()),
            "median_abs_gap": float(abs_gap.median()),
            "max_abs_gap": float(abs_gap.max()),
            "count": int(mask_gap.sum()),
        }
    return results


def _tracking_error_by_regime_table(
    rows: dict[int, dict[str, dict[str, float]]],
    regimes: tuple[str, ...] = (
        "vix_low",
        "vix_mid",
        "vix_high",
        "trend_on",
        "trend_off",
    ),
) -> list[dict[str, float]]:
    table = []
    for lag, metrics in rows.items():
        row = {"lag": float(lag)}
        for regime in regimes:
            value = metrics.get(regime, {}).get("tracking_error")
            row[regime] = value
        table.append(row)
    return table


def _return_gap_by_regime(
    synthetic: pd.Series,
    positions: pd.Series,
    syn_spy_weights: pd.Series,
    syn_ief_weights: pd.Series,
    positions_entries: list[dict[str, object]],
    config,
) -> dict[str, dict[str, float]]:
    if synthetic.empty or positions.empty:
        return {}
    cache_dir = Path(config.cache_dir)
    cached = _load_cached_prices_long(cache_dir)
    if cached is None:
        prices = _load_backfill_module()._download_prices(
            ["SPY", "IEF", "^VIX"],
            synthetic.index.min().strftime("%Y-%m-%d"),
            (synthetic.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        )
        spy = prices["SPY"]
        ief = prices["IEF"]
        vix = prices["^VIX"]
    else:
        spy = cached["SPY"].copy()
        ief = cached["IEF"].copy()
        vix = cached["^VIX"].copy()

    spy.index = pd.to_datetime(spy.index, utc=True).tz_convert(None)
    ief.index = pd.to_datetime(ief.index, utc=True).tz_convert(None)
    vix.index = pd.to_datetime(vix.index, utc=True).tz_convert(None)

    syn_spy = syn_spy_weights.copy()
    syn_ief = syn_ief_weights.copy()
    pos_spy, pos_ief = _positions_weights_series(positions_entries)

    common = syn_spy.index.intersection(syn_ief.index)
    common = common.intersection(pos_spy.index).intersection(pos_ief.index)
    common = (
        common.intersection(spy.index).intersection(ief.index).intersection(vix.index)
    )
    if common.empty:
        return {}

    spy = spy.reindex(common)
    ief = ief.reindex(common)
    vix = vix.reindex(common)
    syn_spy = syn_spy.reindex(common)
    syn_ief = syn_ief.reindex(common)
    pos_spy = pos_spy.reindex(common)
    pos_ief = pos_ief.reindex(common)

    spy_ret = spy.pct_change().dropna()
    ief_ret = ief.pct_change().dropna()
    common_ret = spy_ret.index.intersection(ief_ret.index)
    syn_spy = syn_spy.reindex(common_ret)
    syn_ief = syn_ief.reindex(common_ret)
    pos_spy = pos_spy.reindex(common_ret)
    pos_ief = pos_ief.reindex(common_ret)
    vix = vix.reindex(common_ret)
    spy = spy.reindex(common_ret)

    syn_spy_contrib = syn_spy * spy_ret
    syn_ief_contrib = syn_ief * ief_ret
    pos_spy_contrib = pos_spy * spy_ret
    pos_ief_contrib = pos_ief * ief_ret

    vix_q1 = vix.quantile(0.33)
    vix_q2 = vix.quantile(0.66)
    sma200 = spy.rolling(window=200, min_periods=200).mean()
    trend_on = spy >= sma200
    regimes = {
        "vix_low": vix <= vix_q1,
        "vix_mid": (vix > vix_q1) & (vix <= vix_q2),
        "vix_high": vix > vix_q2,
        "trend_on": trend_on,
        "trend_off": ~trend_on,
    }

    results: dict[str, dict[str, float]] = {}
    for name, mask in regimes.items():
        mask_ret = mask.reindex(common_ret, fill_value=False)
        if not mask_ret.any():
            continue
        spy_gap = pos_spy_contrib[mask_ret] - syn_spy_contrib[mask_ret]
        ief_gap = pos_ief_contrib[mask_ret] - syn_ief_contrib[mask_ret]
        total_gap = spy_gap + ief_gap
        results[name] = {
            "spy_gap": float(spy_gap.mean()),
            "ief_gap": float(ief_gap.mean()),
            "total_gap": float(total_gap.mean()),
            "count": int(mask_ret.sum()),
        }
    return results


def _signal_slippage_stats(
    syn_spy_weights: pd.Series,
    syn_ief_weights: pd.Series,
    positions_entries: list[dict[str, object]],
) -> dict[str, float]:
    pos_spy, pos_ief = _positions_weights_series(positions_entries)
    if pos_spy.empty or pos_ief.empty:
        return {}
    common = syn_spy_weights.index.intersection(syn_ief_weights.index)
    common = common.intersection(pos_spy.index).intersection(pos_ief.index)
    if common.empty:
        return {}
    syn_spy = syn_spy_weights.reindex(common)
    syn_ief = syn_ief_weights.reindex(common)
    pos_spy = pos_spy.reindex(common)
    pos_ief = pos_ief.reindex(common)
    gap = (syn_spy - pos_spy).abs() + (syn_ief - pos_ief).abs()
    gap = gap * 0.5
    return {
        "avg_abs_gap": float(gap.mean()),
        "median_abs_gap": float(gap.median()),
        "p95_abs_gap": float(gap.quantile(0.95)),
        "pct_gt_5": float((gap > 0.05).mean()),
    }


def _normalize_weights(
    spy_weights: pd.Series, ief_weights: pd.Series
) -> tuple[pd.Series, pd.Series]:
    total = (spy_weights + ief_weights).replace(0.0, np.nan)
    return spy_weights / total, ief_weights / total


def _simulate_weighted_normalized(
    spy: pd.Series,
    ief: pd.Series,
    spy_weights: pd.Series,
    ief_weights: pd.Series,
    lag: int = 0,
) -> pd.Series:
    common = spy.index.intersection(ief.index)
    spy = spy.reindex(common)
    ief = ief.reindex(common)
    spy_ret = spy.pct_change().dropna()
    ief_ret = ief.pct_change().dropna()
    common_ret = spy_ret.index.intersection(ief_ret.index)
    spy_ret = spy_ret.reindex(common_ret)
    ief_ret = ief_ret.reindex(common_ret)
    spy_w = spy_weights.shift(lag).reindex(common_ret)
    ief_w = ief_weights.shift(lag).reindex(common_ret)
    spy_w, ief_w = _normalize_weights(spy_w, ief_w)
    valid = spy_w.notna() & ief_w.notna()
    if not valid.any():
        return pd.Series(dtype=float)
    spy_ret = spy_ret[valid]
    ief_ret = ief_ret[valid]
    spy_w = spy_w[valid]
    ief_w = ief_w[valid]
    portfolio = spy_w * spy_ret + ief_w * ief_ret
    return (1.0 + portfolio).cumprod()


def _equity_gap_monthly_by_regime(
    synthetic: pd.Series,
    positions: pd.Series,
    config,
) -> dict[str, pd.DataFrame]:
    if synthetic.empty or positions.empty:
        return {}
    common = synthetic.index.intersection(positions.index)
    syn = synthetic.reindex(common).dropna()
    pos = positions.reindex(common).dropna()
    if syn.empty or pos.empty:
        return {}
    spy, vix = _load_regime_market(common, config)
    common = (
        syn.index.intersection(pos.index)
        .intersection(spy.index)
        .intersection(vix.index)
    )
    if common.empty:
        return {}
    syn = syn.reindex(common)
    pos = pos.reindex(common)
    spy = spy.reindex(common)
    vix = vix.reindex(common)

    vix_q1 = vix.quantile(0.33)
    vix_q2 = vix.quantile(0.66)
    sma200 = spy.rolling(window=200, min_periods=200).mean()
    trend_on = spy >= sma200
    regimes = {
        "vix_low": vix <= vix_q1,
        "vix_mid": (vix > vix_q1) & (vix <= vix_q2),
        "vix_high": vix > vix_q2,
        "trend_on": trend_on,
        "trend_off": ~trend_on,
    }

    gap_pct = (pos - syn).abs() / syn.replace(0.0, np.nan)
    results: dict[str, pd.DataFrame] = {}
    for name, mask in regimes.items():
        mask_gap = mask.reindex(common, fill_value=False)
        series = gap_pct[mask_gap].dropna()
        if series.empty:
            continue
        df = pd.DataFrame({"gap": series})
        df["month"] = df.index.strftime("%Y-%m")
        grouped = df.groupby("month")["gap"].agg(
            mean="mean",
            median="median",
            max="max",
            count="count",
        )
        results[name] = grouped.reset_index()
    return results


def _signal_slippage_spikes(
    syn_spy_weights: pd.Series,
    syn_ief_weights: pd.Series,
    positions_entries: list[dict[str, object]],
    config,
    top_n: int = 10,
) -> list[dict[str, object]]:
    pos_spy, pos_ief = _positions_weights_series(positions_entries)
    if pos_spy.empty or pos_ief.empty:
        return []
    common = syn_spy_weights.index.intersection(syn_ief_weights.index)
    common = common.intersection(pos_spy.index).intersection(pos_ief.index)
    if common.empty:
        return []

    syn_spy = syn_spy_weights.reindex(common)
    syn_ief = syn_ief_weights.reindex(common)
    pos_spy = pos_spy.reindex(common)
    pos_ief = pos_ief.reindex(common)
    gap = ((syn_spy - pos_spy).abs() + (syn_ief - pos_ief).abs()) * 0.5
    gap = gap.dropna()
    if gap.empty:
        return []

    spy, vix = _load_regime_market(gap.index, config)
    common = gap.index.intersection(spy.index).intersection(vix.index)
    gap = gap.reindex(common)
    syn_spy = syn_spy.reindex(common)
    syn_ief = syn_ief.reindex(common)
    pos_spy = pos_spy.reindex(common)
    pos_ief = pos_ief.reindex(common)
    vix = vix.reindex(common)
    sma200 = spy.reindex(common).rolling(window=200, min_periods=200).mean()
    trend_on = spy.reindex(common) >= sma200

    top = gap.sort_values(ascending=False).head(top_n)
    spikes: list[dict[str, object]] = []
    for day, value in top.items():
        spikes.append(
            {
                "date": day.strftime("%Y-%m-%d"),
                "gap": float(value),
                "spy_target": float(syn_spy.loc[day]),
                "spy_realized": float(pos_spy.loc[day]),
                "ief_target": float(syn_ief.loc[day]),
                "ief_realized": float(pos_ief.loc[day]),
                "vix": float(vix.loc[day]),
                "trend_on": bool(trend_on.loc[day]),
            }
        )
    return spikes


def _turnover_by_month(turnover: pd.Series) -> pd.DataFrame:
    if turnover.empty:
        return pd.DataFrame()
    df = pd.DataFrame({"turnover": turnover})
    df["month"] = df.index.strftime("%Y-%m")
    grouped = df.groupby("month")["turnover"].agg(
        mean="mean",
        median="median",
        max="max",
        count="count",
    )
    grouped["pct_gt_10"] = df.groupby("month")["turnover"].apply(
        lambda x: (x > 0.10).mean()
    )
    return grouped.reset_index()


def _execution_parity_stats(
    syn_ief: pd.Series,
    pos_ief: pd.Series,
    vix: pd.Series,
    vix_threshold: float,
    max_lag: int = 5,
    min_target: float = 0.1,
    tolerance: float = 0.02,
) -> list[dict[str, float]]:
    common = syn_ief.index.intersection(pos_ief.index).intersection(vix.index)
    if common.empty:
        return []
    syn_ief = syn_ief.reindex(common)
    pos_ief = pos_ief.reindex(common)
    vix = vix.reindex(common)
    trigger = (vix >= vix_threshold) & (syn_ief >= min_target)
    if not trigger.any():
        return []
    results = []
    for lag in range(max_lag + 1):
        realized = pos_ief.shift(-lag)
        gap = (syn_ief - realized).abs()
        mask = trigger & realized.notna()
        if not mask.any():
            continue
        results.append(
            {
                "lag": float(lag),
                "mean_abs_gap": float(gap[mask].mean()),
                "pct_within": float((gap[mask] <= tolerance).mean()),
                "avg_target": float(syn_ief[mask].mean()),
                "avg_realized": float(realized[mask].mean()),
                "count": float(mask.sum()),
            }
        )
    return results


def _return_gap_decomposition(
    syn_spy: pd.Series,
    syn_ief: pd.Series,
    pos_spy: pd.Series,
    pos_ief: pd.Series,
    spy: pd.Series,
    ief: pd.Series,
) -> dict[str, float]:
    common = syn_spy.index.intersection(syn_ief.index)
    common = common.intersection(pos_spy.index).intersection(pos_ief.index)
    common = common.intersection(spy.index).intersection(ief.index)
    if common.empty:
        return {}
    syn_spy = syn_spy.reindex(common)
    syn_ief = syn_ief.reindex(common)
    pos_spy = pos_spy.reindex(common)
    pos_ief = pos_ief.reindex(common)
    spy = spy.reindex(common)
    ief = ief.reindex(common)
    spy_ret = spy.pct_change().dropna()
    ief_ret = ief.pct_change().dropna()
    common_ret = spy_ret.index.intersection(ief_ret.index)
    syn_spy = syn_spy.reindex(common_ret)
    syn_ief = syn_ief.reindex(common_ret)
    pos_spy = pos_spy.reindex(common_ret)
    pos_ief = pos_ief.reindex(common_ret)
    spy_ret = spy_ret.reindex(common_ret)
    ief_ret = ief_ret.reindex(common_ret)

    spy_gap = (pos_spy - syn_spy) * spy_ret
    ief_gap = (pos_ief - syn_ief) * ief_ret
    total_gap = spy_gap + ief_gap
    return {
        "spy_gap_mean": float(spy_gap.mean()),
        "ief_gap_mean": float(ief_gap.mean()),
        "total_gap_mean": float(total_gap.mean()),
        "spy_gap_cum": float(spy_gap.sum()),
        "ief_gap_cum": float(ief_gap.sum()),
        "total_gap_cum": float(total_gap.sum()),
    }


def _event_study_vix_cross(
    syn: pd.Series,
    pos: pd.Series,
    vix: pd.Series,
    threshold: float,
    horizons: tuple[int, ...] = (5, 10, 20),
) -> dict[str, float]:
    common = syn.index.intersection(pos.index).intersection(vix.index)
    if common.empty:
        return {}
    syn = syn.reindex(common)
    pos = pos.reindex(common)
    vix = vix.reindex(common)
    cross = (vix >= threshold) & (vix.shift(1) < threshold)
    events = cross[cross].index
    if len(events) == 0:
        return {}

    results: dict[str, float] = {
        "events": float(len(events)),
    }
    for horizon in horizons:
        syn_returns = []
        pos_returns = []
        for day in events:
            if day not in syn.index or day not in pos.index:
                continue
            end = day + pd.Timedelta(days=horizon)
            syn_slice = syn.loc[day:end]
            pos_slice = pos.loc[day:end]
            if len(syn_slice) < 2 or len(pos_slice) < 2:
                continue
            syn_returns.append(float(syn_slice.iloc[-1] / syn_slice.iloc[0] - 1.0))
            pos_returns.append(float(pos_slice.iloc[-1] / pos_slice.iloc[0] - 1.0))
        if syn_returns:
            syn_mean = float(np.mean(syn_returns))
            pos_mean = float(np.mean(pos_returns))
            results[f"syn_{horizon}"] = syn_mean
            results[f"pos_{horizon}"] = pos_mean
            results[f"gap_{horizon}"] = pos_mean - syn_mean
    return results


def _event_study_trend_flip(
    syn: pd.Series,
    pos: pd.Series,
    spy: pd.Series,
    horizons: tuple[int, ...] = (5, 10, 20),
) -> dict[str, dict[str, float]]:
    common = syn.index.intersection(pos.index).intersection(spy.index)
    if common.empty:
        return {}
    syn = syn.reindex(common)
    pos = pos.reindex(common)
    spy = spy.reindex(common)
    sma200 = spy.rolling(window=200, min_periods=200).mean()
    trend_on = spy >= sma200
    flips = trend_on != trend_on.shift(1)
    flip_days = flips[flips].index
    if len(flip_days) == 0:
        return {}
    results: dict[str, dict[str, float]] = {}
    for label, mask in {
        "on_to_off": ~trend_on & flips,
        "off_to_on": trend_on & flips,
    }.items():
        events = mask[mask].index
        if len(events) == 0:
            continue
        out: dict[str, float] = {"events": float(len(events))}
        for horizon in horizons:
            syn_returns = []
            pos_returns = []
            for day in events:
                end = day + pd.Timedelta(days=horizon)
                syn_slice = syn.loc[day:end]
                pos_slice = pos.loc[day:end]
                if len(syn_slice) < 2 or len(pos_slice) < 2:
                    continue
                syn_returns.append(float(syn_slice.iloc[-1] / syn_slice.iloc[0] - 1.0))
                pos_returns.append(float(pos_slice.iloc[-1] / pos_slice.iloc[0] - 1.0))
            if syn_returns:
                syn_mean = float(np.mean(syn_returns))
                pos_mean = float(np.mean(pos_returns))
                out[f"syn_{horizon}"] = syn_mean
                out[f"pos_{horizon}"] = pos_mean
                out[f"gap_{horizon}"] = pos_mean - syn_mean
        results[label] = out
    return results


def _event_study_drawdown_breach(
    syn: pd.Series,
    pos: pd.Series,
    threshold: float,
    horizons: tuple[int, ...] = (5, 10, 20),
) -> dict[str, float]:
    common = syn.index.intersection(pos.index)
    if common.empty:
        return {}
    syn = syn.reindex(common)
    pos = pos.reindex(common)
    peak = syn.cummax()
    drawdown = syn / peak - 1.0
    breach = (drawdown <= threshold) & (drawdown.shift(1) > threshold)
    events = breach[breach].index
    if len(events) == 0:
        return {}
    results: dict[str, float] = {"events": float(len(events))}
    for horizon in horizons:
        syn_returns = []
        pos_returns = []
        for day in events:
            end = day + pd.Timedelta(days=horizon)
            syn_slice = syn.loc[day:end]
            pos_slice = pos.loc[day:end]
            if len(syn_slice) < 2 or len(pos_slice) < 2:
                continue
            syn_returns.append(float(syn_slice.iloc[-1] / syn_slice.iloc[0] - 1.0))
            pos_returns.append(float(pos_slice.iloc[-1] / pos_slice.iloc[0] - 1.0))
        if syn_returns:
            syn_mean = float(np.mean(syn_returns))
            pos_mean = float(np.mean(pos_returns))
            results[f"syn_{horizon}"] = syn_mean
            results[f"pos_{horizon}"] = pos_mean
            results[f"gap_{horizon}"] = pos_mean - syn_mean
    return results


def _apply_cost_stress_test(
    daily: list[dict[str, object]],
    turnover: pd.Series,
    config,
    cost_bps: float,
) -> dict[str, float]:
    if not daily:
        return {}
    equity = _equity_series(daily)
    if equity.empty or turnover.empty:
        return {}
    turnover = turnover.reindex(equity.index).fillna(0.0)
    cost_rate = cost_bps / 10000.0
    cumulative_cost = (turnover * equity.shift(1).bfill() * cost_rate).cumsum()
    adjusted_equity = equity - cumulative_cost
    if adjusted_equity.min() <= -float(config.paper_initial_capital):
        return {
            "cost_bps": float(cost_bps),
            "invalid": True,
        }
    adjusted_daily = [
        {
            "date": idx.strftime("%Y-%m-%d"),
            "equity": float(value),
        }
        for idx, value in adjusted_equity.items()
    ]
    metrics = _compute_performance_metrics(
        adjusted_daily,
        config.paper_initial_capital,
        config.risk_free_rate,
        config.pnl_downside_min_days,
    )
    metrics["cost_bps"] = float(cost_bps)
    return metrics


def _apply_variable_cost_model(
    daily: list[dict[str, object]],
    turnover: pd.Series,
    vix: pd.Series,
    config,
    base_bps: float = 2.0,
    vix_bps_per_point: float = 0.1,
    turnover_bps_per_pct: float = 0.2,
) -> dict[str, float]:
    if not daily:
        return {}
    equity = _equity_series(daily)
    if equity.empty or turnover.empty or vix.empty:
        return {}
    common = equity.index.intersection(turnover.index).intersection(vix.index)
    if common.empty:
        return {}
    equity = equity.reindex(common)
    turnover = turnover.reindex(common).fillna(0.0)
    vix = vix.reindex(common).ffill()
    vix_component = (vix - 15.0).clip(lower=0.0) * vix_bps_per_point
    turnover_component = turnover * 100.0 * turnover_bps_per_pct
    cost_bps = base_bps + vix_component + turnover_component
    cost_rate = cost_bps / 10000.0
    cumulative_cost = (turnover * equity.shift(1).bfill() * cost_rate).cumsum()
    adjusted_equity = equity - cumulative_cost
    adjusted_daily = [
        {
            "date": idx.strftime("%Y-%m-%d"),
            "equity": float(value),
        }
        for idx, value in adjusted_equity.items()
    ]
    metrics = _compute_performance_metrics(
        adjusted_daily,
        config.paper_initial_capital,
        config.risk_free_rate,
        config.pnl_downside_min_days,
    )
    for key in ("normalized_end", "cagr", "max_drawdown_pct"):
        value = metrics.get(key)
        if isinstance(value, complex) or (value is not None and not np.isfinite(value)):
            return {
                "base_bps": float(base_bps),
                "vix_bps_per_point": float(vix_bps_per_point),
                "turnover_bps_per_pct": float(turnover_bps_per_pct),
                "avg_cost_bps": float(cost_bps.mean()),
                "invalid": True,
            }
    metrics["base_bps"] = float(base_bps)
    metrics["vix_bps_per_point"] = float(vix_bps_per_point)
    metrics["turnover_bps_per_pct"] = float(turnover_bps_per_pct)
    metrics["avg_cost_bps"] = float(cost_bps.mean())
    return metrics


def _load_regime_market(common_dates: pd.Index, config) -> tuple[pd.Series, pd.Series]:
    cache_dir = Path(config.cache_dir)
    cached = _load_cached_prices_long(cache_dir)
    if cached is not None:
        spy = cached["SPY"].copy()
        vix = cached["^VIX"].copy()
    else:
        market = _load_market_series(common_dates)
        spy = market["spy"].copy()
        vix = market["vix"].copy()
    spy.index = pd.to_datetime(spy.index, utc=True).tz_convert(None)
    vix.index = pd.to_datetime(vix.index, utc=True).tz_convert(None)
    spy = spy.reindex(common_dates).dropna()
    vix = vix.reindex(common_dates).dropna()
    common = spy.index.intersection(vix.index)
    return spy.reindex(common), vix.reindex(common)


def _regime_breakdown_single(
    normalized: pd.Series,
    daily: list[dict[str, object]],
    config,
) -> dict[str, dict[str, float]]:
    if normalized.empty:
        return {}
    spy, vix = _load_regime_market(normalized.index, config)
    common = normalized.index.intersection(spy.index).intersection(vix.index)
    if common.empty:
        return {}
    vix = vix.reindex(common)
    spy = spy.reindex(common)
    normalized = normalized.reindex(common)
    vix_q1 = vix.quantile(0.33)
    vix_q2 = vix.quantile(0.66)
    sma200 = spy.rolling(window=200, min_periods=200).mean()
    trend_on = spy >= sma200

    regimes = {
        "vix_low": vix <= vix_q1,
        "vix_mid": (vix > vix_q1) & (vix <= vix_q2),
        "vix_high": vix > vix_q2,
        "trend_on": trend_on,
        "trend_off": ~trend_on,
    }
    peak = normalized.cummax()
    drawdown = normalized / peak - 1.0
    dd_regimes = {
        "dd_0_5": drawdown >= -0.05,
        "dd_5_15": (drawdown < -0.05) & (drawdown >= -0.15),
        "dd_15_plus": drawdown < -0.15,
    }

    results: dict[str, dict[str, float]] = {}
    for name, mask in {**regimes, **dd_regimes}.items():
        dates = common[mask.reindex(common, fill_value=False)]
        results[name] = _metrics_for_dates(daily, dates, config)
    return results


def _regime_breakdown_vix_quantiles(
    normalized: pd.Series,
    daily: list[dict[str, object]],
    config,
    quantiles: tuple[float, ...] = (0.25, 0.5, 0.75, 0.9),
) -> dict[str, dict[str, float]]:
    if normalized.empty:
        return {}
    spy, vix = _load_regime_market(normalized.index, config)
    common = normalized.index.intersection(spy.index).intersection(vix.index)
    if common.empty:
        return {}
    vix = vix.reindex(common)
    normalized = normalized.reindex(common)
    bounds = vix.quantile(list(quantiles)).to_list()
    buckets = [float("-inf")] + bounds + [float("inf")]
    labels = []
    for idx in range(len(buckets) - 1):
        low = buckets[idx]
        high = buckets[idx + 1]
        if np.isfinite(low) and np.isfinite(high):
            labels.append(f"vix_{int(low)}_{int(high)}")
        elif np.isfinite(high):
            labels.append(f"vix_le_{int(high)}")
        else:
            labels.append(f"vix_ge_{int(low)}")
    vix_bucket = pd.cut(vix, bins=buckets, labels=labels, include_lowest=True)
    results: dict[str, dict[str, float]] = {}
    for label in labels:
        mask = vix_bucket == label
        dates = common[mask.reindex(common, fill_value=False)]
        if len(dates) == 0:
            continue
        results[label] = _metrics_for_dates(daily, dates, config)
        results[label]["points"] = float(len(dates))
    return results


def _compute_exposure_stats(
    entries: list[dict[str, object]], initial_capital: float
) -> dict[str, float]:
    df = _daily_positions(entries)
    if df.empty:
        return {}
    df["notional"] = df["quantity"] * df["mark_price"]
    pivot = df.pivot(index="date", columns="ticker", values="notional").fillna(0.0)
    total_notional = pivot.abs().sum(axis=1)
    equity = initial_capital + pivot.sum(axis=1) - pivot.sum(axis=1).iloc[0]
    leverage = total_notional / equity.replace(0.0, np.nan)
    spy_weight = pivot.get("SPY", 0.0) / total_notional.replace(0.0, np.nan)
    ief_weight = pivot.get("IEF", 0.0) / total_notional.replace(0.0, np.nan)
    return {
        "avg_leverage": float(leverage.mean(skipna=True)),
        "median_leverage": float(leverage.median(skipna=True)),
        "pct_leverage_lt_1": float((leverage < 1.0).mean(skipna=True)),
        "avg_spy_weight": float(spy_weight.mean(skipna=True)),
        "avg_ief_weight": float(ief_weight.mean(skipna=True)),
    }


def _compute_leverage_buckets(
    entries: list[dict[str, object]], initial_capital: float
) -> dict[str, float]:
    df = _daily_positions(entries)
    if df.empty:
        return {}
    df["notional"] = df["quantity"] * df["mark_price"]
    pivot = df.pivot(index="date", columns="ticker", values="notional").fillna(0.0)
    total_notional = pivot.abs().sum(axis=1)
    equity = initial_capital + pivot.sum(axis=1) - pivot.sum(axis=1).iloc[0]
    leverage = total_notional / equity.replace(0.0, np.nan)
    buckets = {
        "lt_0_5": (leverage < 0.5).mean(skipna=True),
        "0_5_0_8": ((leverage >= 0.5) & (leverage < 0.8)).mean(skipna=True),
        "0_8_1_0": ((leverage >= 0.8) & (leverage < 1.0)).mean(skipna=True),
        "1_0_1_2": ((leverage >= 1.0) & (leverage < 1.2)).mean(skipna=True),
        "ge_1_2": (leverage >= 1.2).mean(skipna=True),
    }
    return {key: float(value) for key, value in buckets.items()}


def _compute_leverage_series(
    entries: list[dict[str, object]], initial_capital: float
) -> pd.Series:
    df = _daily_positions(entries)
    if df.empty:
        return pd.Series(dtype=float)
    df["notional"] = df["quantity"] * df["mark_price"]
    pivot = df.pivot(index="date", columns="ticker", values="notional").fillna(0.0)
    total_notional = pivot.abs().sum(axis=1)
    equity = initial_capital + pivot.sum(axis=1) - pivot.sum(axis=1).iloc[0]
    leverage = total_notional / equity.replace(0.0, np.nan)
    return leverage


def _compute_contribution_stats(entries: list[dict[str, object]]) -> dict[str, float]:
    df = _daily_positions(entries)
    if df.empty:
        return {}
    df = df.sort_values(["ticker", "date"])
    df["prev_price"] = df.groupby("ticker")["mark_price"].shift(1)
    df["prev_qty"] = df.groupby("ticker")["quantity"].shift(1)
    df = df.dropna(subset=["prev_price", "prev_qty"])
    df["pnl"] = df["prev_qty"] * (df["mark_price"] - df["prev_price"])
    totals = df.groupby("ticker")["pnl"].sum()
    total_pnl = float(totals.sum()) if not totals.empty else 0.0
    spy_pnl = float(totals.get("SPY", 0.0))
    ief_pnl = float(totals.get("IEF", 0.0))
    return {
        "total_pnl": total_pnl,
        "spy_share": 0.0 if total_pnl == 0 else spy_pnl / total_pnl,
        "ief_share": 0.0 if total_pnl == 0 else ief_pnl / total_pnl,
    }


def _compute_signal_timing(prices: dict[str, pd.Series]) -> dict[str, float]:
    spy = prices["SPY"].copy()
    vix = prices["^VIX"].copy()
    spy.index = pd.to_datetime(spy.index, utc=True).tz_convert(None)
    vix.index = pd.to_datetime(vix.index, utc=True).tz_convert(None)
    common = spy.index.intersection(vix.index)
    spy = spy.reindex(common).dropna()
    vix = vix.reindex(common).dropna()
    returns = spy.pct_change().dropna()
    sma200 = spy.rolling(window=200, min_periods=200).mean()
    window = 60
    target_leverage = []
    dates = []
    for day in spy.index:
        if day not in returns.index:
            continue
        window_returns = returns.loc[:day].tail(window)
        if len(window_returns) < window or window_returns.std() == 0:
            leverage = 1.0
        else:
            zscore = (
                window_returns.iloc[-1] - window_returns.mean()
            ) / window_returns.std()
            leverage = 0.5 if zscore < -1.5 else 1.0
        if not pd.isna(sma200.loc[day]):
            in_trend = float(spy.loc[day]) > float(sma200.loc[day])
            if in_trend and leverage < 0.9:
                leverage = 0.9
            if not in_trend and float(vix.loc[day]) >= 36:
                leverage = min(leverage, 0.5)
        target_leverage.append(leverage)
        dates.append(day)
    if not dates:
        return {}
    series = pd.Series(target_leverage, index=pd.Index(dates))
    trigger = series < 1.0
    peak = spy.cummax()
    drawdown = spy / peak - 1.0
    dd_flag = drawdown <= -0.05
    lookahead = 20
    trigger_hits = 0
    for day in trigger[trigger].index:
        future = dd_flag.loc[day:].head(lookahead)
        if future.any():
            trigger_hits += 1
    dd_events = 0
    dd_hits = 0
    dd_start = dd_flag & (~dd_flag.shift(1, fill_value=False))
    for day in dd_start[dd_start].index:
        dd_events += 1
        past = trigger.loc[:day].tail(lookahead)
        if past.any():
            dd_hits += 1
    return {
        "trigger_days": float(trigger.sum()),
        "trigger_hit_rate": 0.0
        if trigger.sum() == 0
        else trigger_hits / float(trigger.sum()),
        "dd_events": float(dd_events),
        "dd_hit_rate": 0.0 if dd_events == 0 else dd_hits / float(dd_events),
    }


def _dynamic_ief_series(
    prices: dict[str, pd.Series],
    cut: float,
    min_leverage: float,
    vix_cap: float,
    ief_boost: float,
    vix_threshold: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    spy = prices["SPY"].copy()
    ief = prices["IEF"].copy()
    vix = prices["^VIX"].copy()
    spy.index = pd.to_datetime(spy.index, utc=True).tz_convert(None)
    ief.index = pd.to_datetime(ief.index, utc=True).tz_convert(None)
    vix.index = pd.to_datetime(vix.index, utc=True).tz_convert(None)
    common = spy.index.intersection(ief.index).intersection(vix.index)
    spy = spy.reindex(common).dropna()
    ief = ief.reindex(common).dropna()
    vix = vix.reindex(common).dropna()
    returns = spy.pct_change().dropna()
    sma200 = spy.rolling(window=200, min_periods=200).mean()
    weights_spy = []
    weights_ief = []
    portfolio_returns = []
    dates = []
    for day in spy.index:
        if day not in returns.index:
            continue
        window_returns = returns.loc[:day].tail(60)
        if len(window_returns) < 60 or window_returns.std() == 0:
            target_leverage = 1.0
        else:
            zscore = (
                window_returns.iloc[-1] - window_returns.mean()
            ) / window_returns.std()
            target_leverage = cut if zscore < -1.5 else 1.0
        if not pd.isna(sma200.loc[day]):
            in_trend = float(spy.loc[day]) > float(sma200.loc[day])
            if in_trend and target_leverage < min_leverage:
                target_leverage = min_leverage
            if not in_trend and float(vix.loc[day]) >= vix_cap:
                target_leverage = min(target_leverage, 0.5)

        base_ief = max(0.0, 1.0 - target_leverage)
        if not pd.isna(sma200.loc[day]) and float(spy.loc[day]) > float(
            sma200.loc[day]
        ):
            if float(vix.loc[day]) >= vix_threshold:
                base_ief = max(base_ief, ief_boost)

        weight_ief = min(1.0, base_ief)
        weight_spy = max(0.0, 1.0 - weight_ief)

        spy_ret = spy.pct_change().loc[day]
        ief_ret = ief.pct_change().loc[day]
        portfolio_returns.append(weight_spy * spy_ret + weight_ief * ief_ret)
        weights_spy.append(weight_spy)
        weights_ief.append(weight_ief)
        dates.append(day)

    if not dates:
        empty = pd.Series(dtype=float)
        return empty, empty, empty
    portfolio = pd.Series(portfolio_returns, index=pd.Index(dates)).dropna()
    normalized = (1.0 + portfolio).cumprod()
    spy_weights = pd.Series(weights_spy, index=pd.Index(dates)).reindex(
        normalized.index
    )
    ief_weights = pd.Series(weights_ief, index=pd.Index(dates)).reindex(
        normalized.index
    )
    return normalized, spy_weights, ief_weights


def _run_dynamic_ief_variant(
    prices: dict[str, pd.Series],
    config,
    cut: float,
    min_leverage: float,
    vix_cap: float,
    ief_boost: float,
    vix_threshold: float,
) -> tuple[dict[str, float], dict[str, float]]:
    normalized, spy_weights, ief_weights = _dynamic_ief_series(
        prices,
        cut,
        min_leverage,
        vix_cap,
        ief_boost,
        vix_threshold,
    )
    if normalized.empty:
        return {}, {}
    daily = [
        {
            "date": idx.strftime("%Y-%m-%d"),
            "equity": (value - 1.0) * config.paper_initial_capital,
        }
        for idx, value in normalized.items()
    ]
    metrics = _compute_performance_metrics(
        daily,
        config.paper_initial_capital,
        config.risk_free_rate,
        config.pnl_downside_min_days,
    )
    exposure = {
        "avg_spy_weight": float(spy_weights.mean()),
        "avg_ief_weight": float(ief_weights.mean()),
    }
    return metrics, exposure


def _run_variant(
    backfill: object,
    prices: dict[str, pd.Series],
    config,
    cut: float,
    min_leverage: float,
    vix_cap: float,
    max_leverage_off: float,
) -> tuple[dict[str, float], dict[str, float]]:
    backfill.ZSCORE_CUT = cut
    backfill.ZSCORE_TREND_ON_MIN_LEVERAGE = min_leverage
    backfill.ZSCORE_TREND_OFF_VIX = vix_cap
    backfill.ZSCORE_TREND_OFF_MAX_LEVERAGE = max_leverage_off
    entries = backfill._zscore_throttle_strategy(
        prices,
        config.paper_initial_capital,
        config.growth_ticker,
    )
    daily = _build_daily(entries)
    metrics = _compute_performance_metrics(
        daily,
        config.paper_initial_capital,
        config.risk_free_rate,
        config.pnl_downside_min_days,
    )
    exposure = _compute_exposure_stats(entries, config.paper_initial_capital)
    return metrics, exposure


def main() -> int:
    config = load_config()
    strategy_data = _prepare_strategy_data(config)
    common_dates = _common_dates(strategy_data)
    start = common_dates.min().strftime("%Y-%m-%d")
    end = common_dates.max().strftime("%Y-%m-%d")

    report_lines = ["# Strategy Next-Step Investigation", ""]
    report_lines.append(f"Generated: {date.today().strftime('%Y-%m-%d')}")
    report_lines.append(f"Common window: {start} -> {end}")
    report_lines.append("")
    checks_output_dir = Path("reports/selected_table_checks")
    slugify = lambda text: re.sub(r"[^a-z0-9_-]+", "_", text.lower()).strip("_")

    backfill = _load_backfill_module()
    candidate_prices = backfill._download_prices(
        ["^SPX", "SPY", "QQQ", "^VIX", "^VVIX", "^TNX", "CLR.SI", "IEF", "RSP"],
        start,
        (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    best_strategy_data: dict[str, StrategyData] = {}

    z_focus_rows = _sweep_zscore_grid(
        config,
        candidate_prices,
        cuts=[0.35, 0.4, 0.45],
        min_leverages=[0.9, 0.95],
        vix_caps=[34.0, 36.0],
        max_offs=[0.35, 0.4, 0.45],
    )
    z_focus_best = _top_n_rows(z_focus_rows, 1)
    if z_focus_best:
        row = z_focus_best[0]
        backfill.ZSCORE_CUT = float(row.get("cut", backfill.ZSCORE_CUT))
        backfill.ZSCORE_TREND_ON_MIN_LEVERAGE = float(
            row.get("min_leverage", backfill.ZSCORE_TREND_ON_MIN_LEVERAGE)
        )
        backfill.ZSCORE_TREND_OFF_VIX = float(
            row.get("vix_cap", backfill.ZSCORE_TREND_OFF_VIX)
        )
        backfill.ZSCORE_TREND_OFF_MAX_LEVERAGE = float(
            row.get("max_off", backfill.ZSCORE_TREND_OFF_MAX_LEVERAGE)
        )
        entries = backfill._zscore_throttle_strategy(
            candidate_prices,
            config.paper_initial_capital,
            config.growth_ticker,
        )
        path = Path(f"reports/pnl_backfill_zscore_focus_best_{start}_{end}.json")
        path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        best_strategy_data["zscore_focus_best"] = _strategy_from_entries(
            "zscore_focus_best", entries, config.paper_initial_capital
        )

    dyn_focus_rows = _sweep_dynamic_ief_positions_grid(
        config,
        candidate_prices,
        cuts=[0.4, 0.45],
        min_leverages=[0.85, 0.9],
        vix_caps=[34.0, 36.0],
        boosts=[0.25, 0.3],
        thresholds=[26.0, 28.0, 30.0],
        lags=[0, 1],
        tag_prefix="focus_best",
    )
    dyn_focus_best = _top_n_rows(dyn_focus_rows, 1)
    if dyn_focus_best:
        row = dyn_focus_best[0]
        backfill.DYNAMIC_IEF_CUT = float(row.get("cut", backfill.DYNAMIC_IEF_CUT))
        backfill.DYNAMIC_IEF_MIN_LEVERAGE = float(
            row.get("min_leverage", backfill.DYNAMIC_IEF_MIN_LEVERAGE)
        )
        backfill.DYNAMIC_IEF_VIX_CAP = float(
            row.get("vix_cap", backfill.DYNAMIC_IEF_VIX_CAP)
        )
        backfill.DYNAMIC_IEF_BOOST = float(
            row.get("ief_boost", backfill.DYNAMIC_IEF_BOOST)
        )
        backfill.DYNAMIC_IEF_VIX_THRESHOLD = float(
            row.get("vix_threshold", backfill.DYNAMIC_IEF_VIX_THRESHOLD)
        )
        entries = backfill._dynamic_ief_strategy(
            candidate_prices,
            config.paper_initial_capital,
            config.growth_ticker,
            lag_days=int(row.get("lag", 0)),
            tag_suffix="focus_best",
        )
        path = Path(f"reports/pnl_backfill_dynamic_ief_focus_best_{start}_{end}.json")
        path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        best_strategy_data["dynamic_ief_focus_best"] = _strategy_from_entries(
            "dynamic_ief_focus_best", entries, config.paper_initial_capital
        )

    # Step 1: Regime breakdown
    report_lines.append("## Step 1: Regime Breakdown")
    report_lines.append("")
    _, breakdown = _regime_breakdown(strategy_data, config)
    for name, regimes in breakdown.items():
        report_lines.append(f"### {name}")
        report_lines.append(
            "| Regime | Sharpe | Final Equity | CAGR | Vol | Max DD | Calmar | Points |"
        )
        report_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for regime, metrics in regimes.items():
            report_lines.append(
                "| {regime} | {sharpe} | {end} | {cagr} | {vol} | {dd} | {calmar} | {points} |".format(
                    regime=regime,
                    sharpe=_fmt_num(metrics.get("sharpe_ratio")),
                    end=_fmt_num(metrics.get("normalized_end"), 4),
                    cagr=_fmt_pct(metrics.get("cagr")),
                    vol=_fmt_pct(metrics.get("annualized_volatility")),
                    dd=_fmt_pct(metrics.get("max_drawdown_pct")),
                    calmar=_fmt_num(metrics.get("calmar_ratio")),
                    points=metrics.get("data_points", 0),
                )
            )
        report_lines.append("")

    if best_strategy_data:
        report_lines.append("### Best Grid Candidates")
        for name, data in best_strategy_data.items():
            report_lines.append(f"#### {name}")
            report_lines.append(
                "| Regime | Sharpe | Final Equity | CAGR | Vol | Max DD | Calmar | Points |"
            )
            report_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
            regimes = _regime_breakdown_single(data.normalized, data.daily, config)
            for regime, metrics in regimes.items():
                report_lines.append(
                    "| {regime} | {sharpe} | {end} | {cagr} | {vol} | {dd} | {calmar} | {points} |".format(
                        regime=regime,
                        sharpe=_fmt_num(metrics.get("sharpe_ratio")),
                        end=_fmt_num(metrics.get("normalized_end"), 4),
                        cagr=_fmt_pct(metrics.get("cagr")),
                        vol=_fmt_pct(metrics.get("annualized_volatility")),
                        dd=_fmt_pct(metrics.get("max_drawdown_pct")),
                        calmar=_fmt_num(metrics.get("calmar_ratio")),
                        points=metrics.get("data_points", 0),
                    )
                )
            report_lines.append("")

    # Step 2: Rolling-window stability
    report_lines.append("## Step 2: Rolling-Window Stability")
    report_lines.append("")
    for window_years in (3, 5):
        summary = _rolling_summary(strategy_data, config, window_years)
        report_lines.append(f"### {window_years}Y Rolling")
        report_lines.append(
            "| Strategy | Sharpe Mean | Sharpe Min | CAGR Mean | CAGR Min | Windows |"
        )
        report_lines.append("| --- | --- | --- | --- | --- | --- |")
        for name, stats in summary.items():
            report_lines.append(
                "| {name} | {sharpe_mean} | {sharpe_min} | {cagr_mean} | {cagr_min} | {windows} |".format(
                    name=name,
                    sharpe_mean=_fmt_num(stats.get("sharpe_mean")),
                    sharpe_min=_fmt_num(stats.get("sharpe_min")),
                    cagr_mean=_fmt_pct(stats.get("cagr_mean")),
                    cagr_min=_fmt_pct(stats.get("cagr_min")),
                    windows=stats.get("windows", 0),
                )
            )
        report_lines.append("")

    if best_strategy_data:
        for window_years in (3, 5):
            summary = _rolling_summary(best_strategy_data, config, window_years)
            report_lines.append(f"### Best Grid Candidates {window_years}Y Rolling")
            report_lines.append(
                "| Strategy | Sharpe Mean | Sharpe Min | CAGR Mean | CAGR Min | Windows |"
            )
            report_lines.append("| --- | --- | --- | --- | --- | --- |")
            for name, stats in summary.items():
                report_lines.append(
                    "| {name} | {sharpe_mean} | {sharpe_min} | {cagr_mean} | {cagr_min} | {windows} |".format(
                        name=name,
                        sharpe_mean=_fmt_num(stats.get("sharpe_mean")),
                        sharpe_min=_fmt_num(stats.get("sharpe_min")),
                        cagr_mean=_fmt_pct(stats.get("cagr_mean")),
                        cagr_min=_fmt_pct(stats.get("cagr_min")),
                        windows=stats.get("windows", 0),
                    )
                )
            report_lines.append("")

    # Step 3: Blend comparison
    report_lines.append("## Step 3: Blend Comparison")
    report_lines.append("")
    common_metrics = {}
    for name, data in strategy_data.items():
        metrics = _metrics_for_dates(data.daily, common_dates, config)
        common_metrics[name] = metrics

    top = sorted(
        common_metrics.items(),
        key=lambda item: item[1].get("sharpe_ratio") or -999,
        reverse=True,
    )
    if len(top) >= 2:
        base_a = strategy_data[top[0][0]]
        base_b = strategy_data[top[1][0]]
        report_lines.append(f"Base strategies: {base_a.name} + {base_b.name}")
        report_lines.append(
            "| Blend | Sharpe | Final Equity | CAGR | Vol | Max DD | Calmar | Points |"
        )
        report_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for weight in (0.8, 0.7, 0.6, 0.5):
            blended_returns = _blend_returns(base_a.returns, base_b.returns, weight)
            metrics = _blend_metrics(blended_returns, config)
            report_lines.append(
                "| {label} | {sharpe} | {end} | {cagr} | {vol} | {dd} | {calmar} | {points} |".format(
                    label=_blend_label(weight),
                    sharpe=_fmt_num(metrics.get("sharpe_ratio")),
                    end=_fmt_num(metrics.get("normalized_end"), 4),
                    cagr=_fmt_pct(metrics.get("cagr")),
                    vol=_fmt_pct(metrics.get("annualized_volatility")),
                    dd=_fmt_pct(metrics.get("max_drawdown_pct")),
                    calmar=_fmt_num(metrics.get("calmar_ratio")),
                    points=metrics.get("data_points", 0),
                )
            )

        if "regime_blend_ief20" in strategy_data:
            base_c = strategy_data["regime_blend_ief20"]
            report_lines.append("")
            report_lines.append(
                "Three-way blends: zscore_throttle_tuned / zscore_throttle_cut50 / regime_blend_ief20"
            )
            report_lines.append(
                "| Blend | Sharpe | Final Equity | CAGR | Vol | Max DD | Calmar | Points |"
            )
            report_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
            blends = [
                (0.7, 0.2, 0.1),
                (0.6, 0.3, 0.1),
                (0.6, 0.2, 0.2),
                (0.5, 0.3, 0.2),
            ]
            for w_a, w_b, w_c in blends:
                common = base_a.returns.index.intersection(base_b.returns.index)
                common = common.intersection(base_c.returns.index)
                ra = base_a.returns.reindex(common).fillna(0.0)
                rb = base_b.returns.reindex(common).fillna(0.0)
                rc = base_c.returns.reindex(common).fillna(0.0)
                blended_returns = ra * w_a + rb * w_b + rc * w_c
                metrics = _blend_metrics(blended_returns, config)
                label = f"{int(round(w_a * 100))}/{int(round(w_b * 100))}/{int(round(w_c * 100))}"
                report_lines.append(
                    "| {label} | {sharpe} | {end} | {cagr} | {vol} | {dd} | {calmar} | {points} |".format(
                        label=label,
                        sharpe=_fmt_num(metrics.get("sharpe_ratio")),
                        end=_fmt_num(metrics.get("normalized_end"), 4),
                        cagr=_fmt_pct(metrics.get("cagr")),
                        vol=_fmt_pct(metrics.get("annualized_volatility")),
                        dd=_fmt_pct(metrics.get("max_drawdown_pct")),
                        calmar=_fmt_num(metrics.get("calmar_ratio")),
                        points=metrics.get("data_points", 0),
                    )
                )
    else:
        report_lines.append("Not enough strategies for blend analysis.")
    report_lines.append("")

    if len(best_strategy_data) >= 2:
        report_lines.append("### Best Grid Candidates (Blend Comparison)")
        best_common = _common_dates(best_strategy_data)
        best_metrics = {
            name: _metrics_for_dates(data.daily, best_common, config)
            for name, data in best_strategy_data.items()
        }
        best_top = sorted(
            best_metrics.items(),
            key=lambda item: item[1].get("sharpe_ratio") or -999,
            reverse=True,
        )
        base_a = best_strategy_data[best_top[0][0]]
        base_b = best_strategy_data[best_top[1][0]]
        report_lines.append(f"Base strategies: {base_a.name} + {base_b.name}")
        report_lines.append(
            "| Blend | Sharpe | Final Equity | CAGR | Vol | Max DD | Calmar | Points |"
        )
        report_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for weight in (0.8, 0.7, 0.6, 0.5):
            blended_returns = _blend_returns(base_a.returns, base_b.returns, weight)
            metrics = _blend_metrics(blended_returns, config)
            report_lines.append(
                "| {label} | {sharpe} | {end} | {cagr} | {vol} | {dd} | {calmar} | {points} |".format(
                    label=_blend_label(weight),
                    sharpe=_fmt_num(metrics.get("sharpe_ratio")),
                    end=_fmt_num(metrics.get("normalized_end"), 4),
                    cagr=_fmt_pct(metrics.get("cagr")),
                    vol=_fmt_pct(metrics.get("annualized_volatility")),
                    dd=_fmt_pct(metrics.get("max_drawdown_pct")),
                    calmar=_fmt_num(metrics.get("calmar_ratio")),
                    points=metrics.get("data_points", 0),
                )
            )
        report_lines.append("")

    # Step 4: Zscore sensitivity sweep
    report_lines.append("## Step 4: Zscore Sensitivity Sweep")
    report_lines.append("")
    sweep = _sweep_zscore(config, start, end)
    sweep_sorted = sorted(
        sweep,
        key=lambda row: (
            row.get("sharpe") if row.get("sharpe") is not None else -999,
            row.get("final_equity") if row.get("final_equity") is not None else -999,
        ),
        reverse=True,
    )
    report_lines.append(
        "| Cut | Min Lev | VIX Cap | Sharpe | Final Equity | CAGR | Max DD |"
    )
    report_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in sweep_sorted[:20]:
        report_lines.append(
            "| {cut} | {min_lev} | {vix_cap} | {sharpe} | {end} | {cagr} | {dd} |".format(
                cut=_fmt_num(row.get("cut"), 2),
                min_lev=_fmt_num(row.get("min_leverage"), 2),
                vix_cap=_fmt_num(row.get("vix_cap"), 2),
                sharpe=_fmt_num(row.get("sharpe")),
                end=_fmt_num(row.get("final_equity"), 4),
                cagr=_fmt_pct(row.get("cagr")),
                dd=_fmt_pct(row.get("max_dd")),
            )
        )

    report_lines.append("")
    report_lines.append("### Fine Sweep (focus grid)")
    report_lines.append(
        "| Cut | Min Lev | VIX Cap | Sharpe | Final Equity | CAGR | Max DD |"
    )
    report_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    fine_prices = _load_backfill_module()._download_prices(
        ["^SPX", "SPY", "QQQ", "^VIX", "^VVIX", "^TNX", "CLR.SI", "IEF", "RSP"],
        start,
        (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    fine_results = _sweep_zscore_fine(config, fine_prices)
    fine_sorted = sorted(
        fine_results,
        key=lambda row: (
            row.get("sharpe") if row.get("sharpe") is not None else -999,
            row.get("final_equity") if row.get("final_equity") is not None else -999,
        ),
        reverse=True,
    )
    for row in fine_sorted[:20]:
        report_lines.append(
            "| {cut} | {min_lev} | {vix_cap} | {sharpe} | {end} | {cagr} | {dd} |".format(
                cut=_fmt_num(row.get("cut"), 2),
                min_lev=_fmt_num(row.get("min_leverage"), 2),
                vix_cap=_fmt_num(row.get("vix_cap"), 2),
                sharpe=_fmt_num(row.get("sharpe")),
                end=_fmt_num(row.get("final_equity"), 4),
                cagr=_fmt_pct(row.get("cagr")),
                dd=_fmt_pct(row.get("max_dd")),
            )
        )

    report_lines.append("")
    report_lines.append("### Expanded Sweep (higher VIX caps)")
    report_lines.append(
        "| Cut | Min Lev | VIX Cap | Sharpe | Final Equity | CAGR | Max DD |"
    )
    report_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    expanded_results = _sweep_zscore_expanded(config, fine_prices)
    expanded_sorted = sorted(
        expanded_results,
        key=lambda row: (
            row.get("sharpe") if row.get("sharpe") is not None else -999,
            row.get("final_equity") if row.get("final_equity") is not None else -999,
        ),
        reverse=True,
    )
    for row in expanded_sorted[:20]:
        report_lines.append(
            "| {cut} | {min_lev} | {vix_cap} | {sharpe} | {end} | {cagr} | {dd} |".format(
                cut=_fmt_num(row.get("cut"), 2),
                min_lev=_fmt_num(row.get("min_leverage"), 2),
                vix_cap=_fmt_num(row.get("vix_cap"), 2),
                sharpe=_fmt_num(row.get("sharpe")),
                end=_fmt_num(row.get("final_equity"), 4),
                cagr=_fmt_pct(row.get("cagr")),
                dd=_fmt_pct(row.get("max_dd")),
            )
        )

    report_lines.append("")
    report_lines.append("### Cached Sweep (long-history cache)")
    cached_results, cached_note = _sweep_zscore_cached_long(config)
    if cached_note:
        report_lines.append(f"- {cached_note}")
    else:
        report_lines.append(
            "| Cut | Min Lev | VIX Cap | Sharpe | Final Equity | CAGR | Max DD |"
        )
        report_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        cached_sorted = sorted(
            cached_results,
            key=lambda row: (
                row.get("sharpe") if row.get("sharpe") is not None else -999,
                row.get("final_equity")
                if row.get("final_equity") is not None
                else -999,
            ),
            reverse=True,
        )
        for row in cached_sorted[:20]:
            report_lines.append(
                "| {cut} | {min_lev} | {vix_cap} | {sharpe} | {end} | {cagr} | {dd} |".format(
                    cut=_fmt_num(row.get("cut"), 2),
                    min_lev=_fmt_num(row.get("min_leverage"), 2),
                    vix_cap=_fmt_num(row.get("vix_cap"), 2),
                    sharpe=_fmt_num(row.get("sharpe")),
                    end=_fmt_num(row.get("final_equity"), 4),
                    cagr=_fmt_pct(row.get("cagr")),
                    dd=_fmt_pct(row.get("max_dd")),
                )
            )

        report_lines.append("")
        report_lines.append("### Sweep Comparison (top rows)")
        top_fresh = _top_sweep_row(sweep)
        top_cached = _top_sweep_row(cached_results)
        report_lines.append(
            "| Source | Cut | Min Lev | VIX Cap | Sharpe | Final Equity | CAGR | Max DD |"
        )
        report_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        if top_fresh:
            report_lines.append(
                "| fresh | {cut} | {min_lev} | {vix_cap} | {sharpe} | {end} | {cagr} | {dd} |".format(
                    cut=_fmt_num(top_fresh.get("cut"), 2),
                    min_lev=_fmt_num(top_fresh.get("min_leverage"), 2),
                    vix_cap=_fmt_num(top_fresh.get("vix_cap"), 2),
                    sharpe=_fmt_num(top_fresh.get("sharpe")),
                    end=_fmt_num(top_fresh.get("final_equity"), 4),
                    cagr=_fmt_pct(top_fresh.get("cagr")),
                    dd=_fmt_pct(top_fresh.get("max_dd")),
                )
            )
        if top_cached:
            report_lines.append(
                "| cached | {cut} | {min_lev} | {vix_cap} | {sharpe} | {end} | {cagr} | {dd} |".format(
                    cut=_fmt_num(top_cached.get("cut"), 2),
                    min_lev=_fmt_num(top_cached.get("min_leverage"), 2),
                    vix_cap=_fmt_num(top_cached.get("vix_cap"), 2),
                    sharpe=_fmt_num(top_cached.get("sharpe")),
                    end=_fmt_num(top_cached.get("final_equity"), 4),
                    cagr=_fmt_pct(top_cached.get("cagr")),
                    dd=_fmt_pct(top_cached.get("max_dd")),
                )
            )

        if top_cached:
            report_lines.append("")
            report_lines.append("### Candidate Backfill (from top cached params)")
            backfill = _load_backfill_module()
            backfill.ZSCORE_CUT = float(top_cached.get("cut", backfill.ZSCORE_CUT))
            backfill.ZSCORE_TREND_ON_MIN_LEVERAGE = float(
                top_cached.get("min_leverage", backfill.ZSCORE_TREND_ON_MIN_LEVERAGE)
            )
            backfill.ZSCORE_TREND_OFF_VIX = float(
                top_cached.get("vix_cap", backfill.ZSCORE_TREND_OFF_VIX)
            )
            candidate_prices = backfill._download_prices(
                ["^SPX", "SPY", "QQQ", "^VIX", "^VVIX", "^TNX", "CLR.SI", "IEF", "RSP"],
                start,
                (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            )
            candidate_entries = backfill._zscore_throttle_strategy(
                candidate_prices,
                config.paper_initial_capital,
                config.growth_ticker,
            )
            candidate_path = Path(
                f"reports/pnl_backfill_zscore_throttle_candidate_{start}_{end}.json"
            )
            candidate_path.write_text(
                json.dumps(candidate_entries, indent=2),
                encoding="utf-8",
            )
            candidate_daily = _build_daily(candidate_entries)
            candidate_metrics = _compute_performance_metrics(
                candidate_daily,
                config.paper_initial_capital,
                config.risk_free_rate,
                config.pnl_downside_min_days,
            )
            report_lines.append(f"Backfill report: {candidate_path}")
            report_lines.append(
                "Params: cut={cut}, min_leverage={min_lev}, vix_cap={vix_cap}".format(
                    cut=_fmt_num(top_cached.get("cut"), 2),
                    min_lev=_fmt_num(top_cached.get("min_leverage"), 2),
                    vix_cap=_fmt_num(top_cached.get("vix_cap"), 2),
                )
            )
            report_lines.append(
                "| Sharpe | Final Equity | CAGR | Vol | Max DD | Calmar | Points |"
            )
            report_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            report_lines.append(
                "| {sharpe} | {end} | {cagr} | {vol} | {dd} | {calmar} | {points} |".format(
                    sharpe=_fmt_num(candidate_metrics.get("sharpe_ratio")),
                    end=_fmt_num(candidate_metrics.get("normalized_end"), 4),
                    cagr=_fmt_pct(candidate_metrics.get("cagr")),
                    vol=_fmt_pct(candidate_metrics.get("annualized_volatility")),
                    dd=_fmt_pct(candidate_metrics.get("max_drawdown_pct")),
                    calmar=_fmt_num(candidate_metrics.get("calmar_ratio")),
                    points=candidate_metrics.get("data_points", 0),
                )
            )

            report_lines.append("")
            report_lines.append("## Diagnostics (Candidate)")
            exposure = _compute_exposure_stats(
                candidate_entries, config.paper_initial_capital
            )
            contrib = _compute_contribution_stats(candidate_entries)
            report_lines.append("### Attribution & Exposure")
            report_lines.append("| Metric | Value |")
            report_lines.append("| --- | --- |")
            report_lines.append(
                f"| Avg leverage | {_fmt_num(exposure.get('avg_leverage'))} |"
            )
            report_lines.append(
                f"| Median leverage | {_fmt_num(exposure.get('median_leverage'))} |"
            )
            report_lines.append(
                f"| % days leverage < 1 | {_fmt_pct(exposure.get('pct_leverage_lt_1'))} |"
            )
            report_lines.append(
                f"| Avg SPY weight | {_fmt_pct(exposure.get('avg_spy_weight'))} |"
            )
            report_lines.append(
                f"| Avg IEF weight | {_fmt_pct(exposure.get('avg_ief_weight'))} |"
            )
            report_lines.append(
                f"| SPY pnl share | {_fmt_pct(contrib.get('spy_share'))} |"
            )
            report_lines.append(
                f"| IEF pnl share | {_fmt_pct(contrib.get('ief_share'))} |"
            )

            report_lines.append("")
            report_lines.append("### Leverage Distribution")
            buckets = _compute_leverage_buckets(
                candidate_entries, config.paper_initial_capital
            )
            report_lines.append("| Bucket | Share |")
            report_lines.append("| --- | --- |")
            report_lines.append(f"| <0.5 | {_fmt_pct(buckets.get('lt_0_5'))} |")
            report_lines.append(f"| 0.5-0.8 | {_fmt_pct(buckets.get('0_5_0_8'))} |")
            report_lines.append(f"| 0.8-1.0 | {_fmt_pct(buckets.get('0_8_1_0'))} |")
            report_lines.append(f"| 1.0-1.2 | {_fmt_pct(buckets.get('1_0_1_2'))} |")
            report_lines.append(f"| >=1.2 | {_fmt_pct(buckets.get('ge_1_2'))} |")

            report_lines.append("")
            report_lines.append("### Leverage Bucket Performance")
            leverage = _compute_leverage_series(
                candidate_entries, config.paper_initial_capital
            )
            report_lines.append(
                "| Bucket | Sharpe | Final Equity | CAGR | Max DD | Points |"
            )
            report_lines.append("| --- | --- | --- | --- | --- | --- |")
            bucket_defs = {
                "<0.5": leverage < 0.5,
                "0.5-0.8": (leverage >= 0.5) & (leverage < 0.8),
                "0.8-1.0": (leverage >= 0.8) & (leverage < 1.0),
                "1.0-1.2": (leverage >= 1.0) & (leverage < 1.2),
                ">=1.2": leverage >= 1.2,
            }
            for label, mask in bucket_defs.items():
                dates = leverage.index[mask.fillna(False)]
                metrics = _metrics_for_dates(candidate_daily, dates, config)
                report_lines.append(
                    "| {label} | {sharpe} | {end} | {cagr} | {dd} | {points} |".format(
                        label=label,
                        sharpe=_fmt_num(metrics.get("sharpe_ratio")),
                        end=_fmt_num(metrics.get("normalized_end"), 4),
                        cagr=_fmt_pct(metrics.get("cagr")),
                        dd=_fmt_pct(metrics.get("max_drawdown_pct")),
                        points=metrics.get("data_points", 0),
                    )
                )

            report_lines.append("")
            report_lines.append("### Signal Timing (SPY drawdown >= 5%)")
            timing = _compute_signal_timing(candidate_prices)
            report_lines.append("| Metric | Value |")
            report_lines.append("| --- | --- |")
            report_lines.append(
                f"| Trigger days | {_fmt_num(timing.get('trigger_days'))} |"
            )
            report_lines.append(
                f"| Trigger hit rate (20d) | {_fmt_pct(timing.get('trigger_hit_rate'))} |"
            )
            report_lines.append(
                f"| Drawdown events | {_fmt_num(timing.get('dd_events'))} |"
            )
            report_lines.append(
                f"| Drawdown hit rate (20d lookback) | {_fmt_pct(timing.get('dd_hit_rate'))} |"
            )

            report_lines.append("")
            report_lines.append("### Aggressive Cut Variant")
            aggressive_metrics, aggressive_exposure = _run_variant(
                backfill,
                candidate_prices,
                config,
                cut=0.3,
                min_leverage=0.8,
                vix_cap=30.0,
                max_leverage_off=0.3,
            )
            report_lines.append(
                "Params: cut=0.30, min_leverage=0.80, vix_cap=30, max_off=0.30"
            )
            report_lines.append(
                "| Sharpe | Final Equity | CAGR | Vol | Max DD | Calmar | Points |"
            )
            report_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            report_lines.append(
                "| {sharpe} | {end} | {cagr} | {vol} | {dd} | {calmar} | {points} |".format(
                    sharpe=_fmt_num(aggressive_metrics.get("sharpe_ratio")),
                    end=_fmt_num(aggressive_metrics.get("normalized_end"), 4),
                    cagr=_fmt_pct(aggressive_metrics.get("cagr")),
                    vol=_fmt_pct(aggressive_metrics.get("annualized_volatility")),
                    dd=_fmt_pct(aggressive_metrics.get("max_drawdown_pct")),
                    calmar=_fmt_num(aggressive_metrics.get("calmar_ratio")),
                    points=aggressive_metrics.get("data_points", 0),
                )
            )
            report_lines.append("| Metric | Value |")
            report_lines.append("| --- | --- |")
            report_lines.append(
                f"| Avg leverage | {_fmt_num(aggressive_exposure.get('avg_leverage'))} |"
            )
            report_lines.append(
                f"| % days leverage < 1 | {_fmt_pct(aggressive_exposure.get('pct_leverage_lt_1'))} |"
            )

            report_lines.append("")
            report_lines.append("### Dynamic IEF Allocation Variant")
            dyn_metrics, dyn_exposure = _run_dynamic_ief_variant(
                candidate_prices,
                config,
                cut=0.5,
                min_leverage=0.9,
                vix_cap=32.0,
                ief_boost=0.2,
                vix_threshold=28.0,
            )
            report_lines.append(
                "Params: cut=0.50, min_leverage=0.90, vix_cap=32, ief_boost=0.20, vix_threshold=28"
            )
            report_lines.append(
                "| Sharpe | Final Equity | CAGR | Vol | Max DD | Calmar | Points |"
            )
            report_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            report_lines.append(
                "| {sharpe} | {end} | {cagr} | {vol} | {dd} | {calmar} | {points} |".format(
                    sharpe=_fmt_num(dyn_metrics.get("sharpe_ratio")),
                    end=_fmt_num(dyn_metrics.get("normalized_end"), 4),
                    cagr=_fmt_pct(dyn_metrics.get("cagr")),
                    vol=_fmt_pct(dyn_metrics.get("annualized_volatility")),
                    dd=_fmt_pct(dyn_metrics.get("max_drawdown_pct")),
                    calmar=_fmt_num(dyn_metrics.get("calmar_ratio")),
                    points=dyn_metrics.get("data_points", 0),
                )
            )
            report_lines.append("| Metric | Value |")
            report_lines.append("| --- | --- |")
            report_lines.append(
                f"| Avg SPY weight | {_fmt_pct(dyn_exposure.get('avg_spy_weight'))} |"
            )
            report_lines.append(
                f"| Avg IEF weight | {_fmt_pct(dyn_exposure.get('avg_ief_weight'))} |"
            )

            report_lines.append("")
            report_lines.append("### Dynamic IEF Synthetic vs Positions (same params)")
            report_lines.append(
                "Params: cut=0.50, min_leverage=0.90, vix_cap=32, ief_boost=0.20, vix_threshold=28"
            )
            dyn_normalized, dyn_spy_weights, dyn_ief_weights = _dynamic_ief_series(
                candidate_prices,
                cut=0.5,
                min_leverage=0.9,
                vix_cap=32.0,
                ief_boost=0.2,
                vix_threshold=28.0,
            )
            dyn_synth_metrics: dict[str, float] = {}
            if not dyn_normalized.empty:
                dyn_synth_daily = [
                    {
                        "date": idx.strftime("%Y-%m-%d"),
                        "equity": (value - 1.0) * config.paper_initial_capital,
                    }
                    for idx, value in dyn_normalized.items()
                ]
                dyn_synth_metrics = _compute_performance_metrics(
                    dyn_synth_daily,
                    config.paper_initial_capital,
                    config.risk_free_rate,
                    config.pnl_downside_min_days,
                )

            backfill.DYNAMIC_IEF_CUT = 0.5
            backfill.DYNAMIC_IEF_MIN_LEVERAGE = 0.9
            backfill.DYNAMIC_IEF_VIX_CAP = 32.0
            backfill.DYNAMIC_IEF_BOOST = 0.2
            backfill.DYNAMIC_IEF_VIX_THRESHOLD = 28.0
            dyn_pos_entries = backfill._dynamic_ief_strategy(
                candidate_prices,
                config.paper_initial_capital,
                config.growth_ticker,
                lag_days=0,
                tag_suffix="synth_vs_pos",
            )
            dyn_pos_daily = _build_daily(dyn_pos_entries)
            dyn_pos_metrics = _compute_performance_metrics(
                dyn_pos_daily,
                config.paper_initial_capital,
                config.risk_free_rate,
                config.pnl_downside_min_days,
            )
            report_lines.append(
                "| Variant | Sharpe | Final Equity | CAGR | Vol | Max DD | Calmar |"
            )
            report_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            report_lines.append(
                "| synthetic | {sharpe} | {end} | {cagr} | {vol} | {dd} | {calmar} |".format(
                    sharpe=_fmt_num(dyn_synth_metrics.get("sharpe_ratio")),
                    end=_fmt_num(dyn_synth_metrics.get("normalized_end"), 4),
                    cagr=_fmt_pct(dyn_synth_metrics.get("cagr")),
                    vol=_fmt_pct(dyn_synth_metrics.get("annualized_volatility")),
                    dd=_fmt_pct(dyn_synth_metrics.get("max_drawdown_pct")),
                    calmar=_fmt_num(dyn_synth_metrics.get("calmar_ratio")),
                )
            )
            report_lines.append(
                "| positions | {sharpe} | {end} | {cagr} | {vol} | {dd} | {calmar} |".format(
                    sharpe=_fmt_num(dyn_pos_metrics.get("sharpe_ratio")),
                    end=_fmt_num(dyn_pos_metrics.get("normalized_end"), 4),
                    cagr=_fmt_pct(dyn_pos_metrics.get("cagr")),
                    vol=_fmt_pct(dyn_pos_metrics.get("annualized_volatility")),
                    dd=_fmt_pct(dyn_pos_metrics.get("max_drawdown_pct")),
                    calmar=_fmt_num(dyn_pos_metrics.get("calmar_ratio")),
                )
            )

            report_lines.append("")
            report_lines.append("### Dynamic IEF Variant Grid")
            report_lines.append(
                "| IEF Boost | VIX Threshold | Sharpe | Final Equity | CAGR | Max DD |"
            )
            report_lines.append("| --- | --- | --- | --- | --- | --- |")
            for boost in (0.1, 0.2, 0.3):
                for threshold in (24.0, 28.0, 32.0):
                    grid_metrics, _ = _run_dynamic_ief_variant(
                        candidate_prices,
                        config,
                        cut=0.5,
                        min_leverage=0.9,
                        vix_cap=32.0,
                        ief_boost=boost,
                        vix_threshold=threshold,
                    )
                    report_lines.append(
                        "| {boost} | {threshold} | {sharpe} | {end} | {cagr} | {dd} |".format(
                            boost=_fmt_num(boost, 2),
                            threshold=_fmt_num(threshold, 0),
                            sharpe=_fmt_num(grid_metrics.get("sharpe_ratio")),
                            end=_fmt_num(grid_metrics.get("normalized_end"), 4),
                            cagr=_fmt_pct(grid_metrics.get("cagr")),
                            dd=_fmt_pct(grid_metrics.get("max_drawdown_pct")),
                        )
                    )

            report_lines.append("")
            report_lines.append("### Dynamic IEF Backfill (synthetic)")
            dyn_normalized, dyn_spy_weights, dyn_ief_weights = _dynamic_ief_series(
                candidate_prices,
                cut=0.5,
                min_leverage=0.9,
                vix_cap=32.0,
                ief_boost=0.2,
                vix_threshold=28.0,
            )
            dyn_entries: list[dict[str, object]] = []
            for idx, value in dyn_normalized.items():
                equity = (value - 1.0) * config.paper_initial_capital
                dyn_entries.append(
                    {
                        "timestamp": f"{idx.strftime('%Y-%m-%d')} 16:00:00",
                        "category": "Execution",
                        "ticker": "DYN_IEF",
                        "action": "PAPER_PNL",
                        "rationale": "Dynamic IEF allocation synthetic PnL.",
                        "tags": ["paper", "backfill", "strategy", "dynamic_ief"],
                        "details": {
                            "quantity": 0.0,
                            "avg_price": 0.0,
                            "mark_price": 0.0,
                            "unrealized_pnl": float(equity),
                            "module": "alpha",
                        },
                        "entry_type": "execution",
                        "schema_version": 2,
                    }
                )
            dyn_path = Path(
                f"reports/pnl_backfill_dynamic_ief_synthetic_{start}_{end}.json"
            )
            dyn_path.write_text(json.dumps(dyn_entries, indent=2), encoding="utf-8")
            report_lines.append(f"Backfill report: {dyn_path}")

            positions_path = Path(
                f"reports/pnl_backfill_dynamic_ief_{start}_{end}.json"
            )
            positions_entries = _load_entries(positions_path)
            positions_daily = _build_daily(positions_entries)
            positions_normalized = _normalized_series(
                positions_daily,
                config.paper_initial_capital,
            )

            report_lines.append("")
            report_lines.append("### Dynamic IEF Backfill (positions-based)")
            report_lines.append(f"Backfill report: {positions_path}")
            if positions_daily:
                positions_metrics = _compute_performance_metrics(
                    positions_daily,
                    config.paper_initial_capital,
                    config.risk_free_rate,
                    config.pnl_downside_min_days,
                )
                pos_start = positions_normalized.index.min()
                pos_end = positions_normalized.index.max()
                report_lines.append(
                    "| Sharpe | Final Equity | CAGR | Vol | Max DD | Calmar | Start | End |"
                )
                report_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
                report_lines.append(
                    "| {sharpe} | {end} | {cagr} | {vol} | {dd} | {calmar} | {start} | {finish} |".format(
                        sharpe=_fmt_num(positions_metrics.get("sharpe_ratio")),
                        end=_fmt_num(positions_metrics.get("normalized_end"), 4),
                        cagr=_fmt_pct(positions_metrics.get("cagr")),
                        vol=_fmt_pct(positions_metrics.get("annualized_volatility")),
                        dd=_fmt_pct(positions_metrics.get("max_drawdown_pct")),
                        calmar=_fmt_num(positions_metrics.get("calmar_ratio")),
                        start=pos_start.strftime("%Y-%m-%d")
                        if pos_start is not None
                        else "N/A",
                        finish=pos_end.strftime("%Y-%m-%d")
                        if pos_end is not None
                        else "N/A",
                    )
                )
            else:
                report_lines.append(
                    "- Positions-based backfill missing; metrics skipped."
                )

            report_lines.append("")
            report_lines.append("### Dynamic IEF Backfill (positions-based, lagged 1d)")
            lag1_entries = backfill._dynamic_ief_strategy(
                candidate_prices,
                config.paper_initial_capital,
                config.growth_ticker,
                lag_days=1,
                tag_suffix="lag_1d",
            )
            lag1_path = Path(
                f"reports/pnl_backfill_dynamic_ief_lag1_{start}_{end}.json"
            )
            lag1_path.write_text(json.dumps(lag1_entries, indent=2), encoding="utf-8")
            lag1_daily = _build_daily(lag1_entries)
            lag1_metrics = _compute_performance_metrics(
                lag1_daily,
                config.paper_initial_capital,
                config.risk_free_rate,
                config.pnl_downside_min_days,
            )
            lag1_normalized = _normalized_series(
                lag1_daily, config.paper_initial_capital
            )
            lag1_start = lag1_normalized.index.min()
            lag1_end = lag1_normalized.index.max()
            report_lines.append(f"Backfill report: {lag1_path}")
            report_lines.append(
                "| Sharpe | Final Equity | CAGR | Vol | Max DD | Calmar | Start | End |"
            )
            report_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
            report_lines.append(
                "| {sharpe} | {end} | {cagr} | {vol} | {dd} | {calmar} | {start} | {finish} |".format(
                    sharpe=_fmt_num(lag1_metrics.get("sharpe_ratio")),
                    end=_fmt_num(lag1_metrics.get("normalized_end"), 4),
                    cagr=_fmt_pct(lag1_metrics.get("cagr")),
                    vol=_fmt_pct(lag1_metrics.get("annualized_volatility")),
                    dd=_fmt_pct(lag1_metrics.get("max_drawdown_pct")),
                    calmar=_fmt_num(lag1_metrics.get("calmar_ratio")),
                    start=lag1_start.strftime("%Y-%m-%d")
                    if lag1_start is not None
                    else "N/A",
                    finish=lag1_end.strftime("%Y-%m-%d")
                    if lag1_end is not None
                    else "N/A",
                )
            )

            report_lines.append("")
            report_lines.append("### Dynamic IEF Backfill (positions-based, lagged 2d)")
            lag2_entries = backfill._dynamic_ief_strategy(
                candidate_prices,
                config.paper_initial_capital,
                config.growth_ticker,
                lag_days=2,
                tag_suffix="lag_2d",
            )
            lag2_path = Path(
                f"reports/pnl_backfill_dynamic_ief_lag2_{start}_{end}.json"
            )
            lag2_path.write_text(json.dumps(lag2_entries, indent=2), encoding="utf-8")
            lag2_daily = _build_daily(lag2_entries)
            lag2_metrics = _compute_performance_metrics(
                lag2_daily,
                config.paper_initial_capital,
                config.risk_free_rate,
                config.pnl_downside_min_days,
            )
            lag2_normalized = _normalized_series(
                lag2_daily, config.paper_initial_capital
            )
            lag2_start = lag2_normalized.index.min()
            lag2_end = lag2_normalized.index.max()
            report_lines.append(f"Backfill report: {lag2_path}")
            report_lines.append(
                "| Sharpe | Final Equity | CAGR | Vol | Max DD | Calmar | Start | End |"
            )
            report_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
            report_lines.append(
                "| {sharpe} | {end} | {cagr} | {vol} | {dd} | {calmar} | {start} | {finish} |".format(
                    sharpe=_fmt_num(lag2_metrics.get("sharpe_ratio")),
                    end=_fmt_num(lag2_metrics.get("normalized_end"), 4),
                    cagr=_fmt_pct(lag2_metrics.get("cagr")),
                    vol=_fmt_pct(lag2_metrics.get("annualized_volatility")),
                    dd=_fmt_pct(lag2_metrics.get("max_drawdown_pct")),
                    calmar=_fmt_num(lag2_metrics.get("calmar_ratio")),
                    start=lag2_start.strftime("%Y-%m-%d")
                    if lag2_start is not None
                    else "N/A",
                    finish=lag2_end.strftime("%Y-%m-%d")
                    if lag2_end is not None
                    else "N/A",
                )
            )

            report_lines.append("")
            report_lines.append("### Dynamic IEF Backfill (positions-based, lagged 3d)")
            lag3_entries = backfill._dynamic_ief_strategy(
                candidate_prices,
                config.paper_initial_capital,
                config.growth_ticker,
                lag_days=3,
                tag_suffix="lag_3d",
            )
            lag3_path = Path(
                f"reports/pnl_backfill_dynamic_ief_lag3_{start}_{end}.json"
            )
            lag3_path.write_text(json.dumps(lag3_entries, indent=2), encoding="utf-8")
            lag3_daily = _build_daily(lag3_entries)
            lag3_metrics = _compute_performance_metrics(
                lag3_daily,
                config.paper_initial_capital,
                config.risk_free_rate,
                config.pnl_downside_min_days,
            )
            lag3_normalized = _normalized_series(
                lag3_daily, config.paper_initial_capital
            )
            lag3_start = lag3_normalized.index.min()
            lag3_end = lag3_normalized.index.max()
            report_lines.append(f"Backfill report: {lag3_path}")
            report_lines.append(
                "| Sharpe | Final Equity | CAGR | Vol | Max DD | Calmar | Start | End |"
            )
            report_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
            report_lines.append(
                "| {sharpe} | {end} | {cagr} | {vol} | {dd} | {calmar} | {start} | {finish} |".format(
                    sharpe=_fmt_num(lag3_metrics.get("sharpe_ratio")),
                    end=_fmt_num(lag3_metrics.get("normalized_end"), 4),
                    cagr=_fmt_pct(lag3_metrics.get("cagr")),
                    vol=_fmt_pct(lag3_metrics.get("annualized_volatility")),
                    dd=_fmt_pct(lag3_metrics.get("max_drawdown_pct")),
                    calmar=_fmt_num(lag3_metrics.get("calmar_ratio")),
                    start=lag3_start.strftime("%Y-%m-%d")
                    if lag3_start is not None
                    else "N/A",
                    finish=lag3_end.strftime("%Y-%m-%d")
                    if lag3_end is not None
                    else "N/A",
                )
            )

            report_lines.append("")
            report_lines.append("## Step 5: Hyperparameter Tuning")
            report_lines.append("### Dynamic IEF (synthetic grid)")
            dyn_cuts = [0.4, 0.45, 0.5, 0.55, 0.6]
            dyn_min_lev = [0.85, 0.9, 0.95]
            dyn_vix_caps = [30.0, 32.0, 34.0, 36.0]
            dyn_boosts = [0.1, 0.15, 0.2, 0.25, 0.3]
            dyn_thresholds = [26.0, 28.0, 30.0, 32.0]
            dyn_lags = [0, 1, 2, 3]
            dyn_sweep = _sweep_dynamic_ief_grid(
                config,
                candidate_prices,
                dyn_cuts,
                dyn_min_lev,
                dyn_vix_caps,
                dyn_boosts,
                dyn_thresholds,
                dyn_lags,
            )
            dyn_grid_path = Path(f"reports/tuning_dynamic_ief_grid_{start}_{end}.csv")
            if dyn_sweep:
                pd.DataFrame(dyn_sweep).to_csv(dyn_grid_path, index=False)
            top_n_grid = 5
            dyn_top = _top_n_rows_by_key(dyn_sweep, "score", top_n_grid)
            report_lines.append(
                "| Cut | Min Lev | VIX Cap | Boost | VIX Thresh | Lag | Sharpe | Final Equity | CAGR | Max DD | Score |"
            )
            report_lines.append(
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
            )
            for row in dyn_top:
                report_lines.append(
                    "| {cut} | {min_lev} | {vix_cap} | {boost} | {threshold} | {lag} | {sharpe} | {end} | {cagr} | {dd} | {score} |".format(
                        cut=_fmt_num(row.get("cut"), 2),
                        min_lev=_fmt_num(row.get("min_leverage"), 2),
                        vix_cap=_fmt_num(row.get("vix_cap"), 0),
                        boost=_fmt_num(row.get("ief_boost"), 2),
                        threshold=_fmt_num(row.get("vix_threshold"), 0),
                        lag=_fmt_num(row.get("lag"), 0),
                        sharpe=_fmt_num(row.get("sharpe")),
                        end=_fmt_num(row.get("final_equity"), 4),
                        cagr=_fmt_pct(row.get("cagr")),
                        dd=_fmt_pct(row.get("max_dd")),
                        score=_fmt_num(row.get("score")),
                    )
                )

            report_lines.append("")
            report_lines.append(f"### Dynamic IEF (positions-based top {top_n_grid})")
            report_lines.append(
                "| Cut | Min Lev | VIX Cap | Boost | VIX Thresh | Lag | Sharpe | Final Equity | CAGR | Max DD |"
            )
            report_lines.append(
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
            )
            dyn_top_rows: list[dict[str, float]] = []
            for idx, row in enumerate(dyn_top, start=1):
                backfill.DYNAMIC_IEF_CUT = float(
                    row.get("cut", backfill.DYNAMIC_IEF_CUT)
                )
                backfill.DYNAMIC_IEF_MIN_LEVERAGE = float(
                    row.get("min_leverage", backfill.DYNAMIC_IEF_MIN_LEVERAGE)
                )
                backfill.DYNAMIC_IEF_VIX_CAP = float(
                    row.get("vix_cap", backfill.DYNAMIC_IEF_VIX_CAP)
                )
                backfill.DYNAMIC_IEF_BOOST = float(
                    row.get("ief_boost", backfill.DYNAMIC_IEF_BOOST)
                )
                backfill.DYNAMIC_IEF_VIX_THRESHOLD = float(
                    row.get("vix_threshold", backfill.DYNAMIC_IEF_VIX_THRESHOLD)
                )
                lag_days = int(row.get("lag", 0))
                entries = backfill._dynamic_ief_strategy(
                    candidate_prices,
                    config.paper_initial_capital,
                    config.growth_ticker,
                    lag_days=lag_days,
                    tag_suffix=f"tune_{idx:02d}",
                )
                path = Path(
                    f"reports/pnl_backfill_dynamic_ief_tune_{idx:02d}_{start}_{end}.json"
                )
                path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
                daily = _build_daily(entries)
                metrics = _compute_performance_metrics(
                    daily,
                    config.paper_initial_capital,
                    config.risk_free_rate,
                    config.pnl_downside_min_days,
                )
                dyn_top_rows.append(
                    {
                        "cut": row.get("cut"),
                        "min_leverage": row.get("min_leverage"),
                        "vix_cap": row.get("vix_cap"),
                        "ief_boost": row.get("ief_boost"),
                        "vix_threshold": row.get("vix_threshold"),
                        "lag": row.get("lag"),
                        "sharpe": metrics.get("sharpe_ratio"),
                        "final_equity": metrics.get("normalized_end"),
                        "cagr": metrics.get("cagr"),
                        "max_dd": metrics.get("max_drawdown_pct"),
                    }
                )
                report_lines.append(
                    "| {cut} | {min_lev} | {vix_cap} | {boost} | {threshold} | {lag} | {sharpe} | {end} | {cagr} | {dd} |".format(
                        cut=_fmt_num(row.get("cut"), 2),
                        min_lev=_fmt_num(row.get("min_leverage"), 2),
                        vix_cap=_fmt_num(row.get("vix_cap"), 0),
                        boost=_fmt_num(row.get("ief_boost"), 2),
                        threshold=_fmt_num(row.get("vix_threshold"), 0),
                        lag=_fmt_num(row.get("lag"), 0),
                        sharpe=_fmt_num(metrics.get("sharpe_ratio")),
                        end=_fmt_num(metrics.get("normalized_end"), 4),
                        cagr=_fmt_pct(metrics.get("cagr")),
                        dd=_fmt_pct(metrics.get("max_drawdown_pct")),
                    )
                )
            dyn_top_path = Path(
                f"reports/tuning_dynamic_ief_top10_positions_{start}_{end}.csv"
            )
            if dyn_top_rows:
                pd.DataFrame(dyn_top_rows).to_csv(dyn_top_path, index=False)

            report_lines.append("")
            report_lines.append("### Dynamic IEF (positions-based local neighborhoods)")
            dyn_keys = [
                "cut",
                "min_leverage",
                "vix_cap",
                "ief_boost",
                "vix_threshold",
                "lag",
            ]
            dyn_values_by_key = {
                "cut": dyn_cuts,
                "min_leverage": dyn_min_lev,
                "vix_cap": dyn_vix_caps,
                "ief_boost": dyn_boosts,
                "vix_threshold": dyn_thresholds,
                "lag": [float(lag) for lag in dyn_lags],
            }
            dyn_lookup = _grid_lookup(dyn_sweep, dyn_keys)
            report_lines.append(
                "| Base | Variant | Cut | Min Lev | VIX Cap | Boost | VIX Thresh | Lag | Sharpe | Final Equity | CAGR | Max DD |"
            )
            report_lines.append(
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
            )
            for base_idx, base in enumerate(dyn_top[:3], start=1):
                variants = _local_neighborhood_rows(
                    base, dyn_keys, dyn_values_by_key, dyn_lookup
                )
                for var_idx, (label, row) in enumerate(variants, start=1):
                    backfill.DYNAMIC_IEF_CUT = float(
                        row.get("cut", backfill.DYNAMIC_IEF_CUT)
                    )
                    backfill.DYNAMIC_IEF_MIN_LEVERAGE = float(
                        row.get("min_leverage", backfill.DYNAMIC_IEF_MIN_LEVERAGE)
                    )
                    backfill.DYNAMIC_IEF_VIX_CAP = float(
                        row.get("vix_cap", backfill.DYNAMIC_IEF_VIX_CAP)
                    )
                    backfill.DYNAMIC_IEF_BOOST = float(
                        row.get("ief_boost", backfill.DYNAMIC_IEF_BOOST)
                    )
                    backfill.DYNAMIC_IEF_VIX_THRESHOLD = float(
                        row.get("vix_threshold", backfill.DYNAMIC_IEF_VIX_THRESHOLD)
                    )
                    lag_days = int(row.get("lag", 0))
                    entries = backfill._dynamic_ief_strategy(
                        candidate_prices,
                        config.paper_initial_capital,
                        config.growth_ticker,
                        lag_days=lag_days,
                        tag_suffix=f"tune_local_{base_idx}_{var_idx:02d}",
                    )
                    path = Path(
                        f"reports/pnl_backfill_dynamic_ief_local_{base_idx}_{var_idx:02d}_{start}_{end}.json"
                    )
                    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
                    daily = _build_daily(entries)
                    metrics = _compute_performance_metrics(
                        daily,
                        config.paper_initial_capital,
                        config.risk_free_rate,
                        config.pnl_downside_min_days,
                    )
                    report_lines.append(
                        "| {base} | {variant} | {cut} | {min_lev} | {vix_cap} | {boost} | {threshold} | {lag} | {sharpe} | {end} | {cagr} | {dd} |".format(
                            base=base_idx,
                            variant=label,
                            cut=_fmt_num(row.get("cut"), 2),
                            min_lev=_fmt_num(row.get("min_leverage"), 2),
                            vix_cap=_fmt_num(row.get("vix_cap"), 0),
                            boost=_fmt_num(row.get("ief_boost"), 2),
                            threshold=_fmt_num(row.get("vix_threshold"), 0),
                            lag=_fmt_num(row.get("lag"), 0),
                            sharpe=_fmt_num(metrics.get("sharpe_ratio")),
                            end=_fmt_num(metrics.get("normalized_end"), 4),
                            cagr=_fmt_pct(metrics.get("cagr")),
                            dd=_fmt_pct(metrics.get("max_drawdown_pct")),
                        )
                    )

            report_lines.append("")
            report_lines.append("### Zscore Throttle (positions-based grid)")
            z_cuts = [0.4, 0.45, 0.5, 0.55, 0.6]
            z_min_lev = [0.85, 0.9, 0.95]
            z_vix_caps = [30.0, 32.0, 34.0, 36.0]
            z_max_offs = [0.3, 0.4, 0.5]
            z_sweep = _sweep_zscore_grid(
                config,
                candidate_prices,
                z_cuts,
                z_min_lev,
                z_vix_caps,
                z_max_offs,
            )
            z_grid_path = Path(f"reports/tuning_zscore_grid_{start}_{end}.csv")
            if z_sweep:
                pd.DataFrame(z_sweep).to_csv(z_grid_path, index=False)
            z_top = _top_n_rows(z_sweep, 10)
            report_lines.append(
                "| Cut | Min Lev | VIX Cap | Max Off | Sharpe | Final Equity | CAGR | Max DD |"
            )
            report_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
            for row in z_top:
                report_lines.append(
                    "| {cut} | {min_lev} | {vix_cap} | {max_off} | {sharpe} | {end} | {cagr} | {dd} |".format(
                        cut=_fmt_num(row.get("cut"), 2),
                        min_lev=_fmt_num(row.get("min_leverage"), 2),
                        vix_cap=_fmt_num(row.get("vix_cap"), 0),
                        max_off=_fmt_num(row.get("max_off"), 2),
                        sharpe=_fmt_num(row.get("sharpe")),
                        end=_fmt_num(row.get("final_equity"), 4),
                        cagr=_fmt_pct(row.get("cagr")),
                        dd=_fmt_pct(row.get("max_dd")),
                    )
                )

            report_lines.append("")
            report_lines.append("### Zscore Throttle (positions-based top 10)")
            report_lines.append(
                "| Cut | Min Lev | VIX Cap | Max Off | Sharpe | Final Equity | CAGR | Max DD |"
            )
            report_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
            z_top_rows: list[dict[str, float]] = []
            for idx, row in enumerate(z_top, start=1):
                backfill.ZSCORE_CUT = float(row.get("cut", backfill.ZSCORE_CUT))
                backfill.ZSCORE_TREND_ON_MIN_LEVERAGE = float(
                    row.get("min_leverage", backfill.ZSCORE_TREND_ON_MIN_LEVERAGE)
                )
                backfill.ZSCORE_TREND_OFF_VIX = float(
                    row.get("vix_cap", backfill.ZSCORE_TREND_OFF_VIX)
                )
                backfill.ZSCORE_TREND_OFF_MAX_LEVERAGE = float(
                    row.get("max_off", backfill.ZSCORE_TREND_OFF_MAX_LEVERAGE)
                )
                entries = backfill._zscore_throttle_strategy(
                    candidate_prices,
                    config.paper_initial_capital,
                    config.growth_ticker,
                )
                path = Path(
                    f"reports/pnl_backfill_zscore_tune_{idx:02d}_{start}_{end}.json"
                )
                path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
                daily = _build_daily(entries)
                metrics = _compute_performance_metrics(
                    daily,
                    config.paper_initial_capital,
                    config.risk_free_rate,
                    config.pnl_downside_min_days,
                )
                z_top_rows.append(
                    {
                        "cut": row.get("cut"),
                        "min_leverage": row.get("min_leverage"),
                        "vix_cap": row.get("vix_cap"),
                        "max_off": row.get("max_off"),
                        "sharpe": metrics.get("sharpe_ratio"),
                        "final_equity": metrics.get("normalized_end"),
                        "cagr": metrics.get("cagr"),
                        "max_dd": metrics.get("max_drawdown_pct"),
                    }
                )
                report_lines.append(
                    "| {cut} | {min_lev} | {vix_cap} | {max_off} | {sharpe} | {end} | {cagr} | {dd} |".format(
                        cut=_fmt_num(row.get("cut"), 2),
                        min_lev=_fmt_num(row.get("min_leverage"), 2),
                        vix_cap=_fmt_num(row.get("vix_cap"), 0),
                        max_off=_fmt_num(row.get("max_off"), 2),
                        sharpe=_fmt_num(metrics.get("sharpe_ratio")),
                        end=_fmt_num(metrics.get("normalized_end"), 4),
                        cagr=_fmt_pct(metrics.get("cagr")),
                        dd=_fmt_pct(metrics.get("max_drawdown_pct")),
                    )
                )
            z_top_path = Path(
                f"reports/tuning_zscore_top10_positions_{start}_{end}.csv"
            )
            if z_top_rows:
                pd.DataFrame(z_top_rows).to_csv(z_top_path, index=False)

            report_lines.append("")
            report_lines.append(
                "### Zscore Throttle (positions-based local neighborhoods)"
            )
            z_local_rows: list[dict[str, object]] = []
            z_keys = ["cut", "min_leverage", "vix_cap", "max_off"]
            z_values_by_key = {
                "cut": z_cuts,
                "min_leverage": z_min_lev,
                "vix_cap": z_vix_caps,
                "max_off": z_max_offs,
            }
            z_lookup = _grid_lookup(z_sweep, z_keys)
            report_lines.append(
                "| Base | Variant | Cut | Min Lev | VIX Cap | Max Off | Sharpe | Final Equity | CAGR | Max DD |"
            )
            report_lines.append(
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
            )
            for base_idx, base in enumerate(z_top[:3], start=1):
                variants = _local_neighborhood_rows(
                    base, z_keys, z_values_by_key, z_lookup
                )
                for var_idx, (label, row) in enumerate(variants, start=1):
                    backfill.ZSCORE_CUT = float(row.get("cut", backfill.ZSCORE_CUT))
                    backfill.ZSCORE_TREND_ON_MIN_LEVERAGE = float(
                        row.get("min_leverage", backfill.ZSCORE_TREND_ON_MIN_LEVERAGE)
                    )
                    backfill.ZSCORE_TREND_OFF_VIX = float(
                        row.get("vix_cap", backfill.ZSCORE_TREND_OFF_VIX)
                    )
                    backfill.ZSCORE_TREND_OFF_MAX_LEVERAGE = float(
                        row.get("max_off", backfill.ZSCORE_TREND_OFF_MAX_LEVERAGE)
                    )
                    entries = backfill._zscore_throttle_strategy(
                        candidate_prices,
                        config.paper_initial_capital,
                        config.growth_ticker,
                    )
                    path = Path(
                        f"reports/pnl_backfill_zscore_local_{base_idx}_{var_idx:02d}_{start}_{end}.json"
                    )
                    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
                    daily = _build_daily(entries)
                    metrics = _compute_performance_metrics(
                        daily,
                        config.paper_initial_capital,
                        config.risk_free_rate,
                        config.pnl_downside_min_days,
                    )
                    z_local_rows.append(
                        {
                            "cut": row.get("cut"),
                            "min_leverage": row.get("min_leverage"),
                            "vix_cap": row.get("vix_cap"),
                            "max_off": row.get("max_off"),
                            "sharpe": metrics.get("sharpe_ratio"),
                            "final_equity": metrics.get("normalized_end"),
                            "cagr": metrics.get("cagr"),
                            "max_dd": metrics.get("max_drawdown_pct"),
                            "entries": entries,
                        }
                    )
                    report_lines.append(
                        "| {base} | {variant} | {cut} | {min_lev} | {vix_cap} | {max_off} | {sharpe} | {end} | {cagr} | {dd} |".format(
                            base=base_idx,
                            variant=label,
                            cut=_fmt_num(row.get("cut"), 2),
                            min_lev=_fmt_num(row.get("min_leverage"), 2),
                            vix_cap=_fmt_num(row.get("vix_cap"), 0),
                            max_off=_fmt_num(row.get("max_off"), 2),
                            sharpe=_fmt_num(metrics.get("sharpe_ratio")),
                            end=_fmt_num(metrics.get("normalized_end"), 4),
                            cagr=_fmt_pct(metrics.get("cagr")),
                            dd=_fmt_pct(metrics.get("max_drawdown_pct")),
                        )
                    )

            report_lines.append("")
            report_lines.append("### Dynamic IEF (focused vix_cap 34/36 grid)")
            dyn_focus_cuts = [0.4, 0.45]
            dyn_focus_min_lev = [0.85, 0.9]
            dyn_focus_vix_caps = [34.0, 36.0]
            dyn_focus_boosts = [0.25, 0.3]
            dyn_focus_thresholds = [26.0, 28.0, 30.0]
            dyn_focus_lags = [0, 1]
            dyn_focus = _sweep_dynamic_ief_positions_grid(
                config,
                candidate_prices,
                dyn_focus_cuts,
                dyn_focus_min_lev,
                dyn_focus_vix_caps,
                dyn_focus_boosts,
                dyn_focus_thresholds,
                dyn_focus_lags,
                tag_prefix="focus_vix",
            )
            dyn_focus_top = _top_n_rows(dyn_focus, 12)
            report_lines.append(
                "| Cut | Min Lev | VIX Cap | Boost | VIX Thresh | Lag | Sharpe | Final Equity | CAGR | Max DD |"
            )
            report_lines.append(
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
            )
            for row in dyn_focus_top:
                report_lines.append(
                    "| {cut} | {min_lev} | {vix_cap} | {boost} | {threshold} | {lag} | {sharpe} | {end} | {cagr} | {dd} |".format(
                        cut=_fmt_num(row.get("cut"), 2),
                        min_lev=_fmt_num(row.get("min_leverage"), 2),
                        vix_cap=_fmt_num(row.get("vix_cap"), 0),
                        boost=_fmt_num(row.get("ief_boost"), 2),
                        threshold=_fmt_num(row.get("vix_threshold"), 0),
                        lag=_fmt_num(row.get("lag"), 0),
                        sharpe=_fmt_num(row.get("sharpe")),
                        end=_fmt_num(row.get("final_equity"), 4),
                        cagr=_fmt_pct(row.get("cagr")),
                        dd=_fmt_pct(row.get("max_dd")),
                    )
                )

            report_lines.append("")
            report_lines.append("### Zscore Throttle (focused vix_cap 34/36 grid)")
            z_focus_cuts = [0.35, 0.4, 0.45]
            z_focus_min_lev = [0.9, 0.95]
            z_focus_vix_caps = [34.0, 36.0]
            z_focus_max_offs = [0.35, 0.4, 0.45]
            z_focus = _sweep_zscore_grid(
                config,
                candidate_prices,
                z_focus_cuts,
                z_focus_min_lev,
                z_focus_vix_caps,
                z_focus_max_offs,
            )
            z_focus_top = _top_n_rows(z_focus, 12)
            report_lines.append(
                "| Cut | Min Lev | VIX Cap | Max Off | Sharpe | Final Equity | CAGR | Max DD |"
            )
            report_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
            for row in z_focus_top:
                report_lines.append(
                    "| {cut} | {min_lev} | {vix_cap} | {max_off} | {sharpe} | {end} | {cagr} | {dd} |".format(
                        cut=_fmt_num(row.get("cut"), 2),
                        min_lev=_fmt_num(row.get("min_leverage"), 2),
                        vix_cap=_fmt_num(row.get("vix_cap"), 0),
                        max_off=_fmt_num(row.get("max_off"), 2),
                        sharpe=_fmt_num(row.get("sharpe")),
                        end=_fmt_num(row.get("final_equity"), 4),
                        cagr=_fmt_pct(row.get("cagr")),
                        dd=_fmt_pct(row.get("max_dd")),
                    )
                )

            report_lines.append("")
            report_lines.append(
                "### Dynamic IEF lag tradeoff micro-grid (vix_cap 34/36)"
            )
            report_lines.append(
                "Params: cut=0.40, min_leverage=0.85, boost=0.30, vix_threshold=26"
            )
            lag_focus = _sweep_dynamic_ief_positions_grid(
                config,
                candidate_prices,
                cuts=[0.4],
                min_leverages=[0.85],
                vix_caps=[34.0, 36.0],
                boosts=[0.3],
                thresholds=[26.0],
                lags=[0, 1, 2, 3],
                tag_prefix="lag_micro",
            )
            report_lines.append(
                "| VIX Cap | Lag | Sharpe | Final Equity | CAGR | Vol | Max DD |"
            )
            report_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
            for row in sorted(
                lag_focus, key=lambda x: (x.get("vix_cap"), x.get("lag"))
            ):
                report_lines.append(
                    "| {vix_cap} | {lag} | {sharpe} | {end} | {cagr} | {vol} | {dd} |".format(
                        vix_cap=_fmt_num(row.get("vix_cap"), 0),
                        lag=_fmt_num(row.get("lag"), 0),
                        sharpe=_fmt_num(row.get("sharpe")),
                        end=_fmt_num(row.get("final_equity"), 4),
                        cagr=_fmt_pct(row.get("cagr")),
                        vol=_fmt_pct(row.get("vol")),
                        dd=_fmt_pct(row.get("max_dd")),
                    )
                )

            report_lines.append("")
            report_lines.append("### Comparison (common window)")
            tuned_metrics = _metrics_for_dates(
                strategy_data["zscore_throttle_tuned"].daily,
                common_dates,
                config,
            )
            candidate_common_metrics = _metrics_for_dates(
                candidate_daily,
                common_dates,
                config,
            )
            positions_common = common_dates.intersection(positions_normalized.index)
            positions_common_metrics = _metrics_for_dates(
                positions_daily,
                positions_common,
                config,
            )
            lag1_common = common_dates.intersection(lag1_normalized.index)
            lag1_common_metrics = _metrics_for_dates(
                lag1_daily,
                lag1_common,
                config,
            )
            lag2_common = common_dates.intersection(lag2_normalized.index)
            lag2_common_metrics = _metrics_for_dates(
                lag2_daily,
                lag2_common,
                config,
            )
            lag3_common = common_dates.intersection(lag3_normalized.index)
            lag3_common_metrics = _metrics_for_dates(
                lag3_daily,
                lag3_common,
                config,
            )
            report_lines.append(
                "| Strategy | Sharpe | Final Equity | CAGR | Vol | Max DD | Calmar | Start | End |"
            )
            report_lines.append(
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
            )
            report_lines.append(
                "| zscore_throttle_tuned | {sharpe} | {end} | {cagr} | {vol} | {dd} | {calmar} | {start} | {finish} |".format(
                    sharpe=_fmt_num(tuned_metrics.get("sharpe_ratio")),
                    end=_fmt_num(tuned_metrics.get("normalized_end"), 4),
                    cagr=_fmt_pct(tuned_metrics.get("cagr")),
                    vol=_fmt_pct(tuned_metrics.get("annualized_volatility")),
                    dd=_fmt_pct(tuned_metrics.get("max_drawdown_pct")),
                    calmar=_fmt_num(tuned_metrics.get("calmar_ratio")),
                    start=start,
                    finish=end,
                )
            )
            report_lines.append(
                "| zscore_throttle_candidate | {sharpe} | {end} | {cagr} | {vol} | {dd} | {calmar} | {start} | {finish} |".format(
                    sharpe=_fmt_num(candidate_common_metrics.get("sharpe_ratio")),
                    end=_fmt_num(candidate_common_metrics.get("normalized_end"), 4),
                    cagr=_fmt_pct(candidate_common_metrics.get("cagr")),
                    vol=_fmt_pct(candidate_common_metrics.get("annualized_volatility")),
                    dd=_fmt_pct(candidate_common_metrics.get("max_drawdown_pct")),
                    calmar=_fmt_num(candidate_common_metrics.get("calmar_ratio")),
                    start=start,
                    finish=end,
                )
            )
            report_lines.append(
                "| dynamic_ief_positions | {sharpe} | {end} | {cagr} | {vol} | {dd} | {calmar} | {start} | {finish} |".format(
                    sharpe=_fmt_num(positions_common_metrics.get("sharpe_ratio")),
                    end=_fmt_num(positions_common_metrics.get("normalized_end"), 4),
                    cagr=_fmt_pct(positions_common_metrics.get("cagr")),
                    vol=_fmt_pct(positions_common_metrics.get("annualized_volatility")),
                    dd=_fmt_pct(positions_common_metrics.get("max_drawdown_pct")),
                    calmar=_fmt_num(positions_common_metrics.get("calmar_ratio")),
                    start=positions_common.min().strftime("%Y-%m-%d")
                    if not positions_common.empty
                    else "N/A",
                    finish=positions_common.max().strftime("%Y-%m-%d")
                    if not positions_common.empty
                    else "N/A",
                )
            )
            report_lines.append(
                "| dynamic_ief_lag1 | {sharpe} | {end} | {cagr} | {vol} | {dd} | {calmar} | {start} | {finish} |".format(
                    sharpe=_fmt_num(lag1_common_metrics.get("sharpe_ratio")),
                    end=_fmt_num(lag1_common_metrics.get("normalized_end"), 4),
                    cagr=_fmt_pct(lag1_common_metrics.get("cagr")),
                    vol=_fmt_pct(lag1_common_metrics.get("annualized_volatility")),
                    dd=_fmt_pct(lag1_common_metrics.get("max_drawdown_pct")),
                    calmar=_fmt_num(lag1_common_metrics.get("calmar_ratio")),
                    start=lag1_common.min().strftime("%Y-%m-%d")
                    if not lag1_common.empty
                    else "N/A",
                    finish=lag1_common.max().strftime("%Y-%m-%d")
                    if not lag1_common.empty
                    else "N/A",
                )
            )
            report_lines.append(
                "| dynamic_ief_lag2 | {sharpe} | {end} | {cagr} | {vol} | {dd} | {calmar} | {start} | {finish} |".format(
                    sharpe=_fmt_num(lag2_common_metrics.get("sharpe_ratio")),
                    end=_fmt_num(lag2_common_metrics.get("normalized_end"), 4),
                    cagr=_fmt_pct(lag2_common_metrics.get("cagr")),
                    vol=_fmt_pct(lag2_common_metrics.get("annualized_volatility")),
                    dd=_fmt_pct(lag2_common_metrics.get("max_drawdown_pct")),
                    calmar=_fmt_num(lag2_common_metrics.get("calmar_ratio")),
                    start=lag2_common.min().strftime("%Y-%m-%d")
                    if not lag2_common.empty
                    else "N/A",
                    finish=lag2_common.max().strftime("%Y-%m-%d")
                    if not lag2_common.empty
                    else "N/A",
                )
            )
            report_lines.append(
                "| dynamic_ief_lag3 | {sharpe} | {end} | {cagr} | {vol} | {dd} | {calmar} | {start} | {finish} |".format(
                    sharpe=_fmt_num(lag3_common_metrics.get("sharpe_ratio")),
                    end=_fmt_num(lag3_common_metrics.get("normalized_end"), 4),
                    cagr=_fmt_pct(lag3_common_metrics.get("cagr")),
                    vol=_fmt_pct(lag3_common_metrics.get("annualized_volatility")),
                    dd=_fmt_pct(lag3_common_metrics.get("max_drawdown_pct")),
                    calmar=_fmt_num(lag3_common_metrics.get("calmar_ratio")),
                    start=lag3_common.min().strftime("%Y-%m-%d")
                    if not lag3_common.empty
                    else "N/A",
                    finish=lag3_common.max().strftime("%Y-%m-%d")
                    if not lag3_common.empty
                    else "N/A",
                )
            )

            positions_rows = [
                {
                    "name": "zscore_throttle_tuned",
                    "sharpe": tuned_metrics.get("sharpe_ratio"),
                    "final_equity": tuned_metrics.get("normalized_end"),
                    "cagr": tuned_metrics.get("cagr"),
                    "max_dd": tuned_metrics.get("max_drawdown_pct"),
                },
                {
                    "name": "zscore_throttle_candidate",
                    "sharpe": candidate_common_metrics.get("sharpe_ratio"),
                    "final_equity": candidate_common_metrics.get("normalized_end"),
                    "cagr": candidate_common_metrics.get("cagr"),
                    "max_dd": candidate_common_metrics.get("max_drawdown_pct"),
                },
                {
                    "name": "dynamic_ief_positions",
                    "sharpe": positions_common_metrics.get("sharpe_ratio"),
                    "final_equity": positions_common_metrics.get("normalized_end"),
                    "cagr": positions_common_metrics.get("cagr"),
                    "max_dd": positions_common_metrics.get("max_drawdown_pct"),
                },
                {
                    "name": "dynamic_ief_lag1",
                    "sharpe": lag1_common_metrics.get("sharpe_ratio"),
                    "final_equity": lag1_common_metrics.get("normalized_end"),
                    "cagr": lag1_common_metrics.get("cagr"),
                    "max_dd": lag1_common_metrics.get("max_drawdown_pct"),
                },
                {
                    "name": "dynamic_ief_lag2",
                    "sharpe": lag2_common_metrics.get("sharpe_ratio"),
                    "final_equity": lag2_common_metrics.get("normalized_end"),
                    "cagr": lag2_common_metrics.get("cagr"),
                    "max_dd": lag2_common_metrics.get("max_drawdown_pct"),
                },
                {
                    "name": "dynamic_ief_lag3",
                    "sharpe": lag3_common_metrics.get("sharpe_ratio"),
                    "final_equity": lag3_common_metrics.get("normalized_end"),
                    "cagr": lag3_common_metrics.get("cagr"),
                    "max_dd": lag3_common_metrics.get("max_drawdown_pct"),
                },
            ]
            _append_scoreboard(
                report_lines,
                dyn_sweep if isinstance(dyn_sweep, list) else [],
                positions_rows,
                output_dir=checks_output_dir,
                slug=slugify("best_sharpe_returns"),
            )

            report_lines.append("")
            report_lines.append("## Selected Tables: Additional Checks")
            report_lines.append("")
            top_n = 5

            if z_local_rows:
                z_local_sorted = sorted(
                    z_local_rows,
                    key=lambda row: (
                        row.get("sharpe") if row.get("sharpe") is not None else -999,
                        row.get("final_equity")
                        if row.get("final_equity") is not None
                        else -999,
                    ),
                    reverse=True,
                )[:top_n]
                z_local_items: list[dict[str, object]] = []
                collector = _init_checks_collector()
                for idx, row in enumerate(z_local_sorted, start=1):
                    z_local_entries = row.get("entries")
                    if not isinstance(z_local_entries, list):
                        continue
                    z_local_strategy = _strategy_from_entries(
                        f"zscore_local_best_{idx}",
                        z_local_entries,
                        config.paper_initial_capital,
                    )
                    z_local_items.append(
                        {"strategy": z_local_strategy, "entries": z_local_entries}
                    )
                    _append_additional_checks(
                        report_lines,
                        f"Zscore Throttle local neighborhoods (rank {idx})",
                        z_local_strategy,
                        z_local_entries,
                        config,
                        collector=collector,
                    )
                _export_checks_csv(
                    checks_output_dir, slugify("zscore_local_neighborhoods"), collector
                )
                composite_rows = [
                    _composite_row(item["strategy"], item["entries"], config)
                    for item in z_local_items
                ]
                _append_composite_table(
                    report_lines,
                    "Zscore Throttle local neighborhoods",
                    composite_rows,
                    checks_output_dir,
                    slugify("zscore_local_neighborhoods"),
                )
                _append_deep_diagnostics(
                    report_lines,
                    "Zscore Throttle local neighborhoods",
                    z_local_items,
                    config,
                )

            if dyn_focus:
                dyn_focus_sorted = sorted(
                    dyn_focus,
                    key=lambda row: (
                        row.get("sharpe") if row.get("sharpe") is not None else -999,
                        row.get("final_equity")
                        if row.get("final_equity") is not None
                        else -999,
                    ),
                    reverse=True,
                )[:top_n]
                dyn_focus_items: list[dict[str, object]] = []
                collector = _init_checks_collector()
                for idx, row in enumerate(dyn_focus_sorted, start=1):
                    backfill.DYNAMIC_IEF_CUT = float(
                        row.get("cut", backfill.DYNAMIC_IEF_CUT)
                    )
                    backfill.DYNAMIC_IEF_MIN_LEVERAGE = float(
                        row.get("min_leverage", backfill.DYNAMIC_IEF_MIN_LEVERAGE)
                    )
                    backfill.DYNAMIC_IEF_VIX_CAP = float(
                        row.get("vix_cap", backfill.DYNAMIC_IEF_VIX_CAP)
                    )
                    backfill.DYNAMIC_IEF_BOOST = float(
                        row.get("ief_boost", backfill.DYNAMIC_IEF_BOOST)
                    )
                    backfill.DYNAMIC_IEF_VIX_THRESHOLD = float(
                        row.get("vix_threshold", backfill.DYNAMIC_IEF_VIX_THRESHOLD)
                    )
                    dyn_focus_entries = backfill._dynamic_ief_strategy(
                        candidate_prices,
                        config.paper_initial_capital,
                        config.growth_ticker,
                        lag_days=int(row.get("lag", 0)),
                        tag_suffix=f"focus_checks_{idx}",
                    )
                    dyn_focus_strategy = _strategy_from_entries(
                        f"dynamic_ief_focus_best_{idx}",
                        dyn_focus_entries,
                        config.paper_initial_capital,
                    )
                    dyn_focus_syn, dyn_focus_spy_w, dyn_focus_ief_w = (
                        _dynamic_ief_series(
                            candidate_prices,
                            cut=float(row.get("cut")),
                            min_leverage=float(row.get("min_leverage")),
                            vix_cap=float(row.get("vix_cap")),
                            ief_boost=float(row.get("ief_boost")),
                            vix_threshold=float(row.get("vix_threshold")),
                        )
                    )
                    dyn_focus_lag = int(row.get("lag", 0))
                    if dyn_focus_lag > 0:
                        dyn_focus_syn = _simulate_weighted_normalized(
                            candidate_prices["SPY"],
                            candidate_prices["IEF"],
                            dyn_focus_spy_w,
                            dyn_focus_ief_w,
                            lag=dyn_focus_lag,
                        )
                    dyn_focus_te = _tracking_error_metrics(
                        dyn_focus_syn, dyn_focus_strategy.normalized
                    )
                    dyn_focus_regime = _tracking_error_by_regime(
                        dyn_focus_syn,
                        dyn_focus_strategy.normalized,
                        config,
                    )
                    dyn_focus_items.append(
                        {
                            "strategy": dyn_focus_strategy,
                            "entries": dyn_focus_entries,
                            "synthetic_normalized": dyn_focus_syn,
                            "tracking_by_regime": dyn_focus_regime,
                        }
                    )
                    _append_additional_checks(
                        report_lines,
                        f"Dynamic IEF focused vix_cap 34/36 (rank {idx})",
                        dyn_focus_strategy,
                        dyn_focus_entries,
                        config,
                        tracking_error=dyn_focus_te,
                        collector=collector,
                    )
                _export_checks_csv(
                    checks_output_dir, slugify("dynamic_ief_focus_vix"), collector
                )
                composite_rows = [
                    _composite_row(item["strategy"], item["entries"], config)
                    for item in dyn_focus_items
                ]
                score_by_strategy = {
                    row.get("strategy"): row.get("score") for row in composite_rows
                }
                for item in dyn_focus_items:
                    strategy = item.get("strategy")
                    if isinstance(strategy, StrategyData):
                        item["composite_score"] = score_by_strategy.get(strategy.name)
                _append_composite_table(
                    report_lines,
                    "Dynamic IEF focused vix_cap 34/36",
                    composite_rows,
                    checks_output_dir,
                    slugify("dynamic_ief_focus_vix"),
                )
                _append_synth_vs_positions_table(
                    report_lines,
                    "Dynamic IEF focused vix_cap 34/36",
                    dyn_focus_items,
                    config,
                )
                _append_deep_diagnostics(
                    report_lines,
                    "Dynamic IEF focused vix_cap 34/36",
                    dyn_focus_items,
                    config,
                )

            if z_focus:
                z_focus_sorted = sorted(
                    z_focus,
                    key=lambda row: (
                        row.get("sharpe") if row.get("sharpe") is not None else -999,
                        row.get("final_equity")
                        if row.get("final_equity") is not None
                        else -999,
                    ),
                    reverse=True,
                )[:top_n]
                z_focus_items: list[dict[str, object]] = []
                collector = _init_checks_collector()
                for idx, row in enumerate(z_focus_sorted, start=1):
                    backfill.ZSCORE_CUT = float(row.get("cut", backfill.ZSCORE_CUT))
                    backfill.ZSCORE_TREND_ON_MIN_LEVERAGE = float(
                        row.get("min_leverage", backfill.ZSCORE_TREND_ON_MIN_LEVERAGE)
                    )
                    backfill.ZSCORE_TREND_OFF_VIX = float(
                        row.get("vix_cap", backfill.ZSCORE_TREND_OFF_VIX)
                    )
                    backfill.ZSCORE_TREND_OFF_MAX_LEVERAGE = float(
                        row.get("max_off", backfill.ZSCORE_TREND_OFF_MAX_LEVERAGE)
                    )
                    z_focus_entries = backfill._zscore_throttle_strategy(
                        candidate_prices,
                        config.paper_initial_capital,
                        config.growth_ticker,
                    )
                    z_focus_strategy = _strategy_from_entries(
                        f"zscore_focus_best_{idx}",
                        z_focus_entries,
                        config.paper_initial_capital,
                    )
                    z_focus_items.append(
                        {"strategy": z_focus_strategy, "entries": z_focus_entries}
                    )
                    _append_additional_checks(
                        report_lines,
                        f"Zscore Throttle focused vix_cap 34/36 (rank {idx})",
                        z_focus_strategy,
                        z_focus_entries,
                        config,
                        collector=collector,
                    )
                _export_checks_csv(
                    checks_output_dir, slugify("zscore_focus_vix"), collector
                )
                composite_rows = [
                    _composite_row(item["strategy"], item["entries"], config)
                    for item in z_focus_items
                ]
                _append_composite_table(
                    report_lines,
                    "Zscore Throttle focused vix_cap 34/36",
                    composite_rows,
                    checks_output_dir,
                    slugify("zscore_focus_vix"),
                )
                _append_deep_diagnostics(
                    report_lines,
                    "Zscore Throttle focused vix_cap 34/36",
                    z_focus_items,
                    config,
                )

            if lag_focus:
                lag_focus_sorted = sorted(
                    lag_focus,
                    key=lambda row: (
                        row.get("sharpe") if row.get("sharpe") is not None else -999,
                        row.get("final_equity")
                        if row.get("final_equity") is not None
                        else -999,
                    ),
                    reverse=True,
                )[:top_n]
                lag_focus_items: list[dict[str, object]] = []
                collector = _init_checks_collector()
                for idx, row in enumerate(lag_focus_sorted, start=1):
                    backfill.DYNAMIC_IEF_CUT = float(
                        row.get("cut", backfill.DYNAMIC_IEF_CUT)
                    )
                    backfill.DYNAMIC_IEF_MIN_LEVERAGE = float(
                        row.get("min_leverage", backfill.DYNAMIC_IEF_MIN_LEVERAGE)
                    )
                    backfill.DYNAMIC_IEF_VIX_CAP = float(
                        row.get("vix_cap", backfill.DYNAMIC_IEF_VIX_CAP)
                    )
                    backfill.DYNAMIC_IEF_BOOST = float(
                        row.get("ief_boost", backfill.DYNAMIC_IEF_BOOST)
                    )
                    backfill.DYNAMIC_IEF_VIX_THRESHOLD = float(
                        row.get("vix_threshold", backfill.DYNAMIC_IEF_VIX_THRESHOLD)
                    )
                    lag_best = int(row.get("lag", 0))
                    lag_focus_entries = backfill._dynamic_ief_strategy(
                        candidate_prices,
                        config.paper_initial_capital,
                        config.growth_ticker,
                        lag_days=lag_best,
                        tag_suffix=f"lag_checks_{idx}",
                    )
                    lag_focus_strategy = _strategy_from_entries(
                        f"dynamic_ief_lag_best_{idx}",
                        lag_focus_entries,
                        config.paper_initial_capital,
                    )
                    lag_focus_syn, lag_focus_spy_w, lag_focus_ief_w = (
                        _dynamic_ief_series(
                            candidate_prices,
                            cut=float(row.get("cut")),
                            min_leverage=float(row.get("min_leverage")),
                            vix_cap=float(row.get("vix_cap")),
                            ief_boost=float(row.get("ief_boost")),
                            vix_threshold=float(row.get("vix_threshold")),
                        )
                    )
                    if lag_best > 0:
                        lag_focus_syn = _simulate_weighted_normalized(
                            candidate_prices["SPY"],
                            candidate_prices["IEF"],
                            lag_focus_spy_w,
                            lag_focus_ief_w,
                            lag=lag_best,
                        )
                    lag_focus_te = _tracking_error_metrics(
                        lag_focus_syn, lag_focus_strategy.normalized
                    )
                    lag_focus_regime = _tracking_error_by_regime(
                        lag_focus_syn,
                        lag_focus_strategy.normalized,
                        config,
                    )
                    lag_focus_items.append(
                        {
                            "strategy": lag_focus_strategy,
                            "entries": lag_focus_entries,
                            "synthetic_normalized": lag_focus_syn,
                            "tracking_by_regime": lag_focus_regime,
                        }
                    )
                    _append_additional_checks(
                        report_lines,
                        f"Dynamic IEF lag tradeoff micro-grid (rank {idx})",
                        lag_focus_strategy,
                        lag_focus_entries,
                        config,
                        tracking_error=lag_focus_te,
                        collector=collector,
                    )
                _export_checks_csv(
                    checks_output_dir, slugify("dynamic_ief_lag_tradeoff"), collector
                )
                composite_rows = [
                    _composite_row(item["strategy"], item["entries"], config)
                    for item in lag_focus_items
                ]
                score_by_strategy = {
                    row.get("strategy"): row.get("score") for row in composite_rows
                }
                for item in lag_focus_items:
                    strategy = item.get("strategy")
                    if isinstance(strategy, StrategyData):
                        item["composite_score"] = score_by_strategy.get(strategy.name)
                _append_composite_table(
                    report_lines,
                    "Dynamic IEF lag tradeoff micro-grid",
                    composite_rows,
                    checks_output_dir,
                    slugify("dynamic_ief_lag_tradeoff"),
                )
                _append_synth_vs_positions_table(
                    report_lines,
                    "Dynamic IEF lag tradeoff micro-grid",
                    lag_focus_items,
                    config,
                )
                _append_deep_diagnostics(
                    report_lines,
                    "Dynamic IEF lag tradeoff micro-grid",
                    lag_focus_items,
                    config,
                )

            comparison_candidates = []
            tuned_entries = _load_entries(Path(REPORTS["zscore_throttle_tuned"]))
            comparison_candidates.append(
                {
                    "name": "zscore_throttle_tuned",
                    "metrics": tuned_metrics,
                    "strategy": strategy_data["zscore_throttle_tuned"],
                    "entries": tuned_entries,
                }
            )
            if isinstance(candidate_entries, list):
                comparison_candidates.append(
                    {
                        "name": "zscore_throttle_candidate",
                        "metrics": candidate_common_metrics,
                        "strategy": _strategy_from_entries(
                            "zscore_throttle_candidate",
                            candidate_entries,
                            config.paper_initial_capital,
                        ),
                        "entries": candidate_entries,
                    }
                )
            if isinstance(positions_entries, list):
                comparison_candidates.append(
                    {
                        "name": "dynamic_ief_positions",
                        "metrics": positions_common_metrics,
                        "strategy": _strategy_from_entries(
                            "dynamic_ief_positions",
                            positions_entries,
                            config.paper_initial_capital,
                        ),
                        "entries": positions_entries,
                    }
                )
            for name, entries, metrics in (
                ("dynamic_ief_lag1", lag1_entries, lag1_common_metrics),
                ("dynamic_ief_lag2", lag2_entries, lag2_common_metrics),
                ("dynamic_ief_lag3", lag3_entries, lag3_common_metrics),
            ):
                if isinstance(entries, list):
                    comparison_candidates.append(
                        {
                            "name": name,
                            "metrics": metrics,
                            "strategy": _strategy_from_entries(
                                name,
                                entries,
                                config.paper_initial_capital,
                            ),
                            "entries": entries,
                        }
                    )

            comparison_sorted = sorted(
                comparison_candidates,
                key=lambda row: row["metrics"].get("sharpe_ratio") or -999,
                reverse=True,
            )[:top_n]
            comparison_items: list[dict[str, object]] = []
            collector = _init_checks_collector()
            for idx, row in enumerate(comparison_sorted, start=1):
                tracking = None
                synthetic_normalized = None
                if row["name"].startswith("dynamic_ief") and isinstance(
                    row["entries"], list
                ):
                    dyn_syn, dyn_spy_w, dyn_ief_w = _dynamic_ief_series(
                        candidate_prices,
                        cut=float(backfill.DYNAMIC_IEF_CUT),
                        min_leverage=float(backfill.DYNAMIC_IEF_MIN_LEVERAGE),
                        vix_cap=float(backfill.DYNAMIC_IEF_VIX_CAP),
                        ief_boost=float(backfill.DYNAMIC_IEF_BOOST),
                        vix_threshold=float(backfill.DYNAMIC_IEF_VIX_THRESHOLD),
                    )
                    lag_match = 0
                    if row["name"].endswith("lag1"):
                        lag_match = 1
                    elif row["name"].endswith("lag2"):
                        lag_match = 2
                    elif row["name"].endswith("lag3"):
                        lag_match = 3
                    if lag_match > 0:
                        dyn_syn = _simulate_weighted_normalized(
                            candidate_prices["SPY"],
                            candidate_prices["IEF"],
                            dyn_spy_w,
                            dyn_ief_w,
                            lag=lag_match,
                        )
                    tracking = _tracking_error_metrics(
                        dyn_syn, row["strategy"].normalized
                    )
                    synthetic_normalized = dyn_syn
                comparison_items.append(
                    {
                        "strategy": row["strategy"],
                        "entries": row["entries"],
                        "synthetic_normalized": synthetic_normalized,
                    }
                )
                _append_additional_checks(
                    report_lines,
                    f"Comparison table (rank {idx})",
                    row["strategy"],
                    row["entries"],
                    config,
                    tracking_error=tracking,
                    collector=collector,
                )
            _export_checks_csv(
                checks_output_dir, slugify("comparison_table"), collector
            )
            composite_rows = [
                _composite_row(item["strategy"], item["entries"], config)
                for item in comparison_items
            ]
            score_by_strategy = {
                row.get("strategy"): row.get("score") for row in composite_rows
            }
            for item in comparison_items:
                strategy = item.get("strategy")
                if isinstance(strategy, StrategyData):
                    item["composite_score"] = score_by_strategy.get(strategy.name)
            _append_composite_table(
                report_lines,
                "Comparison table",
                composite_rows,
                checks_output_dir,
                slugify("comparison_table"),
            )
            _append_synth_vs_positions_table(
                report_lines,
                "Comparison table",
                comparison_items,
                config,
            )
            _append_deep_diagnostics(
                report_lines,
                "Comparison table",
                comparison_items,
                config,
            )

            report_lines.append("")
            report_lines.append(
                "## Why the real backfill does not beat the current best"
            )
            report_lines.append(
                "- The synthetic grid assumes daily weight targets applied cleanly; the positions-based backfill uses discrete fills and position PnL snapshots, which introduce tracking error vs the idealized equity curve."
            )
            report_lines.append(
                "- The real backfill accumulates realized and unrealized PnL across symbols and only takes the latest per-day unrealized snapshot, which can dampen the benefit of fast re-risking after drawdowns."
            )
            report_lines.append(
                "- The dynamic IEF boost shifts a small average weight to IEF, so the realized diversification is modest and cannot offset the equity curve drag from discrete position sizing."
            )
            report_lines.append(
                "- The tuned zscore strategy has higher exposure to the core SPY signal during trend-on windows, which dominates Sharpe and final equity in this sample window."
            )

            report_lines.append("")
            report_lines.append("## Deeper Investigation (Dynamic IEF)")
            report_lines.append("### 1) Tracking Error (synthetic vs positions-based)")
            if positions_normalized.empty:
                report_lines.append(
                    "- Positions-based backfill missing; tracking error skipped."
                )
            else:
                tracking = _tracking_error_metrics(dyn_normalized, positions_normalized)
                report_lines.append("| Metric | Value |")
                report_lines.append("| --- | --- |")
                report_lines.append(
                    "| Tracking error (annualized) | {value} |".format(
                        value=_fmt_pct(tracking.get("tracking_error")),
                    )
                )
                report_lines.append(
                    "| Return correlation | {value} |".format(
                        value=_fmt_num(tracking.get("return_corr"), 3),
                    )
                )
                report_lines.append(
                    "| Mean abs daily return diff | {value} |".format(
                        value=_fmt_pct(tracking.get("mean_abs_return_diff"), 3),
                    )
                )
                report_lines.append(
                    "| Max equity gap (normalized) | {value} |".format(
                        value=_fmt_num(tracking.get("max_equity_gap"), 3),
                    )
                )
                report_lines.append(
                    "| Max equity gap (pct) | {value} |".format(
                        value=_fmt_pct(tracking.get("max_equity_gap_pct")),
                    )
                )

                report_lines.append("")
                report_lines.append("#### Tracking Error by Regime")
                regime_te = _tracking_error_by_regime(
                    dyn_normalized,
                    positions_normalized,
                    config,
                )
                if regime_te:
                    report_lines.append(
                        "| Regime | Tracking error | Return corr | Mean abs diff | Points |"
                    )
                    report_lines.append("| --- | --- | --- | --- | --- |")
                    for regime in (
                        "vix_low",
                        "vix_mid",
                        "vix_high",
                        "trend_on",
                        "trend_off",
                    ):
                        metrics = regime_te.get(regime)
                        if not metrics:
                            continue
                        report_lines.append(
                            "| {regime} | {te} | {corr} | {diff} | {count} |".format(
                                regime=regime,
                                te=_fmt_pct(metrics.get("tracking_error")),
                                corr=_fmt_num(metrics.get("return_corr"), 3),
                                diff=_fmt_pct(metrics.get("mean_abs_return_diff"), 3),
                                count=metrics.get("count", 0),
                            )
                        )
                else:
                    report_lines.append("- Regime tracking error unavailable.")

            report_lines.append("")
            report_lines.append("### 2) Regime Attribution (positions-based)")
            if positions_normalized.empty:
                report_lines.append(
                    "- Positions-based backfill missing; regime attribution skipped."
                )
            else:
                regime_metrics = _regime_breakdown_single(
                    positions_normalized,
                    positions_daily,
                    config,
                )
                report_lines.append(
                    "| Regime | Sharpe | Final Equity | CAGR | Vol | Max DD | Calmar | Points |"
                )
                report_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
                for regime in (
                    "vix_low",
                    "vix_mid",
                    "vix_high",
                    "trend_on",
                    "trend_off",
                    "dd_0_5",
                    "dd_5_15",
                    "dd_15_plus",
                ):
                    metrics = regime_metrics.get(regime, {})
                    report_lines.append(
                        "| {regime} | {sharpe} | {end} | {cagr} | {vol} | {dd} | {calmar} | {points} |".format(
                            regime=regime,
                            sharpe=_fmt_num(metrics.get("sharpe_ratio")),
                            end=_fmt_num(metrics.get("normalized_end"), 4),
                            cagr=_fmt_pct(metrics.get("cagr")),
                            vol=_fmt_pct(metrics.get("annualized_volatility")),
                            dd=_fmt_pct(metrics.get("max_drawdown_pct")),
                            calmar=_fmt_num(metrics.get("calmar_ratio")),
                            points=metrics.get("data_points", 0),
                        )
                    )

            report_lines.append("")
            report_lines.append("### 3) Turnover vs weight changes")
            if positions_entries:
                turnover = _compute_turnover_series(
                    positions_entries,
                    config.paper_initial_capital,
                )
                turnover_by_ticker = _compute_turnover_by_ticker(
                    positions_entries,
                    config.paper_initial_capital,
                )
            else:
                turnover = pd.Series(dtype=float)
                turnover_by_ticker = pd.DataFrame()
            weight_turnover = _compute_weight_turnover(
                dyn_spy_weights,
                dyn_ief_weights,
            )
            report_lines.append("| Metric | Positions-based | Synthetic weights |")
            report_lines.append("| --- | --- | --- |")
            report_lines.append(
                "| Avg daily turnover | {pos} | {syn} |".format(
                    pos=_fmt_pct(turnover.mean()) if not turnover.empty else "N/A",
                    syn=_fmt_pct(weight_turnover.mean())
                    if not weight_turnover.empty
                    else "N/A",
                )
            )
            report_lines.append(
                "| Median daily turnover | {pos} | {syn} |".format(
                    pos=_fmt_pct(turnover.median()) if not turnover.empty else "N/A",
                    syn=_fmt_pct(weight_turnover.median())
                    if not weight_turnover.empty
                    else "N/A",
                )
            )
            report_lines.append(
                "| 95th pct daily turnover | {pos} | {syn} |".format(
                    pos=_fmt_pct(turnover.quantile(0.95))
                    if not turnover.empty
                    else "N/A",
                    syn=_fmt_pct(weight_turnover.quantile(0.95))
                    if not weight_turnover.empty
                    else "N/A",
                )
            )
            report_lines.append(
                "| Days with turnover > 10% | {pos} | {syn} |".format(
                    pos=_fmt_pct((turnover > 0.10).mean())
                    if not turnover.empty
                    else "N/A",
                    syn=_fmt_pct((weight_turnover > 0.10).mean())
                    if not weight_turnover.empty
                    else "N/A",
                )
            )
            if not turnover_by_ticker.empty:
                report_lines.append("")
                report_lines.append("#### Turnover by Ticker (positions-based)")
                report_lines.append(
                    "| Ticker | Avg daily | Median | 95th pct | Days > 10% |"
                )
                report_lines.append("| --- | --- | --- | --- | --- |")
                for ticker in turnover_by_ticker.columns:
                    series = turnover_by_ticker[ticker].dropna()
                    if series.empty:
                        continue
                    report_lines.append(
                        "| {ticker} | {avg} | {med} | {p95} | {hit} |".format(
                            ticker=ticker,
                            avg=_fmt_pct(series.mean()),
                            med=_fmt_pct(series.median()),
                            p95=_fmt_pct(series.quantile(0.95)),
                            hit=_fmt_pct((series > 0.10).mean()),
                        )
                    )
            report_lines.append(
                "- Note: backfill entries do not include commissions or slippage; turnover is an activity proxy."
            )

            report_lines.append("")
            report_lines.append("### 4) Equity gap by regime")
            if positions_normalized.empty:
                report_lines.append(
                    "- Positions-based backfill missing; equity gap skipped."
                )
            else:
                gap_metrics = _equity_gap_by_regime(
                    dyn_normalized,
                    positions_normalized,
                    config,
                )
                if gap_metrics:
                    report_lines.append(
                        "| Regime | Mean abs gap | Median abs gap | Max abs gap | Points |"
                    )
                    report_lines.append("| --- | --- | --- | --- | --- |")
                    for regime in (
                        "vix_low",
                        "vix_mid",
                        "vix_high",
                        "trend_on",
                        "trend_off",
                    ):
                        metrics = gap_metrics.get(regime)
                        if not metrics:
                            continue
                        report_lines.append(
                            "| {regime} | {mean_gap} | {median_gap} | {max_gap} | {count} |".format(
                                regime=regime,
                                mean_gap=_fmt_pct(metrics.get("mean_abs_gap"), 3),
                                median_gap=_fmt_pct(metrics.get("median_abs_gap"), 3),
                                max_gap=_fmt_pct(metrics.get("max_abs_gap"), 3),
                                count=metrics.get("count", 0),
                            )
                        )
                else:
                    report_lines.append("- Equity gap metrics unavailable.")

            report_lines.append("")
            report_lines.append("### 5) Return gap attribution by regime")
            if positions_entries:
                gap_attr = _return_gap_by_regime(
                    dyn_normalized,
                    positions_normalized,
                    dyn_spy_weights,
                    dyn_ief_weights,
                    positions_entries,
                    config,
                )
            else:
                gap_attr = {}
            if gap_attr:
                report_lines.append(
                    "| Regime | SPY gap | IEF gap | Total gap | Points |"
                )
                report_lines.append("| --- | --- | --- | --- | --- |")
                for regime in (
                    "vix_low",
                    "vix_mid",
                    "vix_high",
                    "trend_on",
                    "trend_off",
                ):
                    metrics = gap_attr.get(regime)
                    if not metrics:
                        continue
                    report_lines.append(
                        "| {regime} | {spy_gap} | {ief_gap} | {total_gap} | {count} |".format(
                            regime=regime,
                            spy_gap=_fmt_bps(metrics.get("spy_gap"), 2),
                            ief_gap=_fmt_bps(metrics.get("ief_gap"), 2),
                            total_gap=_fmt_bps(metrics.get("total_gap"), 2),
                            count=metrics.get("count", 0),
                        )
                    )
            else:
                report_lines.append("- Return gap attribution unavailable.")

            report_lines.append("")
            report_lines.append(
                "### 6) Signal slippage proxy (target vs realized weights)"
            )
            if positions_entries:
                slippage = _signal_slippage_stats(
                    dyn_spy_weights,
                    dyn_ief_weights,
                    positions_entries,
                )
            else:
                slippage = {}
            if slippage:
                report_lines.append("| Metric | Value |")
                report_lines.append("| --- | --- |")
                report_lines.append(
                    "| Avg abs weight gap | {value} |".format(
                        value=_fmt_pct(slippage.get("avg_abs_gap"), 3),
                    )
                )
                report_lines.append(
                    "| Median abs weight gap | {value} |".format(
                        value=_fmt_pct(slippage.get("median_abs_gap"), 3),
                    )
                )
                report_lines.append(
                    "| 95th pct abs weight gap | {value} |".format(
                        value=_fmt_pct(slippage.get("p95_abs_gap"), 3),
                    )
                )
                report_lines.append(
                    "| Days with gap > 5% | {value} |".format(
                        value=_fmt_pct(slippage.get("pct_gt_5"), 2),
                    )
                )
            else:
                report_lines.append("- Signal slippage unavailable.")

            report_lines.append("")
            report_lines.append("### 7) Equity gap by month and regime")
            if positions_normalized.empty:
                report_lines.append(
                    "- Positions-based backfill missing; monthly equity gap skipped."
                )
            else:
                monthly_gap = _equity_gap_monthly_by_regime(
                    dyn_normalized,
                    positions_normalized,
                    config,
                )
                if monthly_gap:
                    for regime, table in monthly_gap.items():
                        if table.empty:
                            continue
                        report_lines.append("")
                        report_lines.append(f"#### {regime}")
                        report_lines.append(
                            "| Month | Mean abs gap | Median abs gap | Max abs gap | Points |"
                        )
                        report_lines.append("| --- | --- | --- | --- | --- |")
                        for _, row in table.iterrows():
                            report_lines.append(
                                "| {month} | {mean_gap} | {median_gap} | {max_gap} | {count} |".format(
                                    month=row.get("month"),
                                    mean_gap=_fmt_pct(row.get("mean"), 3),
                                    median_gap=_fmt_pct(row.get("median"), 3),
                                    max_gap=_fmt_pct(row.get("max"), 3),
                                    count=int(row.get("count", 0)),
                                )
                            )
                else:
                    report_lines.append("- Monthly equity gap unavailable.")

            report_lines.append("")
            report_lines.append("### 8) Signal slippage spikes (top 10 days)")
            if positions_entries:
                spikes = _signal_slippage_spikes(
                    dyn_spy_weights,
                    dyn_ief_weights,
                    positions_entries,
                    config,
                    top_n=10,
                )
            else:
                spikes = []
            if spikes:
                report_lines.append(
                    "| Date | Gap | SPY target | SPY realized | IEF target | IEF realized | VIX | Trend |"
                )
                report_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
                for spike in spikes:
                    report_lines.append(
                        "| {date} | {gap} | {spy_t} | {spy_r} | {ief_t} | {ief_r} | {vix} | {trend} |".format(
                            date=spike.get("date"),
                            gap=_fmt_pct(spike.get("gap"), 3),
                            spy_t=_fmt_pct(spike.get("spy_target"), 3),
                            spy_r=_fmt_pct(spike.get("spy_realized"), 3),
                            ief_t=_fmt_pct(spike.get("ief_target"), 3),
                            ief_r=_fmt_pct(spike.get("ief_realized"), 3),
                            vix=_fmt_num(spike.get("vix"), 2),
                            trend="on" if spike.get("trend_on") else "off",
                        )
                    )
            else:
                report_lines.append("- No slippage spikes available.")

            report_lines.append("")
            report_lines.append("### 9) Turnover by month (positions-based)")
            if positions_entries:
                monthly_turnover = _turnover_by_month(turnover)
            else:
                monthly_turnover = pd.DataFrame()
            if not monthly_turnover.empty:
                report_lines.append(
                    "| Month | Avg turnover | Median | Max | Days > 10% | Points |"
                )
                report_lines.append("| --- | --- | --- | --- | --- | --- |")
                for _, row in monthly_turnover.iterrows():
                    report_lines.append(
                        "| {month} | {mean} | {median} | {max} | {pct} | {count} |".format(
                            month=row.get("month"),
                            mean=_fmt_pct(row.get("mean"), 3),
                            median=_fmt_pct(row.get("median"), 3),
                            max=_fmt_pct(row.get("max"), 3),
                            pct=_fmt_pct(row.get("pct_gt_10"), 2),
                            count=int(row.get("count", 0)),
                        )
                    )
            else:
                report_lines.append("- Monthly turnover unavailable.")

            report_lines.append("")
            report_lines.append("### 10) Execution parity check (VIX-triggered days)")
            if positions_entries:
                spy_series, vix_series = _load_regime_market(
                    positions_normalized.index,
                    config,
                )
                pos_spy, pos_ief = _positions_weights_series(positions_entries)
                parity = _execution_parity_stats(
                    dyn_ief_weights,
                    pos_ief,
                    vix_series,
                    vix_threshold=28.0,
                )
            else:
                parity = []
            if parity:
                report_lines.append(
                    "| Lag (days) | Mean abs gap | % within 2% | Avg target | Avg realized | Points |"
                )
                report_lines.append("| --- | --- | --- | --- | --- | --- |")
                for row in parity:
                    report_lines.append(
                        "| {lag} | {gap} | {within} | {target} | {realized} | {count} |".format(
                            lag=int(row.get("lag", 0)),
                            gap=_fmt_pct(row.get("mean_abs_gap"), 3),
                            within=_fmt_pct(row.get("pct_within"), 2),
                            target=_fmt_pct(row.get("avg_target"), 2),
                            realized=_fmt_pct(row.get("avg_realized"), 2),
                            count=int(row.get("count", 0)),
                        )
                    )
            else:
                report_lines.append("- Execution parity check unavailable.")

            report_lines.append("")
            report_lines.append("### 11) Return gap decomposition (sizing effects)")
            if positions_entries:
                cache_dir = Path(config.cache_dir)
                cached = _load_cached_prices_long(cache_dir)
                if cached is not None:
                    spy_prices = cached["SPY"].copy()
                    ief_prices = cached["IEF"].copy()
                else:
                    price_data = _load_backfill_module()._download_prices(
                        ["SPY", "IEF"],
                        dyn_normalized.index.min().strftime("%Y-%m-%d"),
                        (dyn_normalized.index.max() + pd.Timedelta(days=1)).strftime(
                            "%Y-%m-%d"
                        ),
                    )
                    spy_prices = price_data["SPY"]
                    ief_prices = price_data["IEF"]
                spy_prices.index = pd.to_datetime(
                    spy_prices.index, utc=True
                ).tz_convert(None)
                ief_prices.index = pd.to_datetime(
                    ief_prices.index, utc=True
                ).tz_convert(None)
                pos_spy, pos_ief = _positions_weights_series(positions_entries)
                decomposition = _return_gap_decomposition(
                    dyn_spy_weights,
                    dyn_ief_weights,
                    pos_spy,
                    pos_ief,
                    spy_prices,
                    ief_prices,
                )
            else:
                decomposition = {}
            if decomposition:
                report_lines.append("| Component | Mean daily gap | Cumulative gap |")
                report_lines.append("| --- | --- | --- |")
                report_lines.append(
                    "| SPY sizing | {mean} | {cum} |".format(
                        mean=_fmt_bps(decomposition.get("spy_gap_mean"), 2),
                        cum=_fmt_pct(decomposition.get("spy_gap_cum"), 3),
                    )
                )
                report_lines.append(
                    "| IEF sizing | {mean} | {cum} |".format(
                        mean=_fmt_bps(decomposition.get("ief_gap_mean"), 2),
                        cum=_fmt_pct(decomposition.get("ief_gap_cum"), 3),
                    )
                )
                report_lines.append(
                    "| Total | {mean} | {cum} |".format(
                        mean=_fmt_bps(decomposition.get("total_gap_mean"), 2),
                        cum=_fmt_pct(decomposition.get("total_gap_cum"), 3),
                    )
                )
            else:
                report_lines.append("- Return gap decomposition unavailable.")

            report_lines.append("")
            report_lines.append("### 12) Event study (VIX threshold crossings)")
            if positions_entries:
                _, vix_series = _load_regime_market(
                    positions_normalized.index,
                    config,
                )
                event_study = _event_study_vix_cross(
                    dyn_normalized,
                    positions_normalized,
                    vix_series,
                    threshold=28.0,
                )
            else:
                event_study = {}
            if event_study:
                report_lines.append(
                    "| Horizon | Synthetic mean | Positions mean | Gap | Events |"
                )
                report_lines.append("| --- | --- | --- | --- | --- |")
                for horizon in (5, 10, 20):
                    syn_value = event_study.get(f"syn_{horizon}")
                    pos_value = event_study.get(f"pos_{horizon}")
                    gap_value = event_study.get(f"gap_{horizon}")
                    report_lines.append(
                        "| {h}d | {syn} | {pos} | {gap} | {events} |".format(
                            h=horizon,
                            syn=_fmt_pct(syn_value, 3),
                            pos=_fmt_pct(pos_value, 3),
                            gap=_fmt_pct(gap_value, 3),
                            events=int(event_study.get("events", 0)),
                        )
                    )
            else:
                report_lines.append("- Event study unavailable.")

            report_lines.append("")
            report_lines.append("### 13) Cost stress test (positions-based)")
            if positions_entries:
                costs = []
                for cost_bps in (5.0, 10.0, 20.0):
                    metrics = _apply_cost_stress_test(
                        positions_daily,
                        turnover,
                        config,
                        cost_bps=cost_bps,
                    )
                    if metrics:
                        costs.append(metrics)
            else:
                costs = []
            if costs:
                report_lines.append(
                    "| Cost (bps) | Sharpe | Final Equity | CAGR | Vol | Max DD | Calmar |"
                )
                report_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
                for metrics in costs:
                    report_lines.append(
                        "| {bps} | {sharpe} | {end} | {cagr} | {vol} | {dd} | {calmar} |".format(
                            bps=_fmt_num(metrics.get("cost_bps"), 0),
                            sharpe=_fmt_num(metrics.get("sharpe_ratio")),
                            end=_fmt_num(metrics.get("normalized_end"), 4),
                            cagr=_fmt_pct(metrics.get("cagr")),
                            vol=_fmt_pct(metrics.get("annualized_volatility")),
                            dd=_fmt_pct(metrics.get("max_drawdown_pct")),
                            calmar=_fmt_num(metrics.get("calmar_ratio")),
                        )
                    )
            else:
                report_lines.append("- Cost stress test unavailable.")

            report_lines.append("")
            report_lines.append("### 14) Same-day vs next-day rebalancing")
            if positions_entries:
                cache_dir = Path(config.cache_dir)
                cached = _load_cached_prices_long(cache_dir)
                if cached is not None:
                    spy_prices = cached["SPY"].copy()
                    ief_prices = cached["IEF"].copy()
                else:
                    price_data = _load_backfill_module()._download_prices(
                        ["SPY", "IEF"],
                        dyn_normalized.index.min().strftime("%Y-%m-%d"),
                        (dyn_normalized.index.max() + pd.Timedelta(days=1)).strftime(
                            "%Y-%m-%d"
                        ),
                    )
                    spy_prices = price_data["SPY"]
                    ief_prices = price_data["IEF"]
                spy_prices.index = pd.to_datetime(
                    spy_prices.index, utc=True
                ).tz_convert(None)
                ief_prices.index = pd.to_datetime(
                    ief_prices.index, utc=True
                ).tz_convert(None)
                lag0 = _simulate_weighted_normalized(
                    spy_prices, ief_prices, dyn_spy_weights, dyn_ief_weights, lag=0
                )
                lag1 = _simulate_weighted_normalized(
                    spy_prices, ief_prices, dyn_spy_weights, dyn_ief_weights, lag=1
                )
                lag0_metrics = _compute_performance_metrics(
                    [
                        {
                            "date": idx.strftime("%Y-%m-%d"),
                            "equity": (value - 1.0) * config.paper_initial_capital,
                        }
                        for idx, value in lag0.items()
                    ],
                    config.paper_initial_capital,
                    config.risk_free_rate,
                    config.pnl_downside_min_days,
                )
                lag1_metrics = _compute_performance_metrics(
                    [
                        {
                            "date": idx.strftime("%Y-%m-%d"),
                            "equity": (value - 1.0) * config.paper_initial_capital,
                        }
                        for idx, value in lag1.items()
                    ],
                    config.paper_initial_capital,
                    config.risk_free_rate,
                    config.pnl_downside_min_days,
                )
                lag0_te = _tracking_error_metrics(lag0, positions_normalized)
                lag1_te = _tracking_error_metrics(lag1, positions_normalized)
                report_lines.append(
                    "| Lag | Sharpe | Final Equity | CAGR | Vol | Max DD | TE |"
                )
                report_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
                report_lines.append(
                    "| 0d | {sharpe} | {end} | {cagr} | {vol} | {dd} | {te} |".format(
                        sharpe=_fmt_num(lag0_metrics.get("sharpe_ratio")),
                        end=_fmt_num(lag0_metrics.get("normalized_end"), 4),
                        cagr=_fmt_pct(lag0_metrics.get("cagr")),
                        vol=_fmt_pct(lag0_metrics.get("annualized_volatility")),
                        dd=_fmt_pct(lag0_metrics.get("max_drawdown_pct")),
                        te=_fmt_pct(lag0_te.get("tracking_error")),
                    )
                )
                report_lines.append(
                    "| 1d | {sharpe} | {end} | {cagr} | {vol} | {dd} | {te} |".format(
                        sharpe=_fmt_num(lag1_metrics.get("sharpe_ratio")),
                        end=_fmt_num(lag1_metrics.get("normalized_end"), 4),
                        cagr=_fmt_pct(lag1_metrics.get("cagr")),
                        vol=_fmt_pct(lag1_metrics.get("annualized_volatility")),
                        dd=_fmt_pct(lag1_metrics.get("max_drawdown_pct")),
                        te=_fmt_pct(lag1_te.get("tracking_error")),
                    )
                )
            else:
                report_lines.append("- Rebalancing comparison unavailable.")

            report_lines.append("")
            report_lines.append("### 15) Execution delay simulator (0-5 day lag)")
            if positions_entries:
                report_lines.append(
                    "| Lag | Tracking error | Return corr | Mean abs diff |"
                )
                report_lines.append("| --- | --- | --- | --- |")
                for lag in range(6):
                    simulated = _simulate_weighted_normalized(
                        spy_prices,
                        ief_prices,
                        dyn_spy_weights,
                        dyn_ief_weights,
                        lag=lag,
                    )
                    te = _tracking_error_metrics(simulated, positions_normalized)
                    report_lines.append(
                        "| {lag} | {te} | {corr} | {diff} |".format(
                            lag=lag,
                            te=_fmt_pct(te.get("tracking_error")),
                            corr=_fmt_num(te.get("return_corr"), 3),
                            diff=_fmt_pct(te.get("mean_abs_return_diff"), 3),
                        )
                    )
            else:
                report_lines.append("- Execution delay simulator unavailable.")

            report_lines.append("")
            report_lines.append(
                "### 16) VIX threshold / step-up stress test (synthetic)"
            )
            if positions_entries:
                report_lines.append(
                    "| VIX threshold | IEF boost | Sharpe | Final Equity | Max DD | Avg turnover | Days > 10% |"
                )
                report_lines.append("| --- | --- | --- | --- | --- | --- | --- |")
                for threshold in (26.0, 28.0, 30.0, 32.0):
                    for boost in (0.1, 0.2, 0.3):
                        syn_norm, syn_spy, syn_ief = _dynamic_ief_series(
                            candidate_prices,
                            cut=0.5,
                            min_leverage=0.9,
                            vix_cap=32.0,
                            ief_boost=boost,
                            vix_threshold=threshold,
                        )
                        if syn_norm.empty:
                            continue
                        syn_daily = [
                            {
                                "date": idx.strftime("%Y-%m-%d"),
                                "equity": (value - 1.0) * config.paper_initial_capital,
                            }
                            for idx, value in syn_norm.items()
                        ]
                        syn_metrics = _compute_performance_metrics(
                            syn_daily,
                            config.paper_initial_capital,
                            config.risk_free_rate,
                            config.pnl_downside_min_days,
                        )
                        syn_turnover = _compute_weight_turnover(syn_spy, syn_ief)
                        report_lines.append(
                            "| {threshold} | {boost} | {sharpe} | {end} | {dd} | {avg} | {hit} |".format(
                                threshold=_fmt_num(threshold, 0),
                                boost=_fmt_num(boost, 2),
                                sharpe=_fmt_num(syn_metrics.get("sharpe_ratio")),
                                end=_fmt_num(syn_metrics.get("normalized_end"), 4),
                                dd=_fmt_pct(syn_metrics.get("max_drawdown_pct")),
                                avg=_fmt_pct(syn_turnover.mean())
                                if not syn_turnover.empty
                                else "N/A",
                                hit=_fmt_pct((syn_turnover > 0.10).mean())
                                if not syn_turnover.empty
                                else "N/A",
                            )
                        )
            else:
                report_lines.append("- VIX threshold stress test unavailable.")

            report_lines.append("")
            report_lines.append("## Deeper Investigation (Extended)")
            report_lines.append("### 17) Tracking error by regime vs lag (0-3d)")
            if positions_entries:
                cache_dir = Path(config.cache_dir)
                cached = _load_cached_prices_long(cache_dir)
                if cached is not None:
                    spy_prices = cached["SPY"].copy()
                    ief_prices = cached["IEF"].copy()
                else:
                    price_data = _load_backfill_module()._download_prices(
                        ["SPY", "IEF"],
                        dyn_normalized.index.min().strftime("%Y-%m-%d"),
                        (dyn_normalized.index.max() + pd.Timedelta(days=1)).strftime(
                            "%Y-%m-%d"
                        ),
                    )
                    spy_prices = price_data["SPY"]
                    ief_prices = price_data["IEF"]
                spy_prices.index = pd.to_datetime(
                    spy_prices.index, utc=True
                ).tz_convert(None)
                ief_prices.index = pd.to_datetime(
                    ief_prices.index, utc=True
                ).tz_convert(None)
                lag_te_rows: dict[int, dict[str, dict[str, float]]] = {}
                for lag in range(4):
                    if lag == 0:
                        simulated = dyn_normalized
                    else:
                        simulated = _simulate_weighted_normalized(
                            spy_prices,
                            ief_prices,
                            dyn_spy_weights,
                            dyn_ief_weights,
                            lag=lag,
                        )
                    lag_te_rows[lag] = _tracking_error_by_regime(
                        simulated, positions_normalized, config
                    )
                te_table = _tracking_error_by_regime_table(lag_te_rows)
                if te_table:
                    report_lines.append(
                        "| Lag | vix_low | vix_mid | vix_high | trend_on | trend_off |"
                    )
                    report_lines.append("| --- | --- | --- | --- | --- | --- |")
                    for row in te_table:
                        report_lines.append(
                            "| {lag} | {low} | {mid} | {high} | {on} | {off} |".format(
                                lag=_fmt_num(row.get("lag"), 0),
                                low=_fmt_pct(row.get("vix_low")),
                                mid=_fmt_pct(row.get("vix_mid")),
                                high=_fmt_pct(row.get("vix_high")),
                                on=_fmt_pct(row.get("trend_on")),
                                off=_fmt_pct(row.get("trend_off")),
                            )
                        )
                else:
                    report_lines.append("- Lag by regime tracking error unavailable.")
            else:
                report_lines.append("- Lag by regime tracking error unavailable.")

            report_lines.append("")
            report_lines.append(
                "### 18) Event study expansion (trend flips, drawdown breaches)"
            )
            if positions_entries:
                spy_series, vix_series = _load_regime_market(
                    positions_normalized.index, config
                )
                trend_events = _event_study_trend_flip(
                    dyn_normalized, positions_normalized, spy_series
                )
                if trend_events:
                    report_lines.append("#### Trend flips")
                    report_lines.append(
                        "| Event | Horizon | Synthetic mean | Positions mean | Gap | Events |"
                    )
                    report_lines.append("| --- | --- | --- | --- | --- | --- |")
                    for label, metrics in trend_events.items():
                        for horizon in (5, 10, 20):
                            syn_value = metrics.get(f"syn_{horizon}")
                            pos_value = metrics.get(f"pos_{horizon}")
                            gap_value = metrics.get(f"gap_{horizon}")
                            if syn_value is None or pos_value is None:
                                continue
                            report_lines.append(
                                "| {label} | {h}d | {syn} | {pos} | {gap} | {events} |".format(
                                    label=label,
                                    h=horizon,
                                    syn=_fmt_pct(syn_value, 3),
                                    pos=_fmt_pct(pos_value, 3),
                                    gap=_fmt_pct(gap_value, 3),
                                    events=int(metrics.get("events", 0)),
                                )
                            )
                else:
                    report_lines.append("- Trend flip event study unavailable.")

                report_lines.append("")
                report_lines.append("#### Drawdown breaches")
                report_lines.append(
                    "| Threshold | Horizon | Synthetic mean | Positions mean | Gap | Events |"
                )
                report_lines.append("| --- | --- | --- | --- | --- | --- |")
                for threshold in (-0.05, -0.10):
                    dd_events = _event_study_drawdown_breach(
                        dyn_normalized,
                        positions_normalized,
                        threshold=threshold,
                    )
                    if not dd_events:
                        continue
                    for horizon in (5, 10, 20):
                        syn_value = dd_events.get(f"syn_{horizon}")
                        pos_value = dd_events.get(f"pos_{horizon}")
                        gap_value = dd_events.get(f"gap_{horizon}")
                        if syn_value is None or pos_value is None:
                            continue
                        report_lines.append(
                            "| {thresh} | {h}d | {syn} | {pos} | {gap} | {events} |".format(
                                thresh=_fmt_pct(threshold, 0),
                                h=horizon,
                                syn=_fmt_pct(syn_value, 3),
                                pos=_fmt_pct(pos_value, 3),
                                gap=_fmt_pct(gap_value, 3),
                                events=int(dd_events.get("events", 0)),
                            )
                        )
            else:
                report_lines.append("- Event study expansion unavailable.")

            report_lines.append("")
            report_lines.append("### 19) Variable cost model (VIX + turnover)")
            if positions_entries:
                _, vix_series = _load_regime_market(positions_normalized.index, config)
                cost_model = _apply_variable_cost_model(
                    positions_daily,
                    turnover,
                    vix_series,
                    config,
                    base_bps=2.0,
                    vix_bps_per_point=0.1,
                    turnover_bps_per_pct=0.2,
                )
            else:
                cost_model = {}
            if cost_model:
                report_lines.append(
                    "| Avg cost bps | Base bps | VIX bps/pt | Turnover bps/% | Sharpe | Final Equity | CAGR | Max DD |"
                )
                report_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
                if cost_model.get("invalid"):
                    report_lines.append(
                        "| {avg} | {base} | {vix} | {turn} | N/A | N/A | N/A | N/A |".format(
                            avg=_fmt_num(cost_model.get("avg_cost_bps"), 2),
                            base=_fmt_num(cost_model.get("base_bps"), 2),
                            vix=_fmt_num(cost_model.get("vix_bps_per_point"), 2),
                            turn=_fmt_num(cost_model.get("turnover_bps_per_pct"), 2),
                        )
                    )
                    report_lines.append(
                        "- Variable cost model drove equity non-positive; performance metrics unavailable."
                    )
                else:
                    report_lines.append(
                        "| {avg} | {base} | {vix} | {turn} | {sharpe} | {end} | {cagr} | {dd} |".format(
                            avg=_fmt_num(cost_model.get("avg_cost_bps"), 2),
                            base=_fmt_num(cost_model.get("base_bps"), 2),
                            vix=_fmt_num(cost_model.get("vix_bps_per_point"), 2),
                            turn=_fmt_num(cost_model.get("turnover_bps_per_pct"), 2),
                            sharpe=_fmt_num(cost_model.get("sharpe_ratio")),
                            end=_fmt_num(cost_model.get("normalized_end"), 4),
                            cagr=_fmt_pct(cost_model.get("cagr")),
                            dd=_fmt_pct(cost_model.get("max_drawdown_pct")),
                        )
                    )
            else:
                report_lines.append("- Variable cost model unavailable.")

            report_lines.append("")
            report_lines.append(
                "### 20) VIX quantile regime breakdown (positions-based)"
            )
            if positions_entries:
                vix_quantiles = _regime_breakdown_vix_quantiles(
                    positions_normalized,
                    positions_daily,
                    config,
                )
            else:
                vix_quantiles = {}
            if vix_quantiles:
                report_lines.append(
                    "| VIX bucket | Sharpe | Final Equity | CAGR | Vol | Max DD | Calmar | Points |"
                )
                report_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
                for bucket, metrics in vix_quantiles.items():
                    report_lines.append(
                        "| {bucket} | {sharpe} | {end} | {cagr} | {vol} | {dd} | {calmar} | {points} |".format(
                            bucket=bucket,
                            sharpe=_fmt_num(metrics.get("sharpe_ratio")),
                            end=_fmt_num(metrics.get("normalized_end"), 4),
                            cagr=_fmt_pct(metrics.get("cagr")),
                            vol=_fmt_pct(metrics.get("annualized_volatility")),
                            dd=_fmt_pct(metrics.get("max_drawdown_pct")),
                            calmar=_fmt_num(metrics.get("calmar_ratio")),
                            points=_fmt_num(metrics.get("points"), 0),
                        )
                    )
            else:
                report_lines.append("- VIX quantile breakdown unavailable.")

            report_lines.append("")
            report_lines.append("### 21) Portfolio concentration (positions-based)")
            if positions_entries:
                concentration = _concentration_stats(positions_entries)
            else:
                concentration = {}
            if concentration:
                report_lines.append("| Metric | Value |")
                report_lines.append("| --- | --- |")
                report_lines.append(
                    "| HHI mean | {value} |".format(
                        value=_fmt_num(concentration.get("hhi_mean"), 3),
                    )
                )
                report_lines.append(
                    "| HHI median | {value} |".format(
                        value=_fmt_num(concentration.get("hhi_median"), 3),
                    )
                )
                report_lines.append(
                    "| HHI p95 | {value} |".format(
                        value=_fmt_num(concentration.get("hhi_p95"), 3),
                    )
                )
                report_lines.append(
                    "| Top-1 weight mean | {value} |".format(
                        value=_fmt_pct(concentration.get("top1_mean"), 2),
                    )
                )
                report_lines.append(
                    "| Top-1 weight p95 | {value} |".format(
                        value=_fmt_pct(concentration.get("top1_p95"), 2),
                    )
                )
                report_lines.append(
                    "| Top-2 weight mean | {value} |".format(
                        value=_fmt_pct(concentration.get("top2_mean"), 2),
                    )
                )
                report_lines.append(
                    "| Effective N mean | {value} |".format(
                        value=_fmt_num(concentration.get("effective_n_mean"), 2),
                    )
                )
                report_lines.append(
                    "| Effective N median | {value} |".format(
                        value=_fmt_num(concentration.get("effective_n_median"), 2),
                    )
                )
                report_lines.append(
                    "| Days top-1 > 50% | {value} |".format(
                        value=_fmt_pct(concentration.get("pct_top1_gt_50"), 2),
                    )
                )
            else:
                report_lines.append("- Portfolio concentration unavailable.")

    actual_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_path = Path(f"reports/strategy_next_steps_{actual_time}-notes.md")
    out_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
