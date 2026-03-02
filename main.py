from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import math
import signal
import subprocess
import time

import numpy as np
import pandas as pd
import yfinance as yf

from sg_trader.config import AppConfig, load_config, validate_config
from sg_trader.execution import (
    DryRunBroker,
    ExecutionRequest,
    ExecutionRouter,
    ExternalBrokerStub,
    ManualBroker,
    PaperBroker,
)
from sg_trader.logging_utils import log_transaction
from sg_trader.signals import get_market_signals


@dataclass
class EngineState:
    cash: float
    units: float
    wealth: float
    active_symbol: str


@dataclass
class ForecastSnapshot:
    price: float
    forecast_return: float
    volatility: float
    confidence: float
    target_exposure: float
    data_points: int


@dataclass
class ForecastParams:
    fast_window: int
    slow_window: int
    fast_weight: float
    vol_window: int


@dataclass
class PromotionStats:
    rows: int
    median_ratio: float
    mean_ratio: float
    hit_rate_ge_1: float
    loss_rate_lt_1: float
    min_ratio: float


def _build_parser(config: AppConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="5-minute automated wealth-forecast trading loop.",
    )
    parser.add_argument("--symbol", type=str, default=config.growth_ticker)
    parser.add_argument(
        "--universe-symbols",
        type=str,
        default="",
        help="Comma-separated extra symbols for multi-ticker selection.",
    )
    parser.add_argument(
        "--include-config-tickers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include config core tickers (^SPX, growth, VIX, VVIX) in candidate universe.",
    )
    parser.add_argument(
        "--include-reit-tickers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include config REIT basket symbols in candidate universe.",
    )
    parser.add_argument(
        "--enable-crypto-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow crypto fallback candidates when regular universe is stale.",
    )
    parser.add_argument(
        "--crypto-fallback-symbols",
        type=str,
        default="BTC-USD,ETH-USD",
        help="Comma-separated crypto fallback symbols.",
    )
    parser.add_argument("--interval-minutes", type=int, default=5)
    parser.add_argument(
        "--require-new-bar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only trade/certify when a new interval bar is observed.",
    )
    parser.add_argument(
        "--yahoo-prepost",
        action="store_true",
        help="Include pre/post-market bars when fetching Yahoo intraday data.",
    )
    parser.add_argument("--lookback-period", type=str, default="60d")
    parser.add_argument(
        "--data-fetch-timeout-seconds",
        type=float,
        default=60.0,
        help="Per-attempt Yahoo fetch timeout in seconds.",
    )
    parser.add_argument(
        "--data-fetch-max-attempts",
        type=int,
        default=3,
        help="Maximum Yahoo fetch attempts per symbol before degrading to NO_DATA.",
    )
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument(
        "--replay-smoke-bars",
        type=int,
        default=0,
        help="If >0 with --no-wait, replay recent historical bars sequentially for faster smoke validation.",
    )
    parser.add_argument("--run-forever", action="store_true")
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Run steps back-to-back without waiting for the next interval boundary.",
    )
    parser.add_argument(
        "--flatten-on-no-trade",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Flatten active position when no candidate is eligible (instead of carrying exposure).",
    )
    parser.add_argument("--mode", choices=["paper", "realistic"], default="paper")
    parser.add_argument(
        "--execution-broker",
        type=str,
        default="",
        help="Broker name from execution router (default: paper for paper mode, external for realistic mode).",
    )
    parser.add_argument("--initial-capital", type=float, default=1.0)
    parser.add_argument("--max-exposure", type=float, default=1.0)
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument(
        "--continuity-bonus",
        type=float,
        default=0.05,
        help="Score bonus applied when candidate matches currently active symbol.",
    )
    parser.add_argument(
        "--symbol-regret-window",
        type=int,
        default=12,
        help="Number of recent certified intervals used to estimate per-symbol regret.",
    )
    parser.add_argument(
        "--symbol-regret-penalty",
        type=float,
        default=3.0,
        help="Score penalty multiplier applied to a symbol's recent average regret.",
    )
    parser.add_argument(
        "--index-symbol-penalty",
        type=float,
        default=0.35,
        help="Flat score penalty applied to index-style symbols (prefix '^').",
    )
    parser.add_argument(
        "--index-min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence required for index-style symbols (prefix '^').",
    )
    parser.add_argument(
        "--index-max-exposure",
        type=float,
        default=0.75,
        help="Max absolute exposure allowed for index-style symbols (prefix '^').",
    )
    parser.add_argument("--forecast-threshold", type=float, default=0.0)
    parser.add_argument(
        "--min-projected-edge",
        type=float,
        default=0.0005,
        help="Minimum projected interval edge (projected_ratio-1) required for candidate eligibility.",
    )
    parser.add_argument(
        "--min-projected-edge-buy",
        type=float,
        default=0.0012,
        help="Minimum projected edge required for positive-exposure (buy-side) candidates.",
    )
    parser.add_argument(
        "--symbol-extra-penalty-overrides",
        type=str,
        default="AJBU.SI:0.0,DCRU.SI:0.0",
        help="Comma-separated SYMBOL:value extra score penalties (e.g. AJBU.SI:0.4,DCRU.SI:0.3).",
    )
    parser.add_argument(
        "--symbol-min-confidence-overrides",
        type=str,
        default="AJBU.SI:0.45,DCRU.SI:0.45",
        help="Comma-separated SYMBOL:value min-confidence overrides.",
    )
    parser.add_argument(
        "--symbol-min-projected-edge-buy-overrides",
        type=str,
        default="AJBU.SI:0.0012,DCRU.SI:0.0012",
        help="Comma-separated SYMBOL:value buy-side min projected edge overrides.",
    )
    parser.add_argument("--risk-aversion", type=float, default=1.0)
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.35,
        help="Minimum forecast confidence required for candidate eligibility.",
    )
    parser.add_argument(
        "--vol-floor",
        type=float,
        default=0.0015,
        help="Minimum per-bar log-return volatility used for exposure sizing.",
    )
    parser.add_argument("--paper-slippage-bps", type=float, default=5.0)
    parser.add_argument("--paper-latency-ms", type=int, default=150)
    parser.add_argument("--realistic-slippage-bps", type=float, default=10.0)
    parser.add_argument("--realistic-latency-ms", type=int, default=300)
    parser.add_argument("--commission-bps", type=float, default=0.0)
    parser.add_argument("--order-seed", type=int, default=None)
    parser.add_argument("--cache-dir", type=str, default=config.cache_dir)
    parser.add_argument("--output-csv", type=str, default="reports/wealth_forecast_5m.csv")
    parser.add_argument(
        "--append-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append to existing output CSV and resume state from its last row (disabled by default).",
    )
    parser.add_argument(
        "--pid-file",
        type=str,
        default="reports/wealth_forecast_runner.pid",
        help="PID file used for continuous run lifecycle commands.",
    )
    parser.add_argument(
        "--stop-runner",
        action="store_true",
        help="Stop a running continuous process using the PID file.",
    )
    parser.add_argument(
        "--stop-and-audit",
        action="store_true",
        help="Stop the running continuous process and generate proof audit from CSV snapshot.",
    )
    parser.add_argument("--log-transactions", action="store_true")
    parser.add_argument("--respect-vol-kill", action="store_true")
    parser.add_argument(
        "--proof-tolerance",
        type=float,
        default=1e-9,
        help="Tolerance used when declaring interval optimality vs oracle benchmark.",
    )
    parser.add_argument(
        "--freshness-max-minutes",
        type=int,
        default=30,
        help="Max allowed age (minutes) for latest bar before feed is treated as stale.",
    )
    parser.add_argument(
        "--fallback-symbols",
        type=str,
        default="",
        help="Comma-separated fallback symbols when primary symbol has stale data (e.g. BTC-USD,ETH-USD).",
    )
    parser.add_argument(
        "--max-stale-steps",
        type=int,
        default=6,
        help="Stop run after this many consecutive stale-bar checks (0 disables).",
    )
    parser.add_argument(
        "--walk-forward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable rolling walk-forward tuning before each interval.",
    )
    parser.add_argument("--wf-train-bars", type=int, default=240)
    parser.add_argument("--wf-min-bars", type=int, default=80)
    parser.add_argument("--wf-retrain-every", type=int, default=1)
    parser.add_argument(
        "--wf-fast-grid",
        type=str,
        default="3,5,8",
        help="Comma-separated fast windows for walk-forward tuning.",
    )
    parser.add_argument(
        "--wf-slow-grid",
        type=str,
        default="12,18,24",
        help="Comma-separated slow windows for walk-forward tuning.",
    )
    parser.add_argument(
        "--wf-fast-weight-grid",
        type=str,
        default="0.5,0.65,0.8",
        help="Comma-separated weights for fast component.",
    )
    parser.add_argument(
        "--wf-vol-window-grid",
        type=str,
        default="18,24,36",
        help="Comma-separated volatility windows.",
    )
    parser.add_argument(
        "--promote-profile",
        action="store_true",
        help="Scan historical run CSVs and write best walk-forward profile JSON.",
    )
    parser.add_argument(
        "--promotion-glob",
        type=str,
        default="reports/wealth_forecast_5m*.csv",
        help="Glob for historical wealth forecast CSVs used for profile promotion.",
    )
    parser.add_argument(
        "--promotion-output-json",
        type=str,
        default="reports/wealth_forecast_profile.json",
        help="Output JSON path for promoted profile.",
    )
    parser.add_argument(
        "--promotion-min-rows",
        type=int,
        default=10,
        help="Minimum scored rows for a parameter set to be promotion-eligible.",
    )
    parser.add_argument(
        "--proof-report",
        action="store_true",
        help="Generate a markdown proof audit from a wealth forecast CSV.",
    )
    parser.add_argument(
        "--proof-input-csv",
        type=str,
        default="reports/wealth_forecast_5m_proof_smoke.csv",
        help="Input CSV for proof report generation.",
    )
    parser.add_argument(
        "--proof-output-md",
        type=str,
        default="reports/wealth_forecast_proof_audit.md",
        help="Output markdown path for proof report.",
    )
    parser.add_argument(
        "--proof-worst-n",
        type=int,
        default=10,
        help="Number of worst-regret intervals to include in the proof report.",
    )
    parser.add_argument(
        "--diagnostics-top-n",
        type=int,
        default=0,
        help="If >0, capture and report top-N candidate scores each step.",
    )
    parser.add_argument(
        "--diagnostics-print",
        action="store_true",
        help="Print top-N candidate diagnostics each step when diagnostics-top-n > 0.",
    )
    parser.add_argument(
        "--diagnostics-csv",
        type=str,
        default="",
        help="Optional CSV path to append candidate diagnostics rows.",
    )
    parser.add_argument(
        "--diagnostics-include-ineligible",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include ineligible candidates in top-N diagnostics when eligible list is sparse.",
    )
    return parser


def _resolve_broker_name(mode: str, broker: str) -> str:
    if broker.strip():
        return broker.strip()
    return "paper" if mode == "paper" else "external"


def _init_router() -> ExecutionRouter:
    router = ExecutionRouter()
    router.register("paper", PaperBroker())
    router.register("dry-run", DryRunBroker())
    router.register("manual", ManualBroker())
    router.register("external", ExternalBrokerStub())
    return router


def _fetch_full_close_series(
    symbol: str,
    lookback_period: str,
    interval_minutes: int,
    prepost: bool,
    timeout_seconds: float,
    max_attempts: int,
) -> tuple[pd.Series, pd.Timestamp | None]:
    interval = f"{interval_minutes}m"
    attempts = max(1, int(max_attempts))
    timeout = max(1.0, float(timeout_seconds))
    history: pd.DataFrame | None = None
    for attempt in range(1, attempts + 1):
        try:
            history = yf.Ticker(symbol).history(
                period=lookback_period,
                interval=interval,
                prepost=prepost,
                timeout=timeout,
            )
            break
        except Exception:
            history = None
            if attempt >= attempts:
                return pd.Series(dtype=float), None
            time.sleep(min(2.0, 0.5 * attempt))
    if history is None or not isinstance(history, pd.DataFrame):
        return pd.Series(dtype=float), None
    if "Close" not in history.columns:
        return pd.Series(dtype=float), None
    close = history["Close"].dropna().astype(float)
    if close.empty:
        return close, None
    try:
        bar_ts = pd.to_datetime(close.index[-1])
    except Exception:
        bar_ts = None
    return close, bar_ts


def _parse_symbols(raw: str) -> list[str]:
    if not raw.strip():
        return []
    out: list[str] = []
    for token in raw.split(","):
        symbol = token.strip()
        if symbol:
            out.append(symbol)
    return out


def _unique_symbols(symbols: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for symbol in symbols:
        sym = symbol.strip()
        if not sym:
            continue
        upper = sym.upper()
        if upper in seen:
            continue
        seen.add(upper)
        out.append(sym)
    return out


def _build_candidate_universe(args: argparse.Namespace, config: AppConfig) -> list[str]:
    symbols: list[str] = [str(args.symbol)]
    symbols.extend(_parse_symbols(str(args.universe_symbols)))
    symbols.extend(_parse_symbols(str(args.fallback_symbols)))
    if bool(args.include_config_tickers):
        symbols.extend(
            [
                config.ticker,
                config.growth_ticker,
                config.vix_ticker,
                config.vvix_ticker,
            ]
        )
    if bool(args.include_reit_tickers):
        symbols.extend(list(config.reit_tickers.keys()))
    return _unique_symbols(symbols)


def _fetch_symbol_market_state(
    symbol: str,
    lookback_period: str,
    interval_minutes: int,
    prepost: bool,
    freshness_max_minutes: int,
    last_seen_by_symbol: dict[str, pd.Timestamp],
    fetch_timeout_seconds: float,
    fetch_max_attempts: int,
    replay_step: int,
    replay_smoke_bars: int,
) -> dict[str, object]:
    close, bar_ts = _fetch_full_close_series(
        symbol,
        lookback_period,
        interval_minutes,
        prepost,
        fetch_timeout_seconds,
        fetch_max_attempts,
    )
    if replay_smoke_bars > 0 and not close.empty:
        total = int(len(close))
        replay_window = max(2, min(int(replay_smoke_bars), total))
        replay_start = max(1, total - replay_window)
        replay_index = min(replay_start + max(0, int(replay_step)), total - 1)
        close = close.iloc[: replay_index + 1]
        try:
            bar_ts = pd.to_datetime(close.index[-1])
        except Exception:
            bar_ts = None

    age = _bar_age_minutes(bar_ts)
    is_replay_mode = replay_smoke_bars > 0
    is_fresh = bool(not close.empty) if is_replay_mode else bool(age is not None and age <= float(freshness_max_minutes))
    prev_seen = last_seen_by_symbol.get(symbol)
    is_new_bar = bool(
        bar_ts is not None
        and (prev_seen is None or pd.Timestamp(bar_ts) > pd.Timestamp(prev_seen))
    )
    return {
        "symbol": symbol,
        "close": close,
        "bar_ts": bar_ts,
        "bar_age_minutes": age,
        "is_fresh": is_fresh,
        "is_new_bar": is_new_bar,
    }


def _select_best_candidate(
    candidate_states: list[dict[str, object]],
    args: argparse.Namespace,
    active_params: ForecastParams,
    active_symbol: str,
    recent_symbol_regret: dict[str, float],
) -> tuple[dict[str, object] | None, ForecastSnapshot | None]:
    best_state: dict[str, object] | None = None
    best_snapshot: ForecastSnapshot | None = None
    best_score = float("-inf")

    for state in candidate_states:
        symbol = str(state.get("symbol", ""))
        symbol_key = symbol.upper()
        close = state["close"]
        if not isinstance(close, pd.Series):
            continue
        if close.empty:
            continue
        if not bool(state.get("is_fresh", False)):
            continue
        if bool(args.require_new_bar) and not bool(state.get("is_new_bar", False)):
            continue

        snapshot = _compute_forecast(
            close,
            threshold=float(args.forecast_threshold),
            risk_aversion=float(args.risk_aversion),
            max_exposure=float(args.max_exposure),
            long_only=bool(args.long_only),
            params=active_params,
            vol_floor=float(args.vol_floor),
        )
        if not np.isfinite(snapshot.price):
            continue
        is_index_symbol = symbol.startswith("^")
        min_conf_required = (
            float(args.index_min_confidence)
            if is_index_symbol
            else float(args.min_confidence)
        )
        symbol_conf_overrides = getattr(args, "_symbol_min_confidence_overrides", {})
        if isinstance(symbol_conf_overrides, dict):
            min_conf_required = float(symbol_conf_overrides.get(symbol_key, min_conf_required))
        if snapshot.confidence < min_conf_required:
            continue
        if is_index_symbol:
            index_max_exposure = max(0.0, float(args.index_max_exposure))
            snapshot.target_exposure = float(
                np.clip(snapshot.target_exposure, -index_max_exposure, index_max_exposure)
            )
        projected_ratio = 1.0 + snapshot.target_exposure * snapshot.forecast_return
        projected_edge = projected_ratio - 1.0
        min_edge_required = (
            float(args.min_projected_edge_buy)
            if snapshot.target_exposure > 0.0
            else float(args.min_projected_edge)
        )
        symbol_buy_edge_overrides = getattr(args, "_symbol_min_edge_buy_overrides", {})
        if snapshot.target_exposure > 0.0 and isinstance(symbol_buy_edge_overrides, dict):
            min_edge_required = float(symbol_buy_edge_overrides.get(symbol_key, min_edge_required))
        if projected_edge < min_edge_required:
            continue
        score = projected_ratio * (1.0 + max(0.0, snapshot.confidence))
        score -= float(args.symbol_regret_penalty) * float(recent_symbol_regret.get(symbol, 0.0))
        if symbol.startswith("^"):
            score -= float(args.index_symbol_penalty)
        symbol_extra_penalty = getattr(args, "_symbol_extra_penalty_overrides", {})
        if isinstance(symbol_extra_penalty, dict):
            score -= float(symbol_extra_penalty.get(symbol_key, 0.0))
        if active_symbol and symbol == active_symbol:
            score += float(args.continuity_bonus)
        if score > best_score:
            best_score = score
            best_state = state
            best_snapshot = snapshot

    return best_state, best_snapshot


def _score_candidates(
    candidate_states: list[dict[str, object]],
    args: argparse.Namespace,
    active_params: ForecastParams,
    recent_symbol_regret: dict[str, float],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for state in candidate_states:
        symbol = str(state.get("symbol", ""))
        symbol_key = symbol.upper()
        close = state.get("close")
        is_fresh = bool(state.get("is_fresh", False))
        is_new_bar = bool(state.get("is_new_bar", False))
        bar_age_minutes = state.get("bar_age_minutes")
        if not isinstance(close, pd.Series) or close.empty:
            rows.append(
                {
                    "symbol": symbol,
                    "eligible": 0,
                    "reason": "no_data",
                    "score": float("nan"),
                    "price": float("nan"),
                    "forecast_return": float("nan"),
                    "target_exposure": float("nan"),
                    "confidence": float("nan"),
                    "is_fresh": 1 if is_fresh else 0,
                    "is_new_bar": 1 if is_new_bar else 0,
                    "bar_age_minutes": _safe_float(bar_age_minutes),
                }
            )
            continue

        snapshot = _compute_forecast(
            close,
            threshold=float(args.forecast_threshold),
            risk_aversion=float(args.risk_aversion),
            max_exposure=float(args.max_exposure),
            long_only=bool(args.long_only),
            params=active_params,
            vol_floor=float(args.vol_floor),
        )
        reason = "ok"
        eligible = 1
        if not is_fresh:
            reason = "stale_feed"
            eligible = 0
        elif bool(args.require_new_bar) and not is_new_bar:
            reason = "stale_bar"
            eligible = 0
        elif not np.isfinite(snapshot.price):
            reason = "nan_price"
            eligible = 0
        is_index_symbol = symbol.startswith("^")
        min_conf_required = (
            float(args.index_min_confidence)
            if is_index_symbol
            else float(args.min_confidence)
        )
        symbol_conf_overrides = getattr(args, "_symbol_min_confidence_overrides", {})
        if isinstance(symbol_conf_overrides, dict):
            min_conf_required = float(symbol_conf_overrides.get(symbol_key, min_conf_required))
        if is_index_symbol:
            index_max_exposure = max(0.0, float(args.index_max_exposure))
            snapshot.target_exposure = float(
                np.clip(snapshot.target_exposure, -index_max_exposure, index_max_exposure)
            )
        if eligible and snapshot.confidence < min_conf_required:
            reason = "low_confidence"
            eligible = 0

        score = float("nan")
        projected_ratio = 1.0 + snapshot.target_exposure * snapshot.forecast_return
        projected_edge = projected_ratio - 1.0
        if eligible:
            min_edge_required = (
                float(args.min_projected_edge_buy)
                if snapshot.target_exposure > 0.0
                else float(args.min_projected_edge)
            )
            symbol_buy_edge_overrides = getattr(args, "_symbol_min_edge_buy_overrides", {})
            if snapshot.target_exposure > 0.0 and isinstance(symbol_buy_edge_overrides, dict):
                min_edge_required = float(symbol_buy_edge_overrides.get(symbol_key, min_edge_required))
            if projected_edge < min_edge_required:
                reason = "low_edge"
                eligible = 0
            else:
                score = projected_ratio * (1.0 + max(0.0, snapshot.confidence))
                score -= float(args.symbol_regret_penalty) * float(recent_symbol_regret.get(symbol, 0.0))
                if is_index_symbol:
                    score -= float(args.index_symbol_penalty)
                symbol_extra_penalty = getattr(args, "_symbol_extra_penalty_overrides", {})
                if isinstance(symbol_extra_penalty, dict):
                    score -= float(symbol_extra_penalty.get(symbol_key, 0.0))

        rows.append(
            {
                "symbol": symbol,
                "eligible": eligible,
                "reason": reason,
                "score": float(score),
                "price": snapshot.price,
                "forecast_return": snapshot.forecast_return,
                "target_exposure": snapshot.target_exposure,
                "confidence": snapshot.confidence,
                "projected_edge": projected_edge,
                "recent_symbol_regret": float(recent_symbol_regret.get(symbol, 0.0)),
                "index_symbol_penalty": float(args.index_symbol_penalty) if symbol.startswith("^") else 0.0,
                "is_fresh": 1 if is_fresh else 0,
                "is_new_bar": 1 if is_new_bar else 0,
                "bar_age_minutes": _safe_float(bar_age_minutes),
            }
        )
    return rows


def _recent_symbol_regret(rows: list[dict[str, float | int | str]], window: int) -> dict[str, float]:
    lookback = max(0, int(window))
    if lookback <= 0:
        return {}
    recent = rows[-lookback:]
    by_symbol: dict[str, list[float]] = {}
    for row in recent:
        symbol = str(row.get("execution_symbol", "") or row.get("data_symbol", "")).strip()
        regret = _safe_float(row.get("interval_regret"))
        if not symbol or not np.isfinite(regret):
            continue
        by_symbol.setdefault(symbol, []).append(float(regret))
    return {symbol: float(np.mean(values)) for symbol, values in by_symbol.items() if values}


def _top_n_candidates(
    scored_rows: list[dict[str, float | int | str]],
    top_n: int,
    include_ineligible: bool = False,
) -> list[dict[str, float | int | str]]:
    if include_ineligible:
        ranked = list(scored_rows)
        ranked.sort(
            key=lambda row: (
                int(row.get("eligible", 0)),
                float(row.get("score", float("-inf"))),
            ),
            reverse=True,
        )
        return ranked[: max(0, int(top_n))]

    eligible = [row for row in scored_rows if int(row.get("eligible", 0)) == 1]
    eligible.sort(
        key=lambda row: float(row.get("score", float("-inf"))),
        reverse=True,
    )
    return eligible[: max(0, int(top_n))]


def _safe_float(value: object) -> float:
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return float("nan")
    return float("nan")


def _append_diagnostics_rows(
    path: Path,
    rows: list[dict[str, float | int | str]],
) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _bar_age_minutes(bar_ts: pd.Timestamp | None) -> float | None:
    if bar_ts is None:
        return None
    try:
        ts = pd.Timestamp(bar_ts)
    except Exception:
        return None
    now_ts = pd.Timestamp.now(tz=ts.tz) if ts.tz is not None else pd.Timestamp.now()
    age = (now_ts - ts).total_seconds() / 60.0
    if not np.isfinite(age):
        return None
    return float(age)


def _resolve_market_data(
    primary_symbol: str,
    fallback_symbols: list[str],
    lookback_period: str,
    interval_minutes: int,
    prepost: bool,
    freshness_max_minutes: int,
    fetch_timeout_seconds: float,
    fetch_max_attempts: int,
) -> tuple[str, pd.Series, pd.Timestamp | None, bool, float | None]:
    candidates = [primary_symbol] + [sym for sym in fallback_symbols if sym != primary_symbol]
    best_symbol = primary_symbol
    best_close = pd.Series(dtype=float)
    best_ts: pd.Timestamp | None = None
    best_age: float | None = None

    for symbol in candidates:
        close, ts = _fetch_full_close_series(
            symbol,
            lookback_period,
            interval_minutes,
            prepost,
            fetch_timeout_seconds,
            fetch_max_attempts,
        )
        age = _bar_age_minutes(ts)
        if close.empty or ts is None:
            continue
        if best_ts is None or pd.Timestamp(ts) > pd.Timestamp(best_ts):
            best_symbol = symbol
            best_close = close
            best_ts = ts
            best_age = age
        if age is not None and age <= float(freshness_max_minutes):
            return symbol, close, ts, True, age

    if best_ts is None:
        return primary_symbol, pd.Series(dtype=float), None, False, None
    return best_symbol, best_close, best_ts, False, best_age


def _parse_int_grid(raw: str, default: list[int]) -> list[int]:
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            continue
        if value > 0:
            values.append(value)
    if not values:
        return default
    return sorted(set(values))


def _parse_float_grid(raw: str, default: list[float]) -> list[float]:
    values: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = float(token)
        except ValueError:
            continue
        if 0.0 <= value <= 1.0:
            values.append(value)
    if not values:
        return default
    return sorted(set(values))


def _parse_symbol_float_overrides(raw: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if not raw.strip():
        return out
    for token in raw.split(","):
        item = token.strip()
        if not item or ":" not in item:
            continue
        symbol_raw, value_raw = item.split(":", 1)
        symbol = symbol_raw.strip().upper()
        if not symbol:
            continue
        try:
            value = float(value_raw.strip())
        except ValueError:
            continue
        if np.isfinite(value):
            out[symbol] = float(value)
    return out


def _compute_forecast_from_params(
    close: pd.Series,
    params: ForecastParams,
    threshold: float,
    risk_aversion: float,
    max_exposure: float,
    long_only: bool,
    vol_floor: float,
) -> ForecastSnapshot:
    if close.empty:
        return ForecastSnapshot(
            price=float("nan"),
            forecast_return=0.0,
            volatility=float("nan"),
            confidence=0.0,
            target_exposure=0.0,
            data_points=0,
        )

    price = float(close.iloc[-1])
    returns = (close / close.shift(1)).apply(np.log).dropna()
    if returns.empty:
        return ForecastSnapshot(
            price=price,
            forecast_return=0.0,
            volatility=float("nan"),
            confidence=0.0,
            target_exposure=0.0,
            data_points=int(len(close)),
        )

    fast = float(returns.tail(params.fast_window).mean())
    slow = float(returns.tail(params.slow_window).mean())
    forecast = params.fast_weight * fast + (1.0 - params.fast_weight) * slow
    vol = float(returns.tail(params.vol_window).std(ddof=0))
    effective_vol_floor = max(1e-9, float(vol_floor))
    if not np.isfinite(vol) or vol <= effective_vol_floor:
        vol = effective_vol_floor
    confidence = abs(forecast) / vol

    if abs(forecast) <= threshold:
        target_exposure = 0.0
    else:
        signal = forecast / (vol * max(risk_aversion, 1e-9))
        target_exposure = float(np.clip(signal, -max_exposure, max_exposure))

    if long_only:
        target_exposure = max(0.0, target_exposure)

    return ForecastSnapshot(
        price=price,
        forecast_return=forecast,
        volatility=vol,
        confidence=confidence,
        target_exposure=target_exposure,
        data_points=int(len(close)),
    )


def _compute_forecast(
    close: pd.Series,
    threshold: float,
    risk_aversion: float,
    max_exposure: float,
    long_only: bool,
    params: ForecastParams,
    vol_floor: float,
) -> ForecastSnapshot:
    return _compute_forecast_from_params(
        close=close,
        params=params,
        threshold=threshold,
        risk_aversion=risk_aversion,
        max_exposure=max_exposure,
        long_only=long_only,
        vol_floor=vol_floor,
    )


def _walk_forward_score(
    returns: pd.Series,
    params: ForecastParams,
    threshold: float,
    risk_aversion: float,
    max_exposure: float,
    long_only: bool,
    vol_floor: float,
) -> float:
    min_required = max(params.fast_window, params.slow_window, params.vol_window) + 3
    if len(returns) <= min_required:
        return float("-inf")

    strat = []
    start = max(params.fast_window, params.slow_window, params.vol_window)
    for idx in range(start, len(returns) - 1):
        hist = returns.iloc[: idx + 1]
        fast = float(hist.tail(params.fast_window).mean())
        slow = float(hist.tail(params.slow_window).mean())
        forecast = params.fast_weight * fast + (1.0 - params.fast_weight) * slow
        vol = float(hist.tail(params.vol_window).std(ddof=0))
        effective_vol_floor = max(1e-9, float(vol_floor))
        if not np.isfinite(vol) or vol <= effective_vol_floor:
            vol = effective_vol_floor

        if abs(forecast) <= threshold:
            exposure = 0.0
        else:
            signal = forecast / (vol * max(risk_aversion, 1e-9))
            exposure = float(np.clip(signal, -max_exposure, max_exposure))
        if long_only:
            exposure = max(0.0, exposure)

        realized = float(returns.iloc[idx + 1])
        strat.append(exposure * realized)

    if not strat:
        return float("-inf")

    pnl = np.asarray(strat, dtype=float)
    pnl_std = float(np.std(pnl, ddof=0))
    if pnl_std <= 1e-12 or not np.isfinite(pnl_std):
        return float(np.mean(pnl))
    return float(np.mean(pnl) / pnl_std)


def _tune_walk_forward_params(
    close: pd.Series,
    threshold: float,
    risk_aversion: float,
    max_exposure: float,
    long_only: bool,
    vol_floor: float,
    train_bars: int,
    min_bars: int,
    fast_grid: list[int],
    slow_grid: list[int],
    fast_weight_grid: list[float],
    vol_window_grid: list[int],
) -> ForecastParams:
    returns = (close / close.shift(1)).apply(np.log).dropna()
    if len(returns) < max(min_bars, 20):
        return ForecastParams(fast_window=3, slow_window=12, fast_weight=0.65, vol_window=24)

    train = returns.tail(max(train_bars, min_bars))
    best_params = ForecastParams(fast_window=3, slow_window=12, fast_weight=0.65, vol_window=24)
    best_score = float("-inf")

    for fast_window in fast_grid:
        for slow_window in slow_grid:
            if slow_window <= fast_window:
                continue
            for fast_weight in fast_weight_grid:
                for vol_window in vol_window_grid:
                    params = ForecastParams(
                        fast_window=fast_window,
                        slow_window=slow_window,
                        fast_weight=fast_weight,
                        vol_window=vol_window,
                    )
                    score = _walk_forward_score(
                        returns=train,
                        params=params,
                        threshold=threshold,
                        risk_aversion=risk_aversion,
                        max_exposure=max_exposure,
                        long_only=long_only,
                        vol_floor=vol_floor,
                    )
                    if score > best_score:
                        best_score = score
                        best_params = params

    return best_params


def _seconds_until_next_interval(interval_minutes: int) -> float:
    interval_seconds = max(interval_minutes, 1) * 60
    now = time.time()
    next_boundary = (math.floor(now / interval_seconds) + 1) * interval_seconds
    return max(0.0, next_boundary - now)


def _rebalance_to_target(
    state: EngineState,
    symbol: str,
    broker_name: str,
    router: ExecutionRouter,
    price: float,
    target_exposure: float,
    max_position_qty: float,
    max_notional: float,
    slippage_bps: float,
    latency_ms: int,
    commission_bps: float,
    seed: int | None,
    allow_short: bool,
) -> tuple[EngineState, str, float, float]:
    wealth = state.cash + state.units * price
    target_notional = float(np.clip(target_exposure * wealth, -max_notional, max_notional))
    target_units = target_notional / price if price > 0 else 0.0
    target_units = float(np.clip(target_units, -max_position_qty, max_position_qty))
    order_qty = target_units - state.units

    if abs(order_qty) < 1e-10:
        updated = EngineState(
            cash=state.cash,
            units=state.units,
            wealth=wealth,
            active_symbol=state.active_symbol,
        )
        return updated, "HOLD", 0.0, price

    side = "BUY" if order_qty > 0 else "SELL"
    quantity = abs(order_qty)

    request = ExecutionRequest(
        symbol=symbol,
        side=side,
        quantity=quantity,
        reference_price=price,
        slippage_bps=slippage_bps,
        latency_ms=latency_ms,
        seed=seed,
    )
    result = router.execute(broker_name, request)
    fill_price = float(result.fill_price)
    commission_rate = commission_bps / 10000.0

    if side == "BUY":
        max_affordable = state.cash / (fill_price * (1.0 + commission_rate)) if fill_price > 0 else 0.0
        quantity = min(quantity, max_affordable)
        cash_change = -(quantity * fill_price * (1.0 + commission_rate))
        units_change = quantity
    else:
        if not allow_short:
            quantity = min(quantity, abs(state.units))
        cash_change = quantity * fill_price * (1.0 - commission_rate)
        units_change = -quantity

    new_cash = state.cash + cash_change
    new_units = state.units + units_change
    new_wealth = new_cash + new_units * price

    updated = EngineState(
        cash=float(new_cash),
        units=float(new_units),
        wealth=float(new_wealth),
        active_symbol=symbol if abs(new_units) > 1e-12 else "",
    )
    return updated, side, float(quantity), fill_price


def _write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def _load_existing_rows(path: Path) -> list[dict[str, float | int | str]]:
    if not path.exists():
        return []
    try:
        frame = pd.read_csv(path)
    except Exception:
        return []
    if frame.empty:
        return []
    raw_rows = frame.to_dict(orient="records")
    rows: list[dict[str, float | int | str]] = []
    for raw in raw_rows:
        normalized: dict[str, float | int | str] = {}
        for key, value in raw.items():
            normalized[str(key)] = value
        rows.append(normalized)
    return rows


def _resume_state_from_rows(
    rows: list[dict[str, float | int | str]],
    initial_capital: float,
) -> tuple[EngineState, int]:
    if not rows:
        return (
            EngineState(
                cash=float(initial_capital),
                units=0.0,
                wealth=float(initial_capital),
                active_symbol="",
            ),
            0,
        )
    last = rows[-1]
    try:
        cash = float(last.get("cash", initial_capital))
    except (TypeError, ValueError):
        cash = float(initial_capital)
    try:
        units = float(last.get("units", 0.0))
    except (TypeError, ValueError):
        units = 0.0
    try:
        wealth = float(last.get("wealth", cash))
    except (TypeError, ValueError):
        wealth = cash
    active_symbol = str(last.get("execution_symbol", "") or "")
    try:
        max_step = int(max(float(r.get("step", -1)) for r in rows))
    except Exception:
        max_step = len(rows) - 1
    state = EngineState(
        cash=cash,
        units=units,
        wealth=wealth,
        active_symbol=active_symbol if abs(units) > 1e-12 else "",
    )
    return state, max_step + 1


def _write_pid_file(pid_file: Path) -> None:
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()), encoding="utf-8")


def _read_pid_file(pid_file: Path) -> int | None:
    if not pid_file.exists():
        return None
    try:
        return int(pid_file.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _clear_pid_file(pid_file: Path) -> None:
    if pid_file.exists():
        try:
            pid_file.unlink()
        except Exception:
            pass


def _stop_runner(pid_file: Path) -> bool:
    pid = _read_pid_file(pid_file)
    if pid is None:
        print(f"No valid PID found at {pid_file}.")
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print(f"No running process found for PID {pid}; clearing stale pid file.")
        _clear_pid_file(pid_file)
        return False
    except PermissionError:
        print(f"Permission denied stopping PID {pid}.")
        return False
    print(f"Sent SIGTERM to runner PID {pid}.")
    return True


def _is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _list_continuous_runner_pids() -> list[int]:
    script_name = Path(__file__).name
    try:
        output = subprocess.check_output(
            ["ps", "-ax", "-o", "pid=,command="],
            text=True,
        )
    except Exception:
        return []

    pids: list[int] = []
    current_pid = os.getpid()
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        pid_str, command = parts
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        if pid == current_pid:
            continue
        if script_name not in command:
            continue
        if "--run-forever" not in command:
            continue
        pids.append(pid)
    return sorted(set(pids))


def _stop_all_runners(pid_file: Path) -> tuple[list[int], list[int]]:
    pids = _list_continuous_runner_pids()
    pid_from_file = _read_pid_file(pid_file)
    if pid_from_file is not None:
        pids = sorted(set(pids + [pid_from_file]))

    if not pids:
        print("No running continuous runner processes found.")
        _clear_pid_file(pid_file)
        return [], []

    terminated: list[int] = []
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            terminated.append(pid)
        except ProcessLookupError:
            continue
        except PermissionError:
            print(f"Permission denied stopping PID {pid}.")

    deadline = time.time() + 2.0
    while time.time() < deadline:
        alive = [pid for pid in terminated if _is_process_alive(pid)]
        if not alive:
            break
        time.sleep(0.1)

    still_alive = [pid for pid in terminated if _is_process_alive(pid)]
    forced: list[int] = []
    for pid in still_alive:
        try:
            os.kill(pid, signal.SIGKILL)
            forced.append(pid)
        except ProcessLookupError:
            continue
        except PermissionError:
            print(f"Permission denied force-killing PID {pid}.")

    _clear_pid_file(pid_file)
    if terminated:
        print(f"Terminated runner PIDs: {', '.join(str(pid) for pid in terminated)}")
    if forced:
        print(f"Force-killed runner PIDs: {', '.join(str(pid) for pid in forced)}")
    return terminated, forced


def _oracle_exposure(simple_return: float, max_exposure: float, long_only: bool) -> float:
    if long_only:
        return float(max_exposure) if simple_return > 0 else 0.0
    if simple_return > 0:
        return float(max_exposure)
    if simple_return < 0:
        return float(-max_exposure)
    return 0.0


def _finalize_interval_certificate(
    rows: list[dict[str, float | int | str]],
    next_price: float,
    max_exposure: float,
    long_only: bool,
    tolerance: float,
) -> None:
    if not rows:
        return
    prev_row = rows[-1]
    prev_price = float(prev_row.get("price", float("nan")))
    chosen_exposure = float(prev_row.get("realized_exposure", 0.0))
    if not np.isfinite(prev_price) or prev_price <= 0 or not np.isfinite(next_price):
        prev_row["interval_next_price"] = float("nan")
        prev_row["interval_simple_return"] = float("nan")
        prev_row["interval_model_ratio"] = float("nan")
        prev_row["interval_oracle_exposure"] = float("nan")
        prev_row["interval_oracle_ratio"] = float("nan")
        prev_row["interval_regret"] = float("nan")
        prev_row["interval_optimal_flag"] = float("nan")
        prev_row["wealth_ratio"] = float("nan")
        prev_row["wealth_gain_flag"] = float("nan")
        return

    simple_return = (next_price / prev_price) - 1.0
    model_ratio = 1.0 + chosen_exposure * simple_return
    oracle_exposure = _oracle_exposure(simple_return, max_exposure, long_only)
    oracle_ratio = 1.0 + oracle_exposure * simple_return
    regret = oracle_ratio - model_ratio
    optimal_flag = 1 if model_ratio >= (oracle_ratio - tolerance) else 0

    prev_row["interval_next_price"] = float(next_price)
    prev_row["interval_simple_return"] = float(simple_return)
    prev_row["interval_model_ratio"] = float(model_ratio)
    prev_row["interval_oracle_exposure"] = float(oracle_exposure)
    prev_row["interval_oracle_ratio"] = float(oracle_ratio)
    prev_row["interval_regret"] = float(regret)
    prev_row["interval_optimal_flag"] = int(optimal_flag)
    prev_row["wealth_ratio"] = float(model_ratio)
    prev_row["wealth_gain_flag"] = 1 if model_ratio >= 1.0 else 0


def _finalize_interval_certificate_for_index(
    rows: list[dict[str, float | int | str]],
    row_index: int,
    next_price: float,
    max_exposure: float,
    long_only: bool,
    tolerance: float,
) -> None:
    if row_index < 0 or row_index >= len(rows):
        return
    row = rows[row_index]
    prev_price = _safe_float(row.get("price"))
    chosen_exposure = _safe_float(row.get("realized_exposure"))
    if not np.isfinite(prev_price) or prev_price <= 0 or not np.isfinite(next_price):
        row["interval_next_price"] = float("nan")
        row["interval_simple_return"] = float("nan")
        row["interval_model_ratio"] = float("nan")
        row["interval_oracle_exposure"] = float("nan")
        row["interval_oracle_ratio"] = float("nan")
        row["interval_regret"] = float("nan")
        row["interval_optimal_flag"] = float("nan")
        row["wealth_ratio"] = float("nan")
        row["wealth_gain_flag"] = float("nan")
        return

    simple_return = (next_price / prev_price) - 1.0
    model_ratio = 1.0 + chosen_exposure * simple_return
    oracle_exposure = _oracle_exposure(simple_return, max_exposure, long_only)
    oracle_ratio = 1.0 + oracle_exposure * simple_return
    regret = oracle_ratio - model_ratio
    optimal_flag = 1 if model_ratio >= (oracle_ratio - tolerance) else 0

    row["interval_next_price"] = float(next_price)
    row["interval_simple_return"] = float(simple_return)
    row["interval_model_ratio"] = float(model_ratio)
    row["interval_oracle_exposure"] = float(oracle_exposure)
    row["interval_oracle_ratio"] = float(oracle_ratio)
    row["interval_regret"] = float(regret)
    row["interval_optimal_flag"] = int(optimal_flag)
    row["wealth_ratio"] = float(model_ratio)
    row["wealth_gain_flag"] = 1 if model_ratio >= 1.0 else 0


def _collect_promotion_stats(df: pd.DataFrame) -> dict[tuple[int, int, float, int], PromotionStats]:
    if df.empty:
        return {}
    needed_cols = {
        "wealth",
        "wf_fast_window",
        "wf_slow_window",
        "wf_fast_weight",
        "wf_vol_window",
    }
    if not needed_cols.issubset(df.columns):
        return {}

    scored = df.copy()
    if "interval_model_ratio" in scored.columns:
        scored["wealth_ratio"] = pd.to_numeric(scored["interval_model_ratio"], errors="coerce")
    elif "wealth_ratio" in scored.columns:
        scored["wealth_ratio"] = pd.to_numeric(scored["wealth_ratio"], errors="coerce")
    else:
        scored["wealth"] = pd.to_numeric(scored["wealth"], errors="coerce")
        scored["wealth_ratio"] = scored["wealth"] / scored["wealth"].shift(1)
    scored = scored.dropna(subset=["wealth_ratio"])
    if scored.empty:
        return {}

    group_cols = ["wf_fast_window", "wf_slow_window", "wf_fast_weight", "wf_vol_window"]
    stats: dict[tuple[int, int, float, int], PromotionStats] = {}

    grouped = scored.groupby(group_cols, dropna=True)
    for _, chunk in grouped:
        ratios = pd.to_numeric(chunk["wealth_ratio"], errors="coerce").dropna()
        if ratios.empty:
            continue
        first = chunk.iloc[0]
        key = (
            int(pd.to_numeric(first["wf_fast_window"], errors="coerce")),
            int(pd.to_numeric(first["wf_slow_window"], errors="coerce")),
            float(pd.to_numeric(first["wf_fast_weight"], errors="coerce")),
            int(pd.to_numeric(first["wf_vol_window"], errors="coerce")),
        )
        ratio_values = ratios.to_numpy(dtype=float)
        hits = float(np.mean(ratio_values >= 1.0))
        losses = float(np.mean(ratio_values < 1.0))
        stats[key] = PromotionStats(
            rows=int(len(ratio_values)),
            median_ratio=float(np.median(ratio_values)),
            mean_ratio=float(np.mean(ratio_values)),
            hit_rate_ge_1=hits,
            loss_rate_lt_1=losses,
            min_ratio=float(np.min(ratio_values)),
        )
    return stats


def _merge_promotion_stats(
    aggregate: dict[tuple[int, int, float, int], PromotionStats],
    incoming: dict[tuple[int, int, float, int], PromotionStats],
) -> dict[tuple[int, int, float, int], PromotionStats]:
    for key, inc in incoming.items():
        cur = aggregate.get(key)
        if cur is None:
            aggregate[key] = inc
            continue

        total_rows = cur.rows + inc.rows
        mean_ratio = ((cur.mean_ratio * cur.rows) + (inc.mean_ratio * inc.rows)) / total_rows
        hit_rate = ((cur.hit_rate_ge_1 * cur.rows) + (inc.hit_rate_ge_1 * inc.rows)) / total_rows
        loss_rate = ((cur.loss_rate_lt_1 * cur.rows) + (inc.loss_rate_lt_1 * inc.rows)) / total_rows
        median_ratio = ((cur.median_ratio * cur.rows) + (inc.median_ratio * inc.rows)) / total_rows
        min_ratio = min(cur.min_ratio, inc.min_ratio)
        aggregate[key] = PromotionStats(
            rows=total_rows,
            median_ratio=float(median_ratio),
            mean_ratio=float(mean_ratio),
            hit_rate_ge_1=float(hit_rate),
            loss_rate_lt_1=float(loss_rate),
            min_ratio=float(min_ratio),
        )
    return aggregate


def _promote_profile(
    promotion_glob: str,
    output_json: Path,
    min_rows: int,
) -> int:
    csv_paths = sorted(Path(".").glob(promotion_glob))
    if not csv_paths:
        print(f"No promotion candidates found for glob: {promotion_glob}")
        return 2

    aggregate: dict[tuple[int, int, float, int], PromotionStats] = {}
    files_used = 0
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        file_stats = _collect_promotion_stats(df)
        if not file_stats:
            continue
        aggregate = _merge_promotion_stats(aggregate, file_stats)
        files_used += 1

    if not aggregate:
        print("No eligible walk-forward stats found in candidate files.")
        return 2

    eligible = [(k, v) for k, v in aggregate.items() if v.rows >= min_rows]
    if not eligible:
        print(f"No parameter sets reached minimum rows ({min_rows}).")
        return 2

    eligible.sort(
        key=lambda item: (
            item[1].median_ratio,
            item[1].mean_ratio,
            item[1].hit_rate_ge_1,
            -item[1].loss_rate_lt_1,
            item[1].rows,
        ),
        reverse=True,
    )
    best_key, best_stats = eligible[0]
    promoted = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_glob": promotion_glob,
        "files_scanned": len(csv_paths),
        "files_used": files_used,
        "min_rows_required": min_rows,
        "promoted_params": {
            "fast_window": best_key[0],
            "slow_window": best_key[1],
            "fast_weight": best_key[2],
            "vol_window": best_key[3],
        },
        "stats": {
            "rows": best_stats.rows,
            "median_ratio": best_stats.median_ratio,
            "mean_ratio": best_stats.mean_ratio,
            "hit_rate_ge_1": best_stats.hit_rate_ge_1,
            "loss_rate_lt_1": best_stats.loss_rate_lt_1,
            "min_ratio": best_stats.min_ratio,
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(promoted, indent=2), encoding="utf-8")
    print(
        "Promoted profile: "
        f"fast={best_key[0]}, slow={best_key[1]}, weight={best_key[2]:.2f}, vol={best_key[3]} "
        f"| median_ratio={best_stats.median_ratio:.6f}, hit_rate={best_stats.hit_rate_ge_1:.2%}, rows={best_stats.rows}"
    )
    print(f"Profile saved: {output_json}")
    return 0


def _generate_proof_report(
    input_csv: Path,
    output_md: Path,
    worst_n: int,
) -> int:
    if not input_csv.exists():
        print(f"Proof input CSV not found: {input_csv}")
        return 2

    try:
        frame = pd.read_csv(input_csv)
    except Exception as exc:
        print(f"Failed to read proof input CSV: {exc}")
        return 2

    required = {
        "timestamp",
        "step",
        "interval_model_ratio",
        "interval_oracle_ratio",
        "interval_optimal_flag",
        "interval_regret",
        "action",
        "symbol",
    }
    if not required.issubset(frame.columns):
        missing = sorted(required - set(frame.columns))
        print(f"Missing required proof columns: {', '.join(missing)}")
        return 2

    scored = frame.copy()
    scored["interval_model_ratio"] = pd.to_numeric(
        scored["interval_model_ratio"], errors="coerce"
    )
    scored["interval_oracle_ratio"] = pd.to_numeric(
        scored["interval_oracle_ratio"], errors="coerce"
    )
    scored["interval_optimal_flag"] = pd.to_numeric(
        scored["interval_optimal_flag"], errors="coerce"
    )
    scored["interval_regret"] = pd.to_numeric(scored["interval_regret"], errors="coerce")

    scored = scored.dropna(
        subset=[
            "interval_model_ratio",
            "interval_oracle_ratio",
            "interval_optimal_flag",
            "interval_regret",
        ]
    )
    if scored.empty:
        print("No certifiable intervals found in proof CSV.")
        return 2

    model_ratio = scored["interval_model_ratio"].to_numpy(dtype=float)
    oracle_ratio = scored["interval_oracle_ratio"].to_numpy(dtype=float)
    optimal_flag = scored["interval_optimal_flag"].to_numpy(dtype=float)
    regret = scored["interval_regret"].to_numpy(dtype=float)

    optimal_rate = float(np.mean(optimal_flag >= 1.0))
    hit_rate = float(np.mean(model_ratio >= 1.0))
    loss_rate = float(np.mean(model_ratio < 1.0))
    avg_regret = float(np.mean(regret))
    median_regret = float(np.median(regret))
    max_regret = float(np.max(regret))
    model_median = float(np.median(model_ratio))
    oracle_median = float(np.median(oracle_ratio))
    cert_count = int(len(scored))

    worst_count = max(1, int(worst_n))
    worst = scored.sort_values("interval_regret", ascending=False).head(worst_count)

    lines: list[str] = []
    lines.append("# Wealth Forecast Proof Audit")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Source CSV: {input_csv}")
    lines.append(f"- Certifiable intervals: {cert_count}")
    lines.append("")
    lines.append("## Summary Metrics")
    lines.append("")
    lines.append(f"- Optimality rate (model vs oracle): {optimal_rate:.2%}")
    lines.append(f"- Model hit rate (ratio >= 1): {hit_rate:.2%}")
    lines.append(f"- Model loss rate (ratio < 1): {loss_rate:.2%}")
    lines.append(f"- Median model ratio: {model_median:.6f}")
    lines.append(f"- Median oracle ratio: {oracle_median:.6f}")
    lines.append(f"- Avg regret: {avg_regret:.6f}")
    lines.append(f"- Median regret: {median_regret:.6f}")
    lines.append(f"- Max regret: {max_regret:.6f}")
    lines.append("")
    lines.append("## Worst-Regret Intervals")
    lines.append("")
    lines.append("| Timestamp | Step | Symbol | Action | Model Ratio | Oracle Ratio | Regret | Optimal |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for _, row in worst.iterrows():
        step_value = int(float(row.get("step", 0)))
        model_value = float(row.get("interval_model_ratio", float("nan")))
        oracle_value = float(row.get("interval_oracle_ratio", float("nan")))
        regret_value = float(row.get("interval_regret", float("nan")))
        optimal_value = int(float(row.get("interval_optimal_flag", 0)))
        lines.append(
            "| {timestamp} | {step} | {symbol} | {action} | {model:.6f} | {oracle:.6f} | {regret:.6f} | {optimal} |".format(
                timestamp=str(row.get("timestamp", "")),
                step=step_value,
                symbol=str(row.get("symbol", "")),
                action=str(row.get("action", "")),
                model=model_value,
                oracle=oracle_value,
                regret=regret_value,
                optimal=optimal_value,
            )
        )

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Proof report written: {output_md}")
    print(
        "Summary: "
        f"optimality={optimal_rate:.2%}, hit={hit_rate:.2%}, avg_regret={avg_regret:.6f}, intervals={cert_count}"
    )
    return 0


def run(argv: list[str] | None = None) -> int:
    config = load_config()
    errors = validate_config(config)
    if errors:
        print("Invalid configuration:")
        for err in errors:
            print(f"- {err}")
        return 2

    parser = _build_parser(config)
    args = parser.parse_args(argv)

    if args.promote_profile:
        return _promote_profile(
            promotion_glob=str(args.promotion_glob),
            output_json=Path(args.promotion_output_json),
            min_rows=int(args.promotion_min_rows),
        )

    pid_file = Path(args.pid_file)
    if args.stop_runner:
        _stop_all_runners(pid_file)
        return 0

    if args.stop_and_audit:
        _stop_all_runners(pid_file)
        return _generate_proof_report(
            input_csv=Path(args.proof_input_csv),
            output_md=Path(args.proof_output_md),
            worst_n=int(args.proof_worst_n),
        )

    if args.proof_report:
        return _generate_proof_report(
            input_csv=Path(args.proof_input_csv),
            output_md=Path(args.proof_output_md),
            worst_n=int(args.proof_worst_n),
        )

    if args.interval_minutes != 5:
        print("Warning: strategy is tuned for 5-minute cadence; non-5 value selected.")

    if int(args.replay_smoke_bars) > 0 and not bool(args.no_wait):
        print("replay-smoke-bars requires --no-wait")
        return 2
    if int(args.replay_smoke_bars) > 0 and bool(args.run_forever):
        print("replay-smoke-bars cannot be used with --run-forever")
        return 2

    if args.initial_capital <= 0:
        print("initial-capital must be positive")
        return 2

    broker_name = _resolve_broker_name(args.mode, args.execution_broker)
    router = _init_router()
    if broker_name not in set(router.list_brokers()):
        print(f"Unknown execution broker: {broker_name}")
        print(f"Available brokers: {', '.join(router.list_brokers())}")
        return 2

    slippage_bps = args.paper_slippage_bps if args.mode == "paper" else args.realistic_slippage_bps
    latency_ms = args.paper_latency_ms if args.mode == "paper" else args.realistic_latency_ms
    wf_fast_grid = _parse_int_grid(args.wf_fast_grid, [3, 5, 8])
    wf_slow_grid = _parse_int_grid(args.wf_slow_grid, [12, 18, 24])
    wf_fast_weight_grid = _parse_float_grid(args.wf_fast_weight_grid, [0.5, 0.65, 0.8])
    wf_vol_window_grid = _parse_int_grid(args.wf_vol_window_grid, [18, 24, 36])
    args._symbol_extra_penalty_overrides = _parse_symbol_float_overrides(
        str(args.symbol_extra_penalty_overrides)
    )
    args._symbol_min_confidence_overrides = _parse_symbol_float_overrides(
        str(args.symbol_min_confidence_overrides)
    )
    args._symbol_min_edge_buy_overrides = _parse_symbol_float_overrides(
        str(args.symbol_min_projected_edge_buy_overrides)
    )
    active_params = ForecastParams(fast_window=3, slow_window=12, fast_weight=0.65, vol_window=24)
    crypto_fallback_symbols = _parse_symbols(str(args.crypto_fallback_symbols))
    candidate_universe = _build_candidate_universe(args, config)
    diagnostics_top_n = max(0, int(args.diagnostics_top_n))
    diagnostics_include_ineligible = bool(args.diagnostics_include_ineligible)
    diagnostics_path = Path(str(args.diagnostics_csv)) if str(args.diagnostics_csv).strip() else None
    print(
        "Universe: "
        f"{', '.join(candidate_universe)} | "
        f"crypto_fallback_enabled={1 if bool(args.enable_crypto_fallback) else 0} | "
        f"crypto_fallback_symbols={','.join(crypto_fallback_symbols) if crypto_fallback_symbols else 'none'}"
    )

    output_path = Path(args.output_csv)
    rows: list[dict[str, float | int | str]] = (
        _load_existing_rows(output_path) if bool(args.append_output) else []
    )
    state, step = _resume_state_from_rows(rows, float(args.initial_capital))
    if rows:
        print(f"Resumed from {output_path} with {len(rows)} existing rows.")
    last_seen_by_symbol: dict[str, pd.Timestamp] = {}
    for existing in rows:
        symbol = str(existing.get("execution_symbol", "") or existing.get("data_symbol", ""))
        bar_stamp = existing.get("bar_timestamp")
        if not symbol or bar_stamp is None or str(bar_stamp).strip() == "":
            continue
        try:
            parsed = pd.Timestamp(str(bar_stamp))
        except Exception:
            continue
        prev = last_seen_by_symbol.get(symbol)
        if prev is None or parsed > prev:
            last_seen_by_symbol[symbol] = parsed
    pending_row_index_by_symbol: dict[str, int] = {}
    for idx, existing in enumerate(rows):
        symbol = str(existing.get("execution_symbol", "") or existing.get("data_symbol", "")).strip()
        if not symbol:
            continue
        ratio_val = _safe_float(existing.get("interval_model_ratio"))
        price_val = _safe_float(existing.get("price"))
        if np.isfinite(price_val) and not np.isfinite(ratio_val):
            pending_row_index_by_symbol[symbol] = idx
    stale_steps = 0
    end_step = None if args.run_forever else step + int(args.steps)
    if args.run_forever:
        _write_pid_file(pid_file)
        print(f"Runner PID written to {pid_file}")
    while True:
        if end_step is not None and step >= end_step:
            break

        recent_symbol_regret = _recent_symbol_regret(rows, int(args.symbol_regret_window))
        candidate_states: list[dict[str, object]] = []
        for symbol in candidate_universe:
            candidate_states.append(
                _fetch_symbol_market_state(
                    symbol=symbol,
                    lookback_period=str(args.lookback_period),
                    interval_minutes=int(args.interval_minutes),
                    prepost=bool(args.yahoo_prepost),
                    freshness_max_minutes=int(args.freshness_max_minutes),
                    last_seen_by_symbol=last_seen_by_symbol,
                    fetch_timeout_seconds=float(args.data_fetch_timeout_seconds),
                    fetch_max_attempts=int(args.data_fetch_max_attempts),
                    replay_step=int(step),
                    replay_smoke_bars=int(args.replay_smoke_bars),
                )
            )

            scored_candidates = _score_candidates(
                candidate_states,
                args,
                active_params,
                recent_symbol_regret,
            )
            top_candidates = _top_n_candidates(
                scored_candidates,
                diagnostics_top_n,
                include_ineligible=diagnostics_include_ineligible,
            )

        if args.walk_forward and (
            step == 0 or step % max(1, int(args.wf_retrain_every)) == 0
        ):
            train_close = pd.Series(dtype=float)
            for state_item in candidate_states:
                close_candidate = state_item["close"]
                if isinstance(close_candidate, pd.Series) and not close_candidate.empty:
                    train_close = close_candidate
                    break
            active_params = _tune_walk_forward_params(
                close=train_close,
                threshold=float(args.forecast_threshold),
                risk_aversion=float(args.risk_aversion),
                max_exposure=float(args.max_exposure),
                long_only=bool(args.long_only),
                vol_floor=float(args.vol_floor),
                train_bars=int(args.wf_train_bars),
                min_bars=int(args.wf_min_bars),
                fast_grid=wf_fast_grid,
                slow_grid=wf_slow_grid,
                fast_weight_grid=wf_fast_weight_grid,
                vol_window_grid=wf_vol_window_grid,
            )
            scored_candidates = _score_candidates(
                candidate_states,
                args,
                active_params,
                recent_symbol_regret,
            )
            top_candidates = _top_n_candidates(
                scored_candidates,
                diagnostics_top_n,
                include_ineligible=diagnostics_include_ineligible,
            )

        best_state, snapshot = _select_best_candidate(
            candidate_states,
            args,
            active_params,
            state.active_symbol,
            recent_symbol_regret,
        )
        selected_from_crypto_fallback = False
        if best_state is None and bool(args.enable_crypto_fallback):
            for symbol in crypto_fallback_symbols:
                candidate_states.append(
                    _fetch_symbol_market_state(
                        symbol=symbol,
                        lookback_period=str(args.lookback_period),
                        interval_minutes=int(args.interval_minutes),
                        prepost=bool(args.yahoo_prepost),
                        freshness_max_minutes=int(args.freshness_max_minutes),
                        last_seen_by_symbol=last_seen_by_symbol,
                        fetch_timeout_seconds=float(args.data_fetch_timeout_seconds),
                        fetch_max_attempts=int(args.data_fetch_max_attempts),
                        replay_step=int(step),
                        replay_smoke_bars=int(args.replay_smoke_bars),
                    )
                )
            best_state, snapshot = _select_best_candidate(
                candidate_states,
                args,
                active_params,
                state.active_symbol,
                recent_symbol_regret,
            )
            selected_from_crypto_fallback = best_state is not None
            scored_candidates = _score_candidates(
                candidate_states,
                args,
                active_params,
                recent_symbol_regret,
            )
            top_candidates = _top_n_candidates(
                scored_candidates,
                diagnostics_top_n,
                include_ineligible=diagnostics_include_ineligible,
            )

        no_trade_due_filters = False
        if snapshot is None or best_state is None:
            available_states = []
            for state_item in candidate_states:
                close_candidate = state_item.get("close")
                if isinstance(close_candidate, pd.Series) and not close_candidate.empty:
                    available_states.append(state_item)

            reference_state: dict[str, object] | None = None
            if available_states:
                reference_state = max(
                    available_states,
                    key=lambda row: int(bool(row.get("is_fresh", False))) * 2
                    + int(bool(row.get("is_new_bar", False))),
                )

            has_market_price = bool(reference_state is not None)
            has_fresh = any(bool(row.get("is_fresh", False)) for row in available_states)
            has_fresh_new = any(
                bool(row.get("is_fresh", False)) and bool(row.get("is_new_bar", False))
                for row in available_states
            )

            if reference_state is not None:
                data_symbol = str(reference_state.get("symbol", args.symbol))
                bar_ts_raw = reference_state.get("bar_ts")
                bar_ts = (
                    bar_ts_raw
                    if isinstance(bar_ts_raw, pd.Timestamp) or bar_ts_raw is None
                    else None
                )
                is_fresh = bool(reference_state.get("is_fresh", False))
                is_new_bar = bool(reference_state.get("is_new_bar", False))
                bar_age_minutes = reference_state.get("bar_age_minutes")
                close_ref = reference_state.get("close")
                ref_price = (
                    float(close_ref.iloc[-1])
                    if isinstance(close_ref, pd.Series) and not close_ref.empty
                    else float("nan")
                )
                data_points = int(len(close_ref)) if isinstance(close_ref, pd.Series) else 0
            else:
                data_symbol = str(args.symbol)
                bar_ts = None
                is_fresh = False
                is_new_bar = False
                bar_age_minutes = None
                ref_price = float("nan")
                data_points = 0

            if has_market_price and has_fresh and (
                has_fresh_new or not bool(args.require_new_bar)
            ):
                no_trade_due_filters = True

            snapshot = ForecastSnapshot(
                price=ref_price,
                forecast_return=0.0,
                volatility=float("nan"),
                confidence=0.0,
                target_exposure=0.0,
                data_points=data_points,
            )
        else:
            data_symbol = str(best_state["symbol"])
            bar_ts = best_state["bar_ts"] if isinstance(best_state.get("bar_ts"), pd.Timestamp) or best_state.get("bar_ts") is None else None
            is_fresh = bool(best_state.get("is_fresh", False))
            is_new_bar = bool(best_state.get("is_new_bar", False))
            bar_age_minutes = best_state.get("bar_age_minutes")

        if rows and pending_row_index_by_symbol:
            for state_item in candidate_states:
                symbol = str(state_item.get("symbol", "")).strip()
                if not symbol:
                    continue
                pending_idx = pending_row_index_by_symbol.get(symbol)
                if pending_idx is None:
                    continue
                is_new_for_symbol = bool(state_item.get("is_new_bar", False))
                close_series = state_item.get("close")
                next_price = (
                    float(close_series.iloc[-1])
                    if isinstance(close_series, pd.Series) and not close_series.empty
                    else float("nan")
                )
                if is_new_for_symbol and np.isfinite(next_price):
                    _finalize_interval_certificate_for_index(
                        rows=rows,
                        row_index=int(pending_idx),
                        next_price=next_price,
                        max_exposure=float(args.max_exposure),
                        long_only=bool(args.long_only),
                        tolerance=float(args.proof_tolerance),
                    )
                    pending_row_index_by_symbol.pop(symbol, None)

        target_exposure = snapshot.target_exposure
        if args.respect_vol_kill:
            market = get_market_signals(config)
            if market is not None and market.iv >= config.paper_vol_kill_threshold:
                target_exposure = 0.0

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        forced_execution_symbol = ""

        mark_price = snapshot.price
        if state.active_symbol and state.active_symbol != data_symbol:
            active_price_state = next(
                (s for s in candidate_states if str(s.get("symbol", "")).strip() == state.active_symbol),
                None,
            )
            active_close_series = (
                active_price_state.get("close") if isinstance(active_price_state, dict) else None
            )
            active_mark_price = (
                float(active_close_series.iloc[-1])
                if isinstance(active_close_series, pd.Series) and not active_close_series.empty
                else float("nan")
            )
            if np.isfinite(active_mark_price):
                mark_price = active_mark_price

        if no_trade_due_filters:
            if bool(args.flatten_on_no_trade) and state.active_symbol and abs(state.units) > 1e-12:
                forced_execution_symbol = state.active_symbol
                flatten_price_state = next(
                    (s for s in candidate_states if str(s.get("symbol", "")).strip() == state.active_symbol),
                    None,
                )
                flatten_close_series = (
                    flatten_price_state.get("close") if isinstance(flatten_price_state, dict) else None
                )
                flatten_price = (
                    float(flatten_close_series.iloc[-1])
                    if isinstance(flatten_close_series, pd.Series) and not flatten_close_series.empty
                    else float("nan")
                )
                if np.isfinite(flatten_price) and flatten_price > 0:
                    state, flatten_side, executed_qty, fill_price = _rebalance_to_target(
                        state=state,
                        symbol=state.active_symbol,
                        broker_name=broker_name,
                        router=router,
                        price=flatten_price,
                        target_exposure=0.0,
                        max_position_qty=float(config.paper_max_position_qty),
                        max_notional=float(config.paper_max_notional),
                        slippage_bps=float(slippage_bps),
                        latency_ms=int(latency_ms),
                        commission_bps=float(args.commission_bps),
                        seed=args.order_seed,
                        allow_short=not bool(args.long_only),
                    )
                    side = f"FLATTEN_{flatten_side}" if flatten_side in {"BUY", "SELL"} else "NO_TRADE"
                else:
                    side = "NO_TRADE"
                    executed_qty = 0.0
                    fill_price = float("nan")
                    state = EngineState(
                        cash=state.cash,
                        units=state.units,
                        wealth=state.cash + state.units * mark_price,
                        active_symbol=state.active_symbol,
                    )
            else:
                side = "NO_TRADE"
                executed_qty = 0.0
                fill_price = float("nan")
                state = EngineState(
                    cash=state.cash,
                    units=state.units,
                    wealth=state.cash + state.units * mark_price,
                    active_symbol=state.active_symbol,
                )
            stale_steps = 0
        elif snapshot.data_points == 0 or not np.isfinite(snapshot.price):
            state = EngineState(
                cash=state.cash,
                units=state.units,
                wealth=state.wealth,
                active_symbol=state.active_symbol,
            )
            side = "NO_DATA"
            executed_qty = 0.0
            fill_price = float("nan")
            stale_steps += 1
        elif not is_fresh:
            side = "DATA_STALE"
            executed_qty = 0.0
            fill_price = float("nan")
            state = EngineState(
                cash=state.cash,
                units=state.units,
                wealth=state.cash + state.units * mark_price,
                active_symbol=state.active_symbol,
            )
            stale_steps += 1
        elif args.require_new_bar and not is_new_bar:
            side = "STALE_BAR"
            executed_qty = 0.0
            fill_price = float("nan")
            state = EngineState(
                cash=state.cash,
                units=state.units,
                wealth=state.cash + state.units * mark_price,
                active_symbol=state.active_symbol,
            )
            stale_steps += 1
        else:
            try:
                if state.active_symbol and state.active_symbol != data_symbol and abs(state.units) > 1e-12:
                    active_price_state = next((s for s in candidate_states if str(s.get("symbol")) == state.active_symbol), None)
                    active_close = active_price_state.get("close") if isinstance(active_price_state, dict) else None
                    if isinstance(active_close, pd.Series) and not active_close.empty:
                        active_price = float(active_close.iloc[-1])
                        close_side = "SELL" if state.units > 0 else "BUY"
                        close_qty = abs(state.units)
                        close_request = ExecutionRequest(
                            symbol=state.active_symbol,
                            side=close_side,
                            quantity=close_qty,
                            reference_price=active_price,
                            slippage_bps=float(slippage_bps),
                            latency_ms=int(latency_ms),
                            seed=args.order_seed,
                        )
                        close_result = router.execute(broker_name, close_request)
                        commission_rate = float(args.commission_bps) / 10000.0
                        if close_side == "SELL":
                            state.cash += close_qty * float(close_result.fill_price) * (1.0 - commission_rate)
                        else:
                            state.cash -= close_qty * float(close_result.fill_price) * (1.0 + commission_rate)
                        state.units = 0.0
                        state.active_symbol = ""

                state, side, executed_qty, fill_price = _rebalance_to_target(
                    state=state,
                    symbol=data_symbol,
                    broker_name=broker_name,
                    router=router,
                    price=snapshot.price,
                    target_exposure=target_exposure,
                    max_position_qty=float(config.paper_max_position_qty),
                    max_notional=float(config.paper_max_notional),
                    slippage_bps=float(slippage_bps),
                    latency_ms=int(latency_ms),
                    commission_bps=float(args.commission_bps),
                    seed=args.order_seed,
                    allow_short=not bool(args.long_only),
                )
                state.active_symbol = data_symbol if abs(state.units) > 1e-12 else ""
                stale_steps = 0
            except RuntimeError as exc:
                print(f"Execution error at step {step}: {exc}")
                side = "EXEC_ERROR"
                executed_qty = 0.0
                fill_price = float("nan")
                state = EngineState(
                    cash=state.cash,
                    units=state.units,
                    wealth=state.cash + state.units * snapshot.price,
                    active_symbol=state.active_symbol,
                )
                stale_steps += 1

        row_execution_symbol = forced_execution_symbol if forced_execution_symbol else data_symbol
        row_price = snapshot.price
        if forced_execution_symbol and np.isfinite(mark_price):
            row_price = mark_price
        if side in {"NO_TRADE", "DATA_STALE", "STALE_BAR", "FLATTEN_BUY", "FLATTEN_SELL"} and state.active_symbol:
            row_execution_symbol = state.active_symbol
            active_price_state = next(
                (s for s in candidate_states if str(s.get("symbol", "")).strip() == state.active_symbol),
                None,
            )
            active_close_series = (
                active_price_state.get("close") if isinstance(active_price_state, dict) else None
            )
            active_row_price = (
                float(active_close_series.iloc[-1])
                if isinstance(active_close_series, pd.Series) and not active_close_series.empty
                else float("nan")
            )
            if np.isfinite(active_row_price):
                row_price = active_row_price

        realized_exposure = 0.0
        if state.active_symbol and state.wealth > 0:
            exposure_price_state = next(
                (s for s in candidate_states if str(s.get("symbol", "")).strip() == state.active_symbol),
                None,
            )
            exposure_close_series = (
                exposure_price_state.get("close") if isinstance(exposure_price_state, dict) else None
            )
            exposure_price = (
                float(exposure_close_series.iloc[-1])
                if isinstance(exposure_close_series, pd.Series) and not exposure_close_series.empty
                else float("nan")
            )
            if np.isfinite(exposure_price) and exposure_price > 0:
                realized_exposure = (state.units * exposure_price) / state.wealth

        row = {
            "timestamp": ts,
            "step": step,
            "mode": args.mode,
            "broker": broker_name,
            "universe_size": len(candidate_universe),
            "universe_symbols": ",".join(candidate_universe),
            "crypto_fallback_enabled": 1 if bool(args.enable_crypto_fallback) else 0,
            "crypto_fallback_symbols": ",".join(crypto_fallback_symbols) if crypto_fallback_symbols else "",
            "symbol": args.symbol,
            "execution_symbol": row_execution_symbol,
            "data_symbol": data_symbol,
            "selected_from_crypto_fallback": 1 if selected_from_crypto_fallback else 0,
            "price": row_price,
            "forecast_return": snapshot.forecast_return,
            "volatility": snapshot.volatility,
            "confidence": snapshot.confidence,
            "target_exposure": target_exposure,
            "realized_exposure": realized_exposure,
            "action": side,
            "executed_qty": executed_qty,
            "fill_price": fill_price,
            "cash": state.cash,
            "units": state.units,
            "wealth": state.wealth,
            "wealth_ratio": float("nan"),
            "wealth_gain_flag": float("nan"),
            "interval_next_price": float("nan"),
            "interval_simple_return": float("nan"),
            "interval_model_ratio": float("nan"),
            "interval_oracle_exposure": float("nan"),
            "interval_oracle_ratio": float("nan"),
            "interval_regret": float("nan"),
            "interval_optimal_flag": float("nan"),
            "bar_timestamp": str(bar_ts) if bar_ts is not None else "",
            "bar_age_minutes": bar_age_minutes if bar_age_minutes is not None else float("nan"),
            "is_fresh_bar": 1 if is_fresh else 0,
            "is_new_bar": 1 if is_new_bar else 0,
            "consecutive_stale_steps": stale_steps,
            "diagnostics_top_symbol": str(top_candidates[0]["symbol"]) if top_candidates else "",
            "diagnostics_top_score": float(top_candidates[0]["score"]) if top_candidates else float("nan"),
            "diagnostics_top_n": diagnostics_top_n,
            "data_points": snapshot.data_points,
            "wf_fast_window": active_params.fast_window,
            "wf_slow_window": active_params.slow_window,
            "wf_fast_weight": active_params.fast_weight,
            "wf_vol_window": active_params.vol_window,
        }
        rows.append(row)
        row_symbol_state = next(
            (s for s in candidate_states if str(s.get("symbol", "")).strip() == row_execution_symbol),
            None,
        )
        row_symbol_is_new_bar = bool(row_symbol_state.get("is_new_bar", False)) if isinstance(row_symbol_state, dict) else False
        if row_symbol_is_new_bar and np.isfinite(row_price):
            pending_row_index_by_symbol[row_execution_symbol] = len(rows) - 1
        _write_rows(output_path, rows)

        if diagnostics_top_n > 0 and diagnostics_path is not None:
            diag_rows: list[dict[str, float | int | str]] = []
            for rank, cand in enumerate(top_candidates, start=1):
                diag_rows.append(
                    {
                        "timestamp": ts,
                        "step": step,
                        "selected_symbol": data_symbol,
                        "rank": rank,
                        "symbol": str(cand.get("symbol", "")),
                        "score": float(cand.get("score", float("nan"))),
                        "forecast_return": float(cand.get("forecast_return", float("nan"))),
                        "target_exposure": float(cand.get("target_exposure", float("nan"))),
                        "confidence": float(cand.get("confidence", float("nan"))),
                        "reason": str(cand.get("reason", "")),
                        "eligible": int(cand.get("eligible", 0)),
                        "is_fresh": int(cand.get("is_fresh", 0)),
                        "is_new_bar": int(cand.get("is_new_bar", 0)),
                        "bar_age_minutes": float(cand.get("bar_age_minutes", float("nan"))),
                    }
                )
            _append_diagnostics_rows(diagnostics_path, diag_rows)

        if diagnostics_top_n > 0 and bool(args.diagnostics_print):
            if top_candidates:
                summary = ", ".join(
                    f"{cand['symbol']}:{float(cand['score']):.6f}" for cand in top_candidates
                )
                print(f"top_candidates[{diagnostics_top_n}] {summary}")
            else:
                print(f"top_candidates[{diagnostics_top_n}] none")
        if is_new_bar and bar_ts is not None:
            if isinstance(bar_ts, pd.Timestamp):
                last_seen_by_symbol[data_symbol] = bar_ts
            else:
                parsed = pd.Timestamp(str(bar_ts))
                last_seen_by_symbol[data_symbol] = parsed

        print(
            " | ".join(
                [
                    f"t={ts}",
                    f"step={step}",
                    f"price={snapshot.price:.4f}" if np.isfinite(snapshot.price) else "price=nan",
                    f"fwd={snapshot.forecast_return:.6f}",
                    f"target={target_exposure:.3f}",
                    f"wf=({active_params.fast_window},{active_params.slow_window},{active_params.fast_weight:.2f},{active_params.vol_window})",
                    f"data_symbol={data_symbol}",
                    f"new_bar={1 if is_new_bar else 0}",
                    f"fresh={1 if is_fresh else 0}",
                    f"action={side}",
                    f"qty={executed_qty:.6f}",
                    f"wealth={state.wealth:.6f}",
                    (
                        f"proof_prev(oracle/model)={float(rows[-2]['interval_oracle_ratio']):.6f}/{float(rows[-2]['interval_model_ratio']):.6f}"
                        if len(rows) >= 2 and np.isfinite(float(rows[-2].get("interval_model_ratio", float("nan"))))
                        and np.isfinite(float(rows[-2].get("interval_oracle_ratio", float("nan"))) )
                        else "proof_prev=nan"
                    ),
                    (
                        f"proof_prev_regret={float(rows[-2]['interval_regret']):.6f}"
                        if len(rows) >= 2 and np.isfinite(float(rows[-2].get("interval_regret", float("nan"))))
                        else "proof_prev_regret=nan"
                    ),
                ]
            )
        )

        if args.log_transactions:
            details = {
                "mode": args.mode,
                "broker": broker_name,
                "price": snapshot.price,
                "forecast_return": snapshot.forecast_return,
                "target_exposure": target_exposure,
                "realized_exposure": realized_exposure,
                "executed_qty": executed_qty,
                "fill_price": fill_price,
                "wealth": state.wealth,
                "wealth_ratio": row["wealth_ratio"],
                "wealth_gain_flag": row["wealth_gain_flag"],
                "interval_model_ratio": row["interval_model_ratio"],
                "interval_oracle_ratio": row["interval_oracle_ratio"],
                "interval_optimal_flag": row["interval_optimal_flag"],
                "interval_regret": row["interval_regret"],
                "wf_fast_window": active_params.fast_window,
                "wf_slow_window": active_params.slow_window,
                "wf_fast_weight": active_params.fast_weight,
                "wf_vol_window": active_params.vol_window,
            }
            log_transaction(
                category="ForecastEngine",
                ticker=data_symbol,
                action=side,
                rationale="5-minute wealth-forecast rebalance",
                config=config,
                tags=["5m", args.mode, "wealth_forecast"],
                details=details,
            )

        step += 1
        if int(args.max_stale_steps) > 0 and stale_steps >= int(args.max_stale_steps):
            print(
                "Stopping early due to stale feed: "
                f"{stale_steps} consecutive stale steps (max={int(args.max_stale_steps)})."
            )
            break
        if (args.run_forever or step < args.steps) and not args.no_wait:
            sleep_seconds = _seconds_until_next_interval(args.interval_minutes)
            time.sleep(sleep_seconds)

    if args.run_forever:
        current_pid = os.getpid()
        file_pid = _read_pid_file(pid_file)
        if file_pid == current_pid:
            _clear_pid_file(pid_file)

    if rows:
        final = rows[-1]
        frame = pd.DataFrame(rows)
        ratio_series = pd.to_numeric(frame["interval_model_ratio"], errors="coerce").dropna()
        oracle_series = pd.to_numeric(frame["interval_oracle_ratio"], errors="coerce").dropna()
        optimal_series = pd.to_numeric(frame["interval_optimal_flag"], errors="coerce").dropna()
        regret_series = pd.to_numeric(frame["interval_regret"], errors="coerce").dropna()
        hit_rate = float((ratio_series >= 1.0).mean()) if not ratio_series.empty else float("nan")
        loss_rate = float((ratio_series < 1.0).mean()) if not ratio_series.empty else float("nan")
        optimal_rate = float((optimal_series >= 1.0).mean()) if not optimal_series.empty else float("nan")
        avg_regret = float(regret_series.mean()) if not regret_series.empty else float("nan")
        print(
            f"Completed {len(rows)} steps. Initial={args.initial_capital:.6f}, Final={float(final['wealth']):.6f}."
        )
        if np.isfinite(hit_rate):
            print(f"Interval hit-rate (ratio>=1): {hit_rate:.2%}; loss-rate (ratio<1): {loss_rate:.2%}.")
        if np.isfinite(optimal_rate):
            print(
                "Ex-post optimality rate vs oracle: "
                f"{optimal_rate:.2%}; avg regret={avg_regret:.6f}; certified intervals={len(optimal_series)}."
            )
        if not oracle_series.empty:
            print(f"Oracle median interval ratio: {float(np.median(oracle_series)):.6f}.")
        print(f"Run log: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
