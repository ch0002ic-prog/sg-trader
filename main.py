from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sg_trader.config import load_config
from sg_trader.execution import (
    DryRunBroker,
    ExecutionRequest,
    ExecutionRouter,
    ExternalBrokerStub,
    ManualBroker,
    PaperBroker,
)
from sg_trader.logging_utils import log_transaction
from sg_trader.portfolio_dashboard import write_portfolio_dashboard
from sg_trader.signals import fetch_close_series


SYNTHETIC_TICKERS = {"N/A", "PORTFOLIO", "S-REIT_BASKET", "SPX_PUT"}
CLI_VERSION = "2026.02"


@dataclass
class TickerMetrics:
    ticker: str
    price: float
    lookback_return: float
    annualized_volatility: float
    max_lookback_drawdown: float
    score: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ledger-native allocator. Tickers are sourced only from fortress_alpha_ledger.json."
        )
    )
    parser.add_argument(
        "--ledger-path",
        type=Path,
        default=Path("fortress_alpha_ledger.json"),
        help="Path to fortress alpha ledger JSON.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=63,
        help="Lookback window used for return/volatility scoring.",
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.0,
        help="Annualized risk-free rate in decimal form (example: 0.0365 for 3.65%%).",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=0.30,
        help="Maximum portfolio weight per ticker (0-1).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Maximum number of tickers retained after scoring.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Optional minimum risk-adjusted score required for selection.",
    )
    parser.add_argument(
        "--max-annualized-volatility",
        type=float,
        default=None,
        help="Optional maximum annualized volatility allowed for selection.",
    )
    parser.add_argument(
        "--max-lookback-drawdown",
        type=float,
        default=None,
        help="Optional maximum absolute drawdown over lookback window (0-1).",
    )
    parser.add_argument(
        "--initial-wealth",
        type=float,
        default=1.0,
        help="Initial wealth amount used for allocation output.",
    )
    parser.add_argument(
        "--list-tickers",
        action="store_true",
        help="List extracted ledger tickers and exit.",
    )
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Include synthetic/non-tradable ledger symbols in list output.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/ledger_universe_allocation.json"),
        help="Path to write JSON report.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Do not append allocation proposal into ledger.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print CLI version and exit.",
    )
    parser.add_argument(
        "--cli-capabilities-json",
        action="store_true",
        help="Print machine-readable CLI capabilities JSON and exit.",
    )
    parser.add_argument(
        "--healthcheck",
        action="store_true",
        help="Run deployment health checks and print text output.",
    )
    parser.add_argument(
        "--healthcheck-json",
        action="store_true",
        help="Run deployment health checks and print JSON output.",
    )
    parser.add_argument(
        "--list-brokers",
        action="store_true",
        help="List available execution brokers.",
    )
    parser.add_argument(
        "--portfolio-dashboard",
        action="store_true",
        help="Generate portfolio dashboard outputs from ledger execution entries.",
    )
    parser.add_argument("--portfolio-start", type=str, default="")
    parser.add_argument("--portfolio-end", type=str, default="")
    parser.add_argument(
        "--portfolio-skip-recent",
        action="store_true",
        help="Skip recent performance sections in markdown dashboard.",
    )
    parser.add_argument("--execution-approve", type=str, default="")
    parser.add_argument("--execution-approve-reason", type=str, default="")
    parser.add_argument(
        "--execution-approve-id-only",
        action="store_true",
        help="When used with --execution-approve, print only the approved plan ID.",
    )
    parser.add_argument("--execution-plan", action="store_true")
    parser.add_argument(
        "--execution-ci-smoke",
        action="store_true",
        help="Run plan generation + approval + deterministic replay in one command.",
    )
    parser.add_argument(
        "--execution-ci-smoke-json",
        action="store_true",
        help="When used with --execution-ci-smoke, print a single JSON payload.",
    )
    parser.add_argument(
        "--execution-plan-id-only",
        action="store_true",
        help="When used with --execution-plan, print only the generated plan ID.",
    )
    parser.add_argument("--execution-replay", type=str, default="")
    parser.add_argument(
        "--execution-replay-json",
        action="store_true",
        help="When used with --execution-replay, print a single JSON payload.",
    )
    parser.add_argument("--execution-replay-broker", type=str, default="")
    parser.add_argument("--execution-plan-max-age-hours", type=float, default=None)
    parser.add_argument("--execution-approval-max-age-hours", type=float, default=None)
    parser.add_argument("--execution-broker", type=str, default="paper")
    parser.add_argument("--paper-symbol", type=str, default="SPX_PUT")
    parser.add_argument("--paper-side", type=str, default="SELL")
    parser.add_argument("--paper-qty", type=float, default=1.0)
    parser.add_argument("--paper-reference-price", type=float, default=1.25)
    parser.add_argument("--paper-slippage-bps", type=float, default=5.0)
    parser.add_argument("--paper-latency-ms", type=int, default=150)
    parser.add_argument("--exec-correlation-id", type=str, default="")
    parser.add_argument("--paper-seed", type=int, default=None)
    parser.add_argument("--paper-mark-price", type=float, default=None)
    parser.add_argument(
        "--execution-replay-skip-risk",
        action="store_true",
        help="Skip replay risk checks.",
    )
    parser.add_argument(
        "--execution-replay-allow-random",
        action="store_true",
        help="Allow replay without deterministic paper seed.",
    )
    return parser


def _plan_path_from_ref(plan_ref: str) -> Path:
    plan_path = Path(plan_ref)
    if not plan_path.suffix:
        plan_path = Path("reports") / "execution_plans" / f"execution_plan_{plan_ref}.json"
    return plan_path


def _load_json_dict(
    path: Path,
    *,
    missing_prefix: str,
    invalid_prefix: str,
    invalid_payload_message: str,
) -> dict[str, Any] | None:
    if not path.exists():
        print(f"{missing_prefix}: {path}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"{invalid_prefix}: {exc}")
        return None
    if not isinstance(payload, dict):
        print(invalid_payload_message)
        return None
    payload["_path"] = str(path)
    return payload


def _verify_plan_hash(plan: dict[str, Any]) -> tuple[bool, str]:
    if "plan_hash" not in plan:
        return False, "Missing plan_hash."
    plan_copy = {k: v for k, v in plan.items() if k not in {"plan_hash", "_path"}}
    serialized = json.dumps(plan_copy, sort_keys=True)
    computed = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    if computed != plan.get("plan_hash"):
        return False, "Plan hash mismatch."
    return True, ""


def _age_hours(timestamp: str) -> float | None:
    if not timestamp:
        return None
    try:
        ts = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return (datetime.now() - ts).total_seconds() / 3600.0


def _load_execution_approval(plan_id: str) -> dict[str, Any] | None:
    approval_path = Path("reports") / "execution_plans" / f"execution_approval_{plan_id}.json"
    return _load_json_dict(
        approval_path,
        missing_prefix="Execution approval not found",
        invalid_prefix="Invalid execution approval JSON",
        invalid_payload_message="Invalid execution approval payload.",
    )


def _build_router() -> ExecutionRouter:
    router = ExecutionRouter()
    router.register("paper", PaperBroker())
    router.register_adapter(ManualBroker())
    router.register_adapter(DryRunBroker())
    router.register_adapter(ExternalBrokerStub())
    return router


def _evaluate_replay_risk(config: Any) -> dict[str, Any]:
    reasons: list[str] = []
    if bool(config.paper_kill_switch):
        reasons.append("Paper kill switch is enabled.")
    return {"blocked": bool(reasons), "reasons": reasons}


def _build_execution_plan(
    *,
    broker: str,
    symbol: str,
    side: str,
    quantity: float,
    reference_price: float,
    slippage_bps: float,
    latency_ms: int,
    correlation_id: str,
    risk: dict[str, Any],
) -> dict[str, Any]:
    slip_factor = max(0.0, slippage_bps) / 10000.0
    if side == "BUY":
        expected_min = reference_price
        expected_max = reference_price * (1 + slip_factor)
    else:
        expected_min = reference_price * (1 - slip_factor)
        expected_max = reference_price

    plan_id = uuid.uuid4().hex
    plan = {
        "plan_id": plan_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "broker": broker,
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "reference_price": reference_price,
        "slippage_bps": slippage_bps,
        "latency_ms": latency_ms,
        "expected_fill_range": {
            "min": expected_min,
            "max": expected_max,
        },
        "risk": risk,
        "correlation_id": correlation_id,
    }
    payload = json.dumps(plan, sort_keys=True)
    plan["plan_hash"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return plan


def _validate_paper_inputs(args: argparse.Namespace) -> tuple[bool, str]:
    side = args.paper_side.upper().strip()
    if side not in {"BUY", "SELL"}:
        return False, "paper side must be BUY or SELL."
    if args.paper_qty <= 0:
        return False, "paper qty must be positive."
    if args.paper_reference_price <= 0:
        return False, "paper reference price must be positive."
    if args.paper_latency_ms < 0:
        return False, "paper latency must be non-negative."
    if args.paper_slippage_bps < 0:
        return False, "paper slippage bps must be non-negative."
    return True, ""


def load_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Ledger not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Ledger JSON must be a list of entries")
    return data


def _walk_for_tickers(payload: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_lower = str(key).lower()
            if key_lower in {"ticker", "symbol"} and isinstance(value, str):
                val = value.strip()
                if val:
                    found.add(val)
            found |= _walk_for_tickers(value)
    elif isinstance(payload, list):
        for item in payload:
            found |= _walk_for_tickers(item)
    return found


def extract_ledger_tickers(entries: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    all_symbols: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        ticker = entry.get("ticker")
        if isinstance(ticker, str) and ticker.strip():
            all_symbols.add(ticker.strip())
        details = entry.get("details")
        all_symbols |= _walk_for_tickers(details)

    clean = sorted({symbol for symbol in all_symbols if symbol})
    tradable = sorted(
        symbol for symbol in clean if symbol.upper() not in SYNTHETIC_TICKERS
    )
    return tradable, clean


def compute_metrics(
    tickers: list[str],
    *,
    lookback_days: int,
    risk_free_rate: float,
    cache_dir: Path,
    max_age_hours: int,
) -> list[TickerMetrics]:
    metrics: list[TickerMetrics] = []
    for ticker in tickers:
        series = fetch_close_series(
            ticker,
            period="1y",
            cache_dir=cache_dir,
            max_age_hours=max_age_hours,
        )
        if series is None or series.empty:
            continue

        closes = pd.Series(series).dropna()
        if len(closes) <= lookback_days:
            continue

        window = closes.iloc[-(lookback_days + 1) :]
        returns = np.log(window / window.shift(1)).dropna()
        if returns.empty:
            continue

        running_max = window.cummax()
        drawdown_series = (window / running_max) - 1.0
        max_lookback_drawdown = float(abs(drawdown_series.min()))

        lookback_return = float((window.iloc[-1] / window.iloc[0]) - 1.0)
        annualized_vol = float(returns.std(ddof=1) * np.sqrt(252))
        if not np.isfinite(annualized_vol):
            continue
        vol_floor = max(annualized_vol, 1e-8)
        annualized_rf_window = risk_free_rate * (lookback_days / 252.0)
        score = (lookback_return - annualized_rf_window) / vol_floor

        metrics.append(
            TickerMetrics(
                ticker=ticker,
                price=float(window.iloc[-1]),
                lookback_return=lookback_return,
                annualized_volatility=annualized_vol,
                max_lookback_drawdown=max_lookback_drawdown,
                score=float(score),
            )
        )

    metrics.sort(key=lambda item: item.score, reverse=True)
    return metrics


def capped_weights(scores: list[float], max_weight: float) -> list[float]:
    if not scores:
        return []
    positive = np.maximum(np.array(scores, dtype=float), 0.0)
    if positive.sum() <= 0:
        return [0.0 for _ in scores]

    raw = positive / positive.sum()
    weights = np.zeros_like(raw)
    remaining = 1.0
    remaining_idx = set(range(len(raw)))

    while remaining_idx and remaining > 1e-12:
        prorata_base = raw[list(remaining_idx)].sum()
        if prorata_base <= 0:
            break
        clipped_this_round = False
        for idx in list(remaining_idx):
            w = remaining * raw[idx] / prorata_base
            if w >= max_weight:
                weights[idx] = max_weight
                remaining -= max_weight
                remaining_idx.remove(idx)
                clipped_this_round = True
        if not clipped_this_round:
            for idx in remaining_idx:
                weights[idx] = remaining * raw[idx] / prorata_base
            remaining = 0.0

    total = float(weights.sum())
    if total <= 0:
        return [0.0 for _ in scores]
    normalized = (weights / total).tolist()
    return [float(value) for value in normalized]


def apply_strategy_filters(
    metrics: list[TickerMetrics],
    *,
    min_score: float | None,
    max_annualized_volatility: float | None,
    max_lookback_drawdown: float | None,
) -> list[TickerMetrics]:
    filtered = metrics
    if min_score is not None:
        filtered = [item for item in filtered if item.score >= min_score]
    if max_annualized_volatility is not None:
        filtered = [
            item
            for item in filtered
            if item.annualized_volatility <= max_annualized_volatility
        ]
    if max_lookback_drawdown is not None:
        filtered = [
            item for item in filtered if item.max_lookback_drawdown <= max_lookback_drawdown
        ]
    return filtered


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _check_dir_writable(path: Path) -> tuple[bool, str | None]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".healthcheck_{uuid.uuid4().hex}.tmp"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True, None
    except Exception as exc:
        return False, str(exc)


def _run_healthcheck(ledger_path: Path, cache_dir: Path) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    ledger_check: dict[str, Any] = {"ok": False}
    if not ledger_path.exists():
        ledger_check["error"] = f"ledger not found: {ledger_path}"
    else:
        try:
            payload = json.loads(ledger_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                ledger_check["ok"] = True
                ledger_check["entries"] = len(payload)
            else:
                ledger_check["error"] = "ledger payload is not a JSON list"
        except json.JSONDecodeError as exc:
            ledger_check["error"] = f"invalid ledger json: {exc}"
        except Exception as exc:
            ledger_check["error"] = str(exc)
    checks["ledger_readable"] = ledger_check

    reports_ok, reports_err = _check_dir_writable(Path("reports"))
    checks["reports_writable"] = {"ok": reports_ok, "error": reports_err}

    plans_ok, plans_err = _check_dir_writable(Path("reports") / "execution_plans")
    checks["execution_plans_writable"] = {"ok": plans_ok, "error": plans_err}

    cache_ok, cache_err = _check_dir_writable(cache_dir)
    checks["cache_writable"] = {"ok": cache_ok, "error": cache_err}

    overall_ok = all(bool(item.get("ok")) for item in checks.values())
    return {
        "ok": overall_ok,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ledger_path": str(ledger_path),
        "cache_dir": str(cache_dir),
        "checks": checks,
    }


def _cli_capabilities_payload() -> dict[str, Any]:
    return {
        "name": "sg-trader",
        "version": CLI_VERSION,
        "entrypoint": "main.py",
        "outputs": {
            "default_report": "reports/ledger_universe_allocation.json",
            "execution_plan_dir": "reports/execution_plans",
            "portfolio_dashboard_prefix": "reports/portfolio_dashboard_all",
        },
        "commands": {
            "allocator": {
                "default": True,
                "flags": [
                    "--ledger-path",
                    "--lookback-days",
                    "--risk-free-rate",
                    "--max-weight",
                    "--top-n",
                    "--min-score",
                    "--max-annualized-volatility",
                    "--max-lookback-drawdown",
                    "--initial-wealth",
                    "--report-path",
                    "--no-log",
                ],
            },
            "universe": {
                "flags": ["--list-tickers", "--include-synthetic"],
            },
            "brokers": {
                "flags": ["--list-brokers"],
            },
            "execution_plan": {
                "flags": [
                    "--execution-plan",
                    "--execution-plan-id-only",
                    "--execution-broker",
                    "--paper-symbol",
                    "--paper-side",
                    "--paper-qty",
                    "--paper-reference-price",
                    "--paper-slippage-bps",
                    "--paper-latency-ms",
                    "--exec-correlation-id",
                ],
            },
            "execution_approve": {
                "flags": [
                    "--execution-approve",
                    "--execution-approve-reason",
                    "--execution-approve-id-only",
                ],
            },
            "execution_replay": {
                "flags": [
                    "--execution-replay",
                    "--execution-replay-json",
                    "--execution-replay-broker",
                    "--execution-plan-max-age-hours",
                    "--execution-approval-max-age-hours",
                    "--execution-replay-skip-risk",
                    "--execution-replay-allow-random",
                    "--paper-seed",
                    "--paper-mark-price",
                ],
            },
            "execution_ci_smoke": {
                "flags": ["--execution-ci-smoke", "--execution-ci-smoke-json"],
            },
            "portfolio_dashboard": {
                "flags": [
                    "--portfolio-dashboard",
                    "--portfolio-start",
                    "--portfolio-end",
                    "--portfolio-skip-recent",
                ],
            },
            "meta": {
                "flags": [
                    "--version",
                    "--cli-capabilities-json",
                    "--healthcheck",
                    "--healthcheck-json",
                ],
            },
        },
        "exit_codes": {
            "0": "success",
            "2": "allocator argument validation failure",
            "3": "no tradable tickers in ledger",
            "4": "metrics unavailable from market data",
            "5": "plan/approval/dashboard validation failure",
            "6": "execution replay validation or broker execution failure",
            "8": "risk gate blocked replay/ci smoke",
            "9": "healthcheck failure",
        },
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.version:
        print(CLI_VERSION)
        return 0
    if args.cli_capabilities_json:
        print(json.dumps(_cli_capabilities_payload(), sort_keys=True))
        return 0

    cfg = load_config()
    cfg.log_file = str(args.ledger_path)

    if args.healthcheck or args.healthcheck_json:
        health = _run_healthcheck(args.ledger_path, Path(cfg.cache_dir))
        if args.healthcheck_json:
            print(json.dumps(health, sort_keys=True))
        else:
            print(f"HEALTHCHECK: {'OK' if health['ok'] else 'FAILED'}")
            for name, item in health["checks"].items():
                status = "OK" if item.get("ok") else "FAILED"
                detail = ""
                if item.get("entries") is not None:
                    detail = f" entries={item['entries']}"
                elif item.get("error"):
                    detail = f" error={item['error']}"
                print(f"- {name}: {status}{detail}")
        return 0 if health["ok"] else 9

    if args.list_brokers:
        router = _build_router()
        print("Available brokers:")
        for name in router.list_brokers():
            print(f"- {name}")
        return 0

    if args.portfolio_dashboard:
        start_key = args.portfolio_start.strip() or None
        end_key = args.portfolio_end.strip() or None
        try:
            path = write_portfolio_dashboard(
                cfg,
                "reports",
                start_date=start_key,
                end_date=end_key,
                include_recent=not args.portfolio_skip_recent,
            )
        except ValueError as exc:
            print(f"Invalid portfolio dashboard date: {exc}")
            return 5
        print(f"Portfolio dashboard generated: {path}")
        return 0

    if args.execution_approve:
        plan = _load_json_dict(
            _plan_path_from_ref(args.execution_approve.strip()),
            missing_prefix="Execution plan not found",
            invalid_prefix="Invalid execution plan JSON",
            invalid_payload_message="Invalid execution plan payload.",
        )
        if plan is None:
            return 5
        ok, error = _verify_plan_hash(plan)
        if not ok:
            print(f"Execution plan verification failed: {error}")
            return 5
        plan_id = str(plan.get("plan_id", ""))
        output_dir = Path("reports") / "execution_plans"
        output_dir.mkdir(parents=True, exist_ok=True)
        approval = {
            "plan_id": plan_id,
            "plan_hash": plan.get("plan_hash"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "reason": args.execution_approve_reason.strip() or None,
            "plan_path": plan.get("_path"),
        }
        approval_path = output_dir / f"execution_approval_{plan_id}.json"
        approval_path.write_text(json.dumps(approval, indent=2), encoding="utf-8")
        if args.execution_approve_id_only:
            print(plan_id)
        else:
            print(f"Execution approval saved: {approval_path}")
            print(f"Execution approval id: {plan_id}")
        return 0

    if args.execution_ci_smoke:
        valid, error = _validate_paper_inputs(args)
        if not valid:
            print(f"Execution CI smoke blocked: {error}")
            return 5

        side = args.paper_side.upper().strip()
        risk = _evaluate_replay_risk(cfg)
        if risk["blocked"] and not args.execution_replay_skip_risk:
            print("Execution CI smoke blocked:")
            for reason in risk["reasons"]:
                print(f"- {reason}")
            return 8

        correlation_id = args.exec_correlation_id.strip() or uuid.uuid4().hex
        plan = _build_execution_plan(
            broker=args.execution_broker,
            symbol=args.paper_symbol.strip(),
            side=side,
            quantity=float(args.paper_qty),
            reference_price=float(args.paper_reference_price),
            slippage_bps=float(args.paper_slippage_bps),
            latency_ms=int(args.paper_latency_ms),
            correlation_id=correlation_id,
            risk=risk,
        )

        output_dir = Path("reports") / "execution_plans"
        output_dir.mkdir(parents=True, exist_ok=True)
        plan_path = output_dir / f"execution_plan_{plan['plan_id']}.json"
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

        approval = {
            "plan_id": plan["plan_id"],
            "plan_hash": plan["plan_hash"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "reason": "execution-ci-smoke",
            "plan_path": str(plan_path),
        }
        approval_path = output_dir / f"execution_approval_{plan['plan_id']}.json"
        approval_path.write_text(json.dumps(approval, indent=2), encoding="utf-8")

        seed = args.paper_seed if args.paper_seed is not None else 1
        broker = args.execution_replay_broker.strip() or args.execution_broker
        router = _build_router()
        request = ExecutionRequest(
            symbol=str(plan["symbol"]),
            side=str(plan["side"]),
            quantity=float(plan["quantity"]),
            reference_price=float(plan["reference_price"]),
            slippage_bps=float(plan["slippage_bps"]),
            latency_ms=int(plan["latency_ms"]),
            seed=seed,
        )
        try:
            result = router.execute(broker, request)
        except (ValueError, RuntimeError) as exc:
            print(str(exc))
            return 6

        summary = {
            "plan_id": plan["plan_id"],
            "plan_path": str(plan_path),
            "approval_path": str(approval_path),
            "broker": broker,
            "result": {
                "symbol": result.symbol,
                "side": result.side,
                "quantity": result.quantity,
                "reference_price": result.reference_price,
                "fill_price": result.fill_price,
                "slippage_bps": result.slippage_bps,
                "latency_ms": result.latency_ms,
            },
        }
        if args.execution_ci_smoke_json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print(f"Execution CI smoke plan id: {plan['plan_id']}")
            print(f"Execution CI smoke plan: {plan_path}")
            print(f"Execution CI smoke approval: {approval_path}")
            print(
                "Execution replay: "
                f"{result.side} {result.quantity} {result.symbol} "
                f"@ {result.fill_price:.4f} (ref {result.reference_price:.4f})"
            )
        return 0

    if args.execution_plan:
        valid, error = _validate_paper_inputs(args)
        if not valid:
            print(f"Execution plan blocked: {error}")
            return 5

        side = args.paper_side.upper().strip()
        risk = _evaluate_replay_risk(cfg)
        correlation_id = args.exec_correlation_id.strip() or uuid.uuid4().hex
        plan = _build_execution_plan(
            broker=args.execution_broker,
            symbol=args.paper_symbol.strip(),
            side=side,
            quantity=float(args.paper_qty),
            reference_price=float(args.paper_reference_price),
            slippage_bps=float(args.paper_slippage_bps),
            latency_ms=int(args.paper_latency_ms),
            correlation_id=correlation_id,
            risk=risk,
        )

        output_dir = Path("reports") / "execution_plans"
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"execution_plan_{plan['plan_id']}.json"
        out_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        if args.execution_plan_id_only:
            print(plan["plan_id"])
        else:
            print(f"Execution plan saved: {out_path}")
            print(f"Execution plan id: {plan['plan_id']}")
            if risk["blocked"]:
                print("Execution plan blocked:")
                for reason in risk["reasons"]:
                    print(f"- {reason}")
        return 0

    if args.execution_replay:
        plan = _load_json_dict(
            _plan_path_from_ref(args.execution_replay.strip()),
            missing_prefix="Execution plan not found",
            invalid_prefix="Invalid execution plan JSON",
            invalid_payload_message="Invalid execution plan payload.",
        )
        if plan is None:
            return 6
        ok, error = _verify_plan_hash(plan)
        if not ok:
            print(f"Execution replay blocked: {error}")
            return 6

        plan_id = str(plan.get("plan_id", ""))
        if not plan_id:
            print("Execution replay blocked: missing plan ID.")
            return 6

        max_age_hours = (
            args.execution_plan_max_age_hours
            if args.execution_plan_max_age_hours is not None
            else cfg.execution_plan_max_age_hours
        )
        plan_age = _age_hours(str(plan.get("timestamp", "")))
        if plan_age is None:
            print("Execution replay blocked: invalid plan timestamp.")
            return 6
        if plan_age > max_age_hours:
            print(
                "Execution replay blocked: plan expired. "
                f"age {plan_age:.2f}h > limit {max_age_hours:.2f}h"
            )
            return 6

        approval = _load_execution_approval(plan_id)
        if approval is None:
            print("Execution replay blocked: approval not found.")
            return 6

        if args.execution_approval_max_age_hours is not None:
            approval_age = _age_hours(str(approval.get("timestamp", "")))
            if approval_age is None:
                print("Execution replay blocked: invalid approval timestamp.")
                return 6
            if approval_age > args.execution_approval_max_age_hours:
                print(
                    "Execution replay blocked: approval expired. "
                    f"age {approval_age:.2f}h > limit {args.execution_approval_max_age_hours:.2f}h"
                )
                return 6

        if approval.get("plan_hash") != plan.get("plan_hash"):
            print("Execution replay blocked: approval hash mismatch.")
            return 6

        symbol = str(plan.get("symbol", ""))
        side = str(plan.get("side", "")).upper()
        quantity = float(plan.get("quantity", 0.0))
        reference_price = float(plan.get("reference_price", 0.0))
        slippage_bps = float(plan.get("slippage_bps", 0.0))
        latency_ms = int(plan.get("latency_ms", 0))
        broker_override = args.execution_replay_broker.strip()
        broker = broker_override or str(plan.get("broker", "")) or args.execution_broker
        if broker_override:
            planned_broker = str(plan.get("broker", ""))
            if planned_broker and planned_broker != broker_override:
                print(f"Execution replay broker override: {planned_broker} -> {broker_override}")

        if not symbol or side not in {"BUY", "SELL"} or quantity <= 0:
            print("Execution replay blocked: invalid plan inputs.")
            return 6
        if not args.execution_replay_allow_random and args.paper_seed is None:
            print("Execution replay blocked: missing --paper-seed.")
            return 6

        if args.execution_replay_skip_risk:
            risk = {"blocked": False, "reasons": []}
            print("Execution replay risk checks skipped.")
        else:
            risk = _evaluate_replay_risk(cfg)
            if risk["blocked"]:
                print("Execution replay blocked:")
                for reason in risk["reasons"]:
                    print(f"- {reason}")
                return 8

        router = _build_router()
        request = ExecutionRequest(
            symbol=symbol,
            side=side,
            quantity=quantity,
            reference_price=reference_price,
            slippage_bps=slippage_bps,
            latency_ms=latency_ms,
            seed=args.paper_seed,
        )
        try:
            result = router.execute(broker, request)
        except (ValueError, RuntimeError) as exc:
            print(str(exc))
            return 6

        summary = {
            "plan_id": plan_id,
            "broker": broker,
            "risk_checks_skipped": bool(args.execution_replay_skip_risk),
            "result": {
                "symbol": result.symbol,
                "side": result.side,
                "quantity": result.quantity,
                "reference_price": result.reference_price,
                "fill_price": result.fill_price,
                "slippage_bps": result.slippage_bps,
                "latency_ms": result.latency_ms,
            },
        }
        if args.paper_mark_price is not None:
            summary["paper_mark_price"] = args.paper_mark_price

        if args.execution_replay_json:
            print(json.dumps(summary, sort_keys=True))
        else:
            print(
                "Execution replay: "
                f"{result.side} {result.quantity} {result.symbol} "
                f"@ {result.fill_price:.4f} (ref {result.reference_price:.4f})"
            )
            if args.paper_mark_price is not None:
                print(f"Paper mark price: {args.paper_mark_price:.4f}")
        return 0

    if args.lookback_days < 2:
        print("lookback-days must be at least 2")
        return 2
    if args.top_n < 1:
        print("top-n must be at least 1")
        return 2
    if args.max_annualized_volatility is not None and args.max_annualized_volatility <= 0:
        print("max-annualized-volatility must be positive")
        return 2
    if args.max_lookback_drawdown is not None and not (0 < args.max_lookback_drawdown <= 1):
        print("max-lookback-drawdown must be in (0, 1]")
        return 2
    if not (0 < args.max_weight <= 1):
        print("max-weight must be in (0, 1]")
        return 2
    if args.initial_wealth <= 0:
        print("initial-wealth must be positive")
        return 2

    entries = load_ledger(args.ledger_path)
    tradable_tickers, all_tickers = extract_ledger_tickers(entries)

    if args.list_tickers:
        tickers_to_print = all_tickers if args.include_synthetic else tradable_tickers
        print("\n".join(tickers_to_print))
        return 0

    if not tradable_tickers:
        print("No tradable tickers found in ledger.")
        return 3

    metrics = compute_metrics(
        tradable_tickers,
        lookback_days=args.lookback_days,
        risk_free_rate=args.risk_free_rate,
        cache_dir=Path(cfg.cache_dir),
        max_age_hours=cfg.market_cache_max_age_hours,
    )

    if not metrics:
        print("No ticker metrics computed (missing data or insufficient lookback window).")
        return 4

    filtered_metrics = apply_strategy_filters(
        metrics,
        min_score=args.min_score,
        max_annualized_volatility=args.max_annualized_volatility,
        max_lookback_drawdown=args.max_lookback_drawdown,
    )

    if not filtered_metrics:
        print(
            "No ticker metrics remain after strategy filters "
            "(min-score/max-annualized-volatility/max-lookback-drawdown)."
        )
        return 4

    selected = filtered_metrics[: args.top_n]
    weights = capped_weights([item.score for item in selected], args.max_weight)

    allocations = []
    for item, weight in zip(selected, weights):
        allocations.append(
            {
                "ticker": item.ticker,
                "price": item.price,
                "score": item.score,
                "lookback_return": item.lookback_return,
                "annualized_volatility": item.annualized_volatility,
                "max_lookback_drawdown": item.max_lookback_drawdown,
                "weight": weight,
                "allocation": float(weight * args.initial_wealth),
            }
        )

    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ledger_path": str(args.ledger_path),
        "universe_size": len(tradable_tickers),
        "lookback_days": args.lookback_days,
        "risk_free_rate": args.risk_free_rate,
        "max_weight": args.max_weight,
        "initial_wealth": args.initial_wealth,
        "top_n": args.top_n,
        "filters": {
            "min_score": args.min_score,
            "max_annualized_volatility": args.max_annualized_volatility,
            "max_lookback_drawdown": args.max_lookback_drawdown,
            "candidates_after_filters": len(filtered_metrics),
        },
        "allocations": allocations,
    }
    write_report(args.report_path, report)

    print(f"Ledger universe size: {len(tradable_tickers)}")
    print(f"Selected tickers: {len(allocations)}")
    for row in allocations:
        print(
            f"{row['ticker']}: weight={row['weight']:.4f}, "
            f"alloc=${row['allocation']:.4f}, score={row['score']:.4f}"
        )
    print(f"Report written: {args.report_path}")

    if not args.no_log:
        log_transaction(
            category="Allocator",
            ticker="PORTFOLIO",
            action="REBALANCE_PROPOSAL",
            rationale="Ledger-only ticker universe allocation proposal.",
            config=cfg,
            tags=["allocator", "ledger_universe", "proposal"],
            details={
                "universe_size": len(tradable_tickers),
                "selected": [row["ticker"] for row in allocations],
                "weights": {row["ticker"]: row["weight"] for row in allocations},
                "initial_wealth": args.initial_wealth,
                "lookback_days": args.lookback_days,
                "risk_free_rate": args.risk_free_rate,
            },
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())