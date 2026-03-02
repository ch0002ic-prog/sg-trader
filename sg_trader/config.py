from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_REIT_TICKERS = {
    "A17U.SI": "CapitaLand Ascendas REIT",
    "M44U.SI": "Mapletree Logistics Trust",
    "ME8U.SI": "Mapletree Industrial Trust",
    "AJBU.SI": "Keppel DC REIT",
    "DCRU.SI": "Digital Core REIT",
}


@dataclass
class AppConfig:
    telegram_token: str
    telegram_chat_id: str
    ticker: str = "^SPX"
    vix_ticker: str = "^VIX"
    vvix_ticker: str = "^VVIX"
    growth_ticker: str = "QQQ"
    reit_tickers: dict[str, str] = field(
        default_factory=lambda: DEFAULT_REIT_TICKERS.copy()
    )
    alpha_spread_threshold: float = 5.0
    vvix_safe_threshold: float = 110.0
    reit_spread_threshold: float = 3.1
    risk_free_rate: float = 3.65
    alloc_fortress: float | None = None
    alloc_alpha: float | None = None
    alloc_shield: float | None = None
    alloc_growth: float | None = None
    drift_band: float = 0.05
    growth_ma_days: int = 200
    dedup_window_seconds: int = 300
    mas_months_back: int = 2
    cache_dir: str = ".cache"
    mas_cache_max_age_hours: int = 12
    mas_cache_warn_pct: float = 0.8
    paper_max_position_qty: float = 5.0
    paper_max_daily_loss: float = 0.0
    paper_kill_switch: bool = False
    data_freshness_days: float = 2.0
    market_cache_max_age_hours: int = 48
    market_cache_warn_pct: float = 0.8
    paper_max_daily_trades: int = 10
    paper_max_notional: float = 100000.0
    paper_initial_capital: float = 100000.0
    paper_vol_kill_threshold: float = 40.0
    pnl_downside_min_days: int = 10
    monitoring_window_days: int = 7
    monitoring_spike_mult: float = 2.0
    monitoring_min_days: int = 3
    monitoring_regime_window_days: int = 60
    monitoring_regime_zscore: float = 2.0
    monitoring_regime_critical_zscore: float = 3.0
    monitoring_regime_min_points: int = 20
    monitoring_signal_drift_window_days: int = 3
    monitoring_signal_drift_low_mult: float = 0.5
    monitoring_alert_throttle_hours: float = 6.0
    monitoring_alerts_enabled: bool = False
    telegram_timeout_seconds: float = 10.0
    telegram_retries: int = 3
    telegram_backoff_seconds: float = 1.0
    execution_plan_max_age_hours: float = 24.0
    log_file: str = "fortress_alpha_ledger.json"


def load_config() -> AppConfig:
    load_dotenv()

    def _parse_optional_int(value: str, default: int) -> int:
        if not value:
            return default
        try:
            return int(value)
        except ValueError:
            return default

    def _parse_optional_float(value: str) -> float | None:
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def _parse_bool(value: str) -> bool:
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}

    return AppConfig(
        telegram_token=os.getenv("TELEGRAM_TOKEN", ""),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        alloc_fortress=_parse_optional_float(os.getenv("ALLOC_FORTRESS", "")),
        alloc_alpha=_parse_optional_float(os.getenv("ALLOC_ALPHA", "")),
        alloc_shield=_parse_optional_float(os.getenv("ALLOC_SHIELD", "")),
        alloc_growth=_parse_optional_float(os.getenv("ALLOC_GROWTH", "")),
        drift_band=float(os.getenv("DRIFT_BAND", "0.05")),
        dedup_window_seconds=_parse_optional_int(
            os.getenv("DEDUP_WINDOW_SECONDS", ""), 300
        ),
        growth_ma_days=_parse_optional_int(os.getenv("GROWTH_MA_DAYS", ""), 200),
        mas_months_back=_parse_optional_int(os.getenv("MAS_MONTHS_BACK", ""), 2),
        cache_dir=os.getenv("CACHE_DIR", ".cache"),
        mas_cache_max_age_hours=_parse_optional_int(
            os.getenv("MAS_CACHE_MAX_AGE_HOURS", ""), 12
        ),
        mas_cache_warn_pct=float(os.getenv("MAS_CACHE_WARN_PCT", "0.8")),
        paper_max_position_qty=_parse_optional_float(
            os.getenv("PAPER_MAX_POSITION_QTY", "5")
        )
        or 5.0,
        paper_max_daily_loss=_parse_optional_float(
            os.getenv("PAPER_MAX_DAILY_LOSS", "0")
        )
        or 0.0,
        paper_kill_switch=_parse_bool(os.getenv("PAPER_KILL_SWITCH", "")),
        data_freshness_days=_parse_optional_float(
            os.getenv("DATA_FRESHNESS_DAYS", "2")
        )
        or 2.0,
        market_cache_max_age_hours=_parse_optional_int(
            os.getenv("MARKET_CACHE_MAX_AGE_HOURS", ""), 48
        ),
        market_cache_warn_pct=float(os.getenv("MARKET_CACHE_WARN_PCT", "0.8")),
        paper_max_daily_trades=_parse_optional_int(
            os.getenv("PAPER_MAX_DAILY_TRADES", ""), 10
        ),
        paper_max_notional=_parse_optional_float(
            os.getenv("PAPER_MAX_NOTIONAL", "100000")
        )
        or 100000.0,
        paper_initial_capital=_parse_optional_float(
            os.getenv("PAPER_INITIAL_CAPITAL", "100000")
        )
        or 100000.0,
        paper_vol_kill_threshold=_parse_optional_float(
            os.getenv("PAPER_VOL_KILL_THRESHOLD", "40")
        )
        or 40.0,
        pnl_downside_min_days=_parse_optional_int(
            os.getenv("PNL_DOWNSIDE_MIN_DAYS", "10"), 10
        ),
        monitoring_window_days=_parse_optional_int(
            os.getenv("MONITORING_WINDOW_DAYS", ""), 7
        ),
        monitoring_spike_mult=_parse_optional_float(
            os.getenv("MONITORING_SPIKE_MULT", "2")
        )
        or 2.0,
        monitoring_min_days=_parse_optional_int(
            os.getenv("MONITORING_MIN_DAYS", ""), 3
        ),
        monitoring_regime_window_days=_parse_optional_int(
            os.getenv("MONITORING_REGIME_WINDOW_DAYS", ""), 60
        ),
        monitoring_regime_zscore=_parse_optional_float(
            os.getenv("MONITORING_REGIME_ZSCORE", "2")
        )
        or 2.0,
        monitoring_regime_critical_zscore=_parse_optional_float(
            os.getenv("MONITORING_REGIME_CRITICAL_ZSCORE", "3")
        )
        or 3.0,
        monitoring_regime_min_points=_parse_optional_int(
            os.getenv("MONITORING_REGIME_MIN_POINTS", ""), 20
        ),
        monitoring_signal_drift_window_days=_parse_optional_int(
            os.getenv("MONITORING_SIGNAL_DRIFT_WINDOW_DAYS", ""), 3
        ),
        monitoring_signal_drift_low_mult=_parse_optional_float(
            os.getenv("MONITORING_SIGNAL_DRIFT_LOW_MULT", "0.5")
        )
        or 0.5,
        monitoring_alert_throttle_hours=_parse_optional_float(
            os.getenv("MONITORING_ALERT_THROTTLE_HOURS", "6")
        )
        or 6.0,
        monitoring_alerts_enabled=_parse_bool(
            os.getenv("MONITORING_ALERTS_ENABLED", "")
        ),
        telegram_timeout_seconds=_parse_optional_float(
            os.getenv("TELEGRAM_TIMEOUT_SECONDS", "10")
        )
        or 10.0,
        telegram_retries=_parse_optional_int(
            os.getenv("TELEGRAM_RETRIES", ""), 3
        ),
        telegram_backoff_seconds=_parse_optional_float(
            os.getenv("TELEGRAM_BACKOFF_SECONDS", "1")
        )
        or 1.0,
        execution_plan_max_age_hours=_parse_optional_float(
            os.getenv("EXECUTION_PLAN_MAX_AGE_HOURS", "24")
        )
        or 24.0,
    )


def validate_config(config: AppConfig) -> list[str]:
    errors: list[str] = []
    if config.alpha_spread_threshold <= 0:
        errors.append("alpha_spread_threshold must be positive")
    if config.vvix_safe_threshold <= 0:
        errors.append("vvix_safe_threshold must be positive")
    if config.reit_spread_threshold < 0:
        errors.append("reit_spread_threshold must be non-negative")
    if config.drift_band <= 0 or config.drift_band >= 0.5:
        errors.append("drift_band must be between 0 and 0.5")
    if config.growth_ma_days <= 0:
        errors.append("growth_ma_days must be positive")
    if config.dedup_window_seconds < 0:
        errors.append("dedup_window_seconds must be non-negative")
    if config.mas_months_back < 0:
        errors.append("mas_months_back must be non-negative")
    if config.mas_cache_max_age_hours <= 0:
        errors.append("mas_cache_max_age_hours must be positive")
    if not 0 < config.mas_cache_warn_pct <= 1:
        errors.append("mas_cache_warn_pct must be between 0 and 1")
    if config.paper_max_position_qty <= 0:
        errors.append("paper_max_position_qty must be positive")
    if config.paper_max_daily_loss < 0:
        errors.append("paper_max_daily_loss must be non-negative")
    if config.data_freshness_days <= 0:
        errors.append("data_freshness_days must be positive")
    if config.market_cache_max_age_hours <= 0:
        errors.append("market_cache_max_age_hours must be positive")
    if not 0 < config.market_cache_warn_pct <= 1:
        errors.append("market_cache_warn_pct must be between 0 and 1")
    if config.paper_max_daily_trades < 0:
        errors.append("paper_max_daily_trades must be non-negative")
    if config.paper_max_notional <= 0:
        errors.append("paper_max_notional must be positive")
    if config.paper_initial_capital <= 0:
        errors.append("paper_initial_capital must be positive")
    if config.paper_vol_kill_threshold < 0:
        errors.append("paper_vol_kill_threshold must be non-negative")
    if config.pnl_downside_min_days < 2:
        errors.append("pnl_downside_min_days must be at least 2")
    if config.monitoring_window_days <= 1:
        errors.append("monitoring_window_days must be greater than 1")
    if config.monitoring_spike_mult <= 0:
        errors.append("monitoring_spike_mult must be positive")
    if config.monitoring_min_days < 1:
        errors.append("monitoring_min_days must be at least 1")
    if config.monitoring_regime_window_days <= 1:
        errors.append("monitoring_regime_window_days must be greater than 1")
    if config.monitoring_regime_zscore <= 0:
        errors.append("monitoring_regime_zscore must be positive")
    if config.monitoring_regime_critical_zscore <= 0:
        errors.append("monitoring_regime_critical_zscore must be positive")
    if config.monitoring_regime_min_points < 5:
        errors.append("monitoring_regime_min_points must be at least 5")
    if config.monitoring_signal_drift_window_days < 1:
        errors.append("monitoring_signal_drift_window_days must be at least 1")
    if not 0 < config.monitoring_signal_drift_low_mult < 1:
        errors.append("monitoring_signal_drift_low_mult must be between 0 and 1")
    if config.monitoring_alert_throttle_hours < 0:
        errors.append("monitoring_alert_throttle_hours must be non-negative")
    if config.telegram_timeout_seconds <= 0:
        errors.append("telegram_timeout_seconds must be positive")
    if config.telegram_retries < 0:
        errors.append("telegram_retries must be non-negative")
    if config.telegram_backoff_seconds < 0:
        errors.append("telegram_backoff_seconds must be non-negative")
    if config.execution_plan_max_age_hours <= 0:
        errors.append("execution_plan_max_age_hours must be positive")

    allocs = [
        config.alloc_fortress,
        config.alloc_alpha,
        config.alloc_shield,
        config.alloc_growth,
    ]
    if all(value is not None for value in allocs):
        total = sum(value for value in allocs if value is not None)
        if total <= 0:
            errors.append("allocation total must be positive")
        elif abs(total - 1.0) > 0.2 and abs(total - 100.0) > 20:
            errors.append("allocation total must be near 1.0 or 100.0")

    return errors
