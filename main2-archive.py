import argparse
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sg_trader.config import AppConfig, load_config
from sg_trader.signals import fetch_close_series


@dataclass
class DecayConfig:
    tickers: list[str]
    window_days: int
    min_drop_sharpe: float
    min_drop_cagr: float
    warn_drop_sharpe: float
    warn_drop_cagr: float
    start_date: str | None
    data_end_date: str | None
    use_slope: bool
    slope_days: int
    include_vix_regime: bool
    vix_quantile: float
    vol_quantile: float
    vol_quantile_window: int
    trend_ma_days: int
    trend_strength_pct: float
    drawdown_threshold: float
    run_label: str
    policy_min_events: int
    policy_min_events_equity: int
    policy_min_events_vol: int
    policy_rerisk: float
    policy_hold: float
    policy_cap: float
    policy_derisk: float
    policy_rerisk_vol: float
    policy_hold_vol: float
    policy_cap_vol: float
    policy_derisk_vol: float
    policy_rerisk_vol_trend_off: float
    policy_hold_vol_trend_off: float
    policy_cap_vol_trend_off: float
    policy_derisk_vol_trend_off: float
    policy_rerisk_vix_trend_off: float | None
    policy_hold_vix_trend_off: float | None
    policy_cap_vix_trend_off: float | None
    policy_derisk_vix_trend_off: float | None
    policy_rerisk_vvix_trend_off: float | None
    policy_hold_vvix_trend_off: float | None
    policy_cap_vvix_trend_off: float | None
    policy_derisk_vvix_trend_off: float | None
    equity_hold_boost: float
    equity_hold_boost_regime: str
    vol_floor: float
    vol_floor_boost: float
    oos_split_date: str | None
    vol_tickers: set[str]
    weak_tickers: set[str]
    safeguard_negative: bool
    cooldown_days: int
    cooldown_dd_extra: int
    cooldown_reentry_sharpe: float
    policy_fine_overrides: bool
    policy_rs_quantiles: int
    policy_fine_overrides_equity_only: bool
    policy_fine_overrides_require_not_decreasing: bool
    policy_fine_override_tickers: set[str]
    policy_fine_override_rs_bins: set[str]
    dd_hard_stop: bool
    archive_run: bool
    archive_tag: str | None
    output_md: Path
    output_csv_dir: Path


def _rolling_sharpe(returns: pd.Series, window: int) -> pd.Series:
    def _sharpe(values: np.ndarray) -> float:
        if values.size < 2:
            return np.nan
        std = np.std(values, ddof=1)
        if std == 0:
            return np.nan
        return np.sqrt(252.0) * float(np.mean(values) / std)

    return returns.rolling(window).apply(_sharpe, raw=True)


def _rolling_cagr(prices: pd.Series, window: int) -> pd.Series:
    shifted = prices.shift(window)
    cagr = (prices / shifted) ** (252.0 / window) - 1.0
    return cagr.replace([np.inf, -np.inf], np.nan)


def _apply_data_end_date(series: pd.Series, data_end_date: str | None) -> pd.Series:
    if not data_end_date:
        return series
    cutoff = pd.to_datetime(data_end_date, errors="coerce")
    if pd.isna(cutoff):
        return series
    series_index = series.index
    index_tz = getattr(series_index, "tz", None)
    cutoff_ts = pd.Timestamp(cutoff)
    if index_tz is not None and cutoff_ts.tzinfo is None:
        cutoff_ts = cutoff_ts.tz_localize(index_tz)
    elif index_tz is None and cutoff_ts.tzinfo is not None:
        cutoff_ts = cutoff_ts.tz_localize(None)
    return series.loc[series_index <= cutoff_ts]


def _policy_for_combo(row: pd.Series) -> str:
    vix_bucket = row.get("vix_bucket", "unknown")
    trend_state = row.get("trend_state", "unknown")
    drawdown_state = row.get("drawdown_state", "unknown")
    vol_state = row.get("vol_state", "unknown")

    if vix_bucket == "high" and trend_state == "off" and drawdown_state == "dd":
        return "re-risk (mean-revert)"
    if vix_bucket == "high" and trend_state == "off" and drawdown_state == "ok":
        return "de-risk"
    if trend_state == "on" and vol_state == "high":
        return "cap leverage"
    return "hold"


def _build_rs_bin(series: pd.Series, quantiles: int) -> pd.Series:
    valid = series.dropna()
    out = pd.Series(pd.NA, index=series.index, dtype="object")
    if valid.empty:
        return out
    q = max(int(quantiles), 2)
    labels = [f"rs_q{i + 1}" for i in range(q)]
    try:
        bins = pd.qcut(valid, q=q, labels=labels, duplicates="drop")
    except ValueError:
        return out
    out.loc[valid.index] = bins.astype(str)
    return out


def _apply_fine_policy_overrides(
    df: pd.DataFrame,
    *,
    ticker: str,
    vol_tickers: set[str],
    equity_only: bool,
    require_not_decreasing: bool,
    allowed_tickers: set[str],
    allowed_rs_bins: set[str],
) -> pd.DataFrame:
    if df.empty:
        return df
    if equity_only and ticker in vol_tickers:
        return df
    if allowed_tickers and ticker not in allowed_tickers:
        return df
    if not allowed_rs_bins:
        return df
    needed = {"regime_combo", "rs_bin", "policy_action"}
    if not needed.issubset(df.columns):
        return df
    mask = (
        (df["regime_combo"] == "mid|trend_off|dd_ok|vol_low")
        & (df["rs_bin"].isin(allowed_rs_bins))
        & (df["policy_action"] == "de-risk")
    )
    if require_not_decreasing and "decreasing" in df.columns:
        mask = mask & (~df["decreasing"].fillna(False))
    if mask.any():
        df.loc[mask, "policy_action"] = "hold"
    return df


def _apply_policy_thresholds(
    df: pd.DataFrame,
    combo_summary: pd.DataFrame,
    min_events: int,
) -> pd.DataFrame:
    if df.empty or combo_summary.empty:
        df["policy_action"] = "hold"
        return df
    eligible = combo_summary.loc[combo_summary["events"] >= min_events, "regime_combo"]
    eligible_set = set(eligible.tolist())
    df["policy_action"] = np.where(
        df["regime_combo"].isin(eligible_set),
        df["policy_action"],
        "hold",
    )
    return df


def _policy_map_for_ticker(
    config: DecayConfig,
    ticker: str,
    trend_state: str | None = None,
) -> dict[str, float]:
    if ticker in config.vol_tickers and trend_state == "off":
        rerisk = config.policy_rerisk_vol_trend_off
        hold = config.policy_hold_vol_trend_off
        cap = config.policy_cap_vol_trend_off
        derisk = config.policy_derisk_vol_trend_off

        if ticker == "^VIX":
            rerisk = (
                config.policy_rerisk_vix_trend_off
                if config.policy_rerisk_vix_trend_off is not None
                else rerisk
            )
            hold = (
                config.policy_hold_vix_trend_off
                if config.policy_hold_vix_trend_off is not None
                else hold
            )
            cap = (
                config.policy_cap_vix_trend_off
                if config.policy_cap_vix_trend_off is not None
                else cap
            )
            derisk = (
                config.policy_derisk_vix_trend_off
                if config.policy_derisk_vix_trend_off is not None
                else derisk
            )
        elif ticker == "^VVIX":
            rerisk = (
                config.policy_rerisk_vvix_trend_off
                if config.policy_rerisk_vvix_trend_off is not None
                else rerisk
            )
            hold = (
                config.policy_hold_vvix_trend_off
                if config.policy_hold_vvix_trend_off is not None
                else hold
            )
            cap = (
                config.policy_cap_vvix_trend_off
                if config.policy_cap_vvix_trend_off is not None
                else cap
            )
            derisk = (
                config.policy_derisk_vvix_trend_off
                if config.policy_derisk_vvix_trend_off is not None
                else derisk
            )

        return {
            "re-risk (mean-revert)": rerisk,
            "hold": hold,
            "cap leverage": cap,
            "de-risk": derisk,
            "hold_boost": hold,
            "vol_boost": hold,
        }
    if ticker in config.vol_tickers:
        return {
            "re-risk (mean-revert)": config.policy_rerisk_vol,
            "hold": config.policy_hold_vol,
            "cap leverage": config.policy_cap_vol,
            "de-risk": config.policy_derisk_vol,
            "hold_boost": config.policy_hold_vol,
            "vol_boost": config.policy_hold_vol,
        }
    return {
        "re-risk (mean-revert)": config.policy_rerisk,
        "hold": config.policy_hold,
        "cap leverage": config.policy_cap,
        "de-risk": config.policy_derisk,
        "hold_boost": config.equity_hold_boost,
        "vol_boost": config.vol_floor_boost,
    }


def _simulate_overlay(
    prices: pd.Series,
    policy_df: pd.DataFrame,
    config: DecayConfig,
    ticker: str,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> dict[str, float]:
    returns = prices.pct_change().dropna()
    if getattr(returns.index, "tz", None) is not None:
        returns = returns.tz_convert(None)
    if start_date is not None:
        returns = returns[returns.index >= start_date]
    if end_date is not None:
        returns = returns[returns.index < end_date]
    exposure = pd.Series(1.0, index=returns.index)
    cols = ["policy_action"]
    if "trend_state" in policy_df.columns:
        cols.append("trend_state")
    policy_frame = (
        policy_df.set_index("date")[cols]
        .reindex(returns.index)
        .dropna(subset=["policy_action"])
    )
    for idx, row in policy_frame.iterrows():
        action = row["policy_action"]
        trend_state = row.get("trend_state") if "trend_state" in row else None
        if isinstance(trend_state, float) and np.isnan(trend_state):
            trend_state = None
        policy_map = _policy_map_for_ticker(config, ticker, trend_state)
        exposure.loc[idx] = policy_map.get(action, config.policy_hold)

    exposure_lag = exposure.shift(1).fillna(config.policy_hold)
    pnl = (returns * exposure_lag).dropna()
    if pnl.empty:
        return {
            "sharpe": float("nan"),
            "cagr": float("nan"),
            "vol": float("nan"),
            "max_dd": float("nan"),
        }
    sharpe = np.sqrt(252.0) * pnl.mean() / pnl.std()
    cum = (1.0 + pnl).cumprod()
    years = len(cum) / 252.0
    cagr = cum.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 else float("nan")
    vol = pnl.std() * np.sqrt(252.0)
    peak = cum.cummax()
    max_dd = (cum / peak - 1.0).min()
    return {
        "sharpe": float(sharpe),
        "cagr": float(cagr),
        "vol": float(vol),
        "max_dd": float(max_dd),
    }


def _simulate_overlay_split(
    prices: pd.Series,
    policy_df: pd.DataFrame,
    config: DecayConfig,
    ticker: str,
) -> dict[str, dict[str, float]]:
    if not config.oos_split_date:
        return {"full": _simulate_overlay(prices, policy_df, config, ticker)}
    split = pd.to_datetime(config.oos_split_date)
    policy_df = policy_df.copy()
    policy_dates = pd.to_datetime(policy_df["date"])
    if getattr(policy_dates.dt, "tz", None) is not None:
        policy_dates = policy_dates.dt.tz_convert(None)
    policy_df["date"] = policy_dates
    in_sample = policy_df[policy_df["date"] < split]
    oos = policy_df[policy_df["date"] >= split]
    return {
        "full": _simulate_overlay(prices, policy_df, config, ticker),
        "in_sample": _simulate_overlay(
            prices,
            in_sample,
            config,
            ticker,
            end_date=split,
        ),
        "oos": _simulate_overlay(
            prices,
            oos,
            config,
            ticker,
            start_date=split,
        ),
    }


def _equal_weight_daily_proxy(
    label: str,
    config: DecayConfig,
    hold_on_nondecreasing: bool = True,
    hold_exposure: float | None = None,
) -> pd.Series:
    """Equal-weight daily PnL proxy from regime_daily exports.

    Method: exposure defaults to policy_hold on non-decreasing days (or 0.0 if disabled)
    and switches to the policy map when a day is flagged as decreasing.
    """
    pnl_by_ticker = []
    for ticker in config.tickers:
        path = config.output_csv_dir / f"regime_daily_{ticker}_{label}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df["date"] = df["date"].dt.tz_convert(None)
        df = df.dropna(subset=["date", "price"])
        df = df.sort_values("date").set_index("date")
        returns = df["price"].pct_change()
        policy_frame = df[["policy_action", "trend_state", "decreasing"]].reindex(
            returns.index
        )
        if hold_on_nondecreasing:
            default_exposure = (
                hold_exposure if hold_exposure is not None else config.policy_hold
            )
        else:
            default_exposure = 0.0
        exposure = pd.Series(default_exposure, index=returns.index)
        for idx, row in policy_frame.iterrows():
            if not bool(row.get("decreasing", False)):
                continue
            action = row["policy_action"]
            trend_state = row["trend_state"]
            if isinstance(trend_state, float) and np.isnan(trend_state):
                trend_state = None
            policy_map = _policy_map_for_ticker(config, ticker, trend_state)
            exposure.loc[idx] = policy_map.get(action, config.policy_hold)
        exposure_lag = exposure.shift(1).fillna(default_exposure)
        pnl_by_ticker.append((returns * exposure_lag).rename(ticker))

    if not pnl_by_ticker:
        return pd.Series(dtype=float)
    return pd.concat(pnl_by_ticker, axis=1).mean(axis=1, skipna=True)


def _daily_proxy_metrics(pnl: pd.Series) -> dict[str, float]:
    pnl = pnl.dropna()
    if pnl.empty:
        return {"sharpe": float("nan"), "cagr": float("nan")}
    sharpe = np.sqrt(252.0) * pnl.mean() / pnl.std()
    cum = (1.0 + pnl).cumprod()
    years = len(cum) / 252.0
    cagr = cum.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 else float("nan")
    return {"sharpe": float(sharpe), "cagr": float(cagr)}


V9_PROXY_LABELS = [
    "run1_tight_weak_v9_regimeboost",
    "run1_tight_weak_v9_ddhard",
    "run1_tight_weak_v9_volfloor",
]
PROXY_HOLD_EXPOSURE = 1.4


def _format_proxy_label(label: str) -> str:
    prefix = "run1_tight_weak_"
    if label.startswith(prefix):
        return label[len(prefix) :]
    return label


def _build_proxy_section(
    config: DecayConfig,
    labels: list[str],
    hold_exposure: float | None = None,
) -> list[str]:
    rows = []
    for label in labels:
        pnl = _equal_weight_daily_proxy(
            label,
            config,
            hold_on_nondecreasing=True,
            hold_exposure=hold_exposure,
        )
        if pnl.empty:
            continue
        metrics = _daily_proxy_metrics(pnl)
        rows.append((label, metrics))

    if not rows:
        return []

    lines = [
        "### Equal-Weight Daily Portfolio Proxy (Hold on non-decreasing days)",
        "| Run | Sharpe | CAGR |",
        "| --- | --- | --- |",
    ]
    for label, metrics in rows:
        display = _format_proxy_label(label)
        lines.append(
            "| {label} | {sharpe} | {cagr} |".format(
                label=display,
                sharpe=f"{metrics.get('sharpe', float('nan')):.3f}",
                cagr=f"{metrics.get('cagr', float('nan')):.2%}",
            )
        )
    lines.append("")
    lines.append("Notes:")
    lines.append(
        "- Method: core tickers, equal-weight daily returns from regime_daily exports."
    )
    lines.append(
        "- Exposure: policy_hold on non-decreasing days; policy map on decreasing days; applied with a one-day lag to avoid look-ahead bias."
    )
    if hold_exposure is not None:
        lines.append(
            "- Hold exposure uses {value:.2f} to match the legacy table; CAGR uses 252 trading days.".format(
                value=hold_exposure
            )
        )
    else:
        lines.append("- CAGR uses 252 trading days.")
    lines.append("")
    return lines


def _write_overlay_summary(
    overlay_results: dict[str, dict[str, dict[str, float]]],
    output_csv_dir: Path,
    label: str,
) -> Path:
    rows = []
    for ticker, stats in overlay_results.items():
        for segment, values in stats.items():
            rows.append(
                {
                    "ticker": ticker,
                    "segment": segment,
                    "sharpe": values.get("sharpe", float("nan")),
                    "cagr": values.get("cagr", float("nan")),
                    "vol": values.get("vol", float("nan")),
                    "max_dd": values.get("max_dd", float("nan")),
                }
            )
    df = pd.DataFrame(rows).sort_values(["ticker", "segment"]).reset_index(drop=True)
    path = output_csv_dir / f"overlay_summary_{label}.csv"
    df.to_csv(path, index=False)
    return path


def _write_overlay_comparison(
    overlay_results: dict[str, dict[str, dict[str, float]]],
    output_csv_dir: Path,
    label: str,
    config: DecayConfig,
    cache_dir: Path,
    max_age_hours: float,
) -> Path:
    baseline_label = "policy_oos_2019_v2"
    rows = []
    for ticker, stats in overlay_results.items():
        baseline_path = output_csv_dir / f"decay_policy_{ticker}_{baseline_label}.csv"
        if not baseline_path.exists():
            continue
        baseline_df = pd.read_csv(baseline_path)
        if baseline_df.empty:
            continue
        baseline_df["date"] = pd.to_datetime(baseline_df["date"])
        series = fetch_close_series(
            ticker,
            period="max",
            cache_dir=cache_dir,
            max_age_hours=max_age_hours,
        )
        if series is None or series.empty:
            continue
        series = series.sort_index()
        series = _apply_data_end_date(series, config.data_end_date)
        if series.empty:
            continue
        baseline = _simulate_overlay(series, baseline_df, config, ticker)
        run_full = stats.get("full", {})
        rows.append(
            {
                "ticker": ticker,
                "run1_sharpe": run_full.get("sharpe", float("nan")),
                "baseline_sharpe": baseline.get("sharpe", float("nan")),
                "delta_sharpe": run_full.get("sharpe", float("nan"))
                - baseline.get("sharpe", float("nan")),
                "run1_cagr": run_full.get("cagr", float("nan")),
                "baseline_cagr": baseline.get("cagr", float("nan")),
                "delta_cagr": run_full.get("cagr", float("nan"))
                - baseline.get("cagr", float("nan")),
                "run1_vol": run_full.get("vol", float("nan")),
                "baseline_vol": baseline.get("vol", float("nan")),
                "delta_vol": run_full.get("vol", float("nan"))
                - baseline.get("vol", float("nan")),
                "run1_max_dd": run_full.get("max_dd", float("nan")),
                "baseline_max_dd": baseline.get("max_dd", float("nan")),
                "delta_max_dd": run_full.get("max_dd", float("nan"))
                - baseline.get("max_dd", float("nan")),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty and "ticker" in df.columns:
        df = df.sort_values("ticker").reset_index(drop=True)
    path = output_csv_dir / f"overlay_comparison_{label}_vs_policy_oos_2019_v2.csv"
    df.to_csv(path, index=False)
    return path


def _archive_run_outputs(
    output_csv_dir: Path,
    output_md: Path,
    label: str,
    summary_path: Path,
    comparison_path: Path,
) -> Path:
    archive_dir = Path("reports") / "decay_runs" / label
    archive_dir.mkdir(parents=True, exist_ok=True)
    for path in output_csv_dir.glob(f"*_{label}.csv"):
        shutil.copy2(path, archive_dir / path.name)
    if summary_path.exists():
        shutil.copy2(summary_path, archive_dir / summary_path.name)
    if comparison_path.exists():
        shutil.copy2(comparison_path, archive_dir / comparison_path.name)
    if output_md.exists():
        shutil.copy2(output_md, archive_dir / output_md.name)
    return archive_dir


def _build_base_table(
    ticker: str,
    prices: pd.Series,
    window_days: int,
    min_drop_sharpe: float,
    min_drop_cagr: float,
    warn_drop_sharpe: float,
    warn_drop_cagr: float,
    start_date: str | None,
    use_slope: bool,
    slope_days: int,
    vix_series: pd.Series | None,
    vix_quantile: float,
    vol_quantile: float,
    vol_quantile_window: int,
    trend_ma_days: int,
    trend_strength_pct: float,
    drawdown_threshold: float,
    vol_tickers: set[str],
    weak_tickers: set[str],
    safeguard_negative: bool,
    cooldown_days: int,
    cooldown_dd_extra: int,
    cooldown_reentry_sharpe: float,
    policy_fine_overrides: bool,
    policy_rs_quantiles: int,
    policy_fine_overrides_equity_only: bool,
    policy_fine_overrides_require_not_decreasing: bool,
    policy_fine_override_tickers: set[str],
    policy_fine_override_rs_bins: set[str],
    equity_hold_boost: float,
    equity_hold_boost_regime: str,
    vol_floor: float,
    vol_floor_boost: float,
    dd_hard_stop: bool,
) -> pd.DataFrame:
    returns = prices.pct_change().dropna()
    sharpe = _rolling_sharpe(returns, window_days)
    cagr = _rolling_cagr(prices, window_days)

    df = pd.DataFrame(
        {
            "price": prices,
            "rolling_sharpe": sharpe,
            "rolling_cagr": cagr,
        }
    )
    df["sharpe_delta"] = df["rolling_sharpe"].diff()
    df["cagr_delta"] = df["rolling_cagr"].diff()
    if use_slope:
        df["sharpe_slope"] = (
            df["rolling_sharpe"] - df["rolling_sharpe"].shift(slope_days)
        ) / float(slope_days)
        df["cagr_slope"] = (
            df["rolling_cagr"] - df["rolling_cagr"].shift(slope_days)
        ) / float(slope_days)
        df["decreasing"] = (df["sharpe_slope"] <= -min_drop_sharpe) & (
            df["cagr_slope"] <= -min_drop_cagr
        )
    else:
        df["decreasing"] = (df["sharpe_delta"] <= -min_drop_sharpe) & (
            df["cagr_delta"] <= -min_drop_cagr
        )
    if start_date:
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_convert(None)
        df = df[df.index >= pd.to_datetime(start_date)]
    if df.empty:
        return df

    if getattr(prices.index, "tz", None) is not None:
        prices = prices.tz_convert(None)
    prices = prices.reindex(df.index)
    if vix_series is not None and not vix_series.empty:
        vix_aligned = vix_series.reindex(df.index).astype(float)
        vix_q1 = float(vix_series.quantile(0.33))
        vix_q2 = float(vix_series.quantile(0.66))
        df["vix"] = vix_aligned
        df["vix_bucket"] = np.where(
            vix_aligned <= vix_q1,
            "low",
            np.where(vix_aligned <= vix_q2, "mid", "high"),
        )
        threshold = float(vix_series.quantile(vix_quantile))
        df["vix_regime"] = np.where(vix_aligned >= threshold, "high", "low")
    else:
        df["vix_bucket"] = "unknown"
        df["vix_regime"] = "unknown"

    trend_ma = prices.rolling(trend_ma_days).mean()
    trend_threshold = trend_ma * (1.0 + trend_strength_pct)
    df["trend_state"] = np.where(prices >= trend_threshold, "on", "off")
    rolling_peak = prices.cummax()
    drawdown = (prices / rolling_peak) - 1.0
    df["drawdown"] = drawdown
    df["drawdown_state"] = np.where(drawdown <= -drawdown_threshold, "dd", "ok")

    returns = prices.pct_change(fill_method=None)
    realized_vol = returns.rolling(20).std() * np.sqrt(252.0)
    if vol_quantile_window > 0:
        vol_threshold = realized_vol.rolling(
            vol_quantile_window,
            min_periods=20,
        ).quantile(vol_quantile)
    else:
        vol_threshold = float(realized_vol.quantile(vol_quantile))
    df["realized_vol"] = realized_vol
    df["vol_state"] = np.where(realized_vol >= vol_threshold, "high", "low")

    df["regime_combo"] = (
        df["vix_bucket"].astype(str)
        + "|trend_"
        + df["trend_state"].astype(str)
        + "|dd_"
        + df["drawdown_state"].astype(str)
        + "|vol_"
        + df["vol_state"].astype(str)
    )
    if policy_fine_overrides:
        df["rs_bin"] = _build_rs_bin(df["rolling_sharpe"], policy_rs_quantiles)
    else:
        df["rs_bin"] = pd.NA

    df["policy_action"] = df.apply(_policy_for_combo, axis=1)
    if ticker in weak_tickers:
        rerisk_allowed = (df["trend_state"] == "on") & (df["vol_state"] == "low")
        df.loc[
            (df["policy_action"] == "re-risk (mean-revert)") & ~rerisk_allowed,
            "policy_action",
        ] = "hold"
        df.loc[df["policy_action"] == "cap leverage", "policy_action"] = "hold"
    if safeguard_negative:
        negative_mask = (df["rolling_sharpe"] < 0) & (df["rolling_cagr"] < 0)
        df.loc[negative_mask, "policy_action"] = "de-risk"

        if cooldown_days > 0:
            cooldown_remaining = 0
            cooldown_active = False
            for idx, row in df.iterrows():
                if negative_mask.loc[idx]:
                    cooldown_remaining = cooldown_days
                    if cooldown_dd_extra > 0 and row["drawdown_state"] == "dd":
                        cooldown_remaining += cooldown_dd_extra
                    cooldown_active = True
                if cooldown_active:
                    df.at[idx, "policy_action"] = "de-risk"
                    if cooldown_dd_extra > 0 and row["drawdown_state"] == "dd":
                        cooldown_remaining = max(
                            cooldown_remaining,
                            cooldown_days + cooldown_dd_extra,
                        )
                    if cooldown_remaining > 0:
                        cooldown_remaining -= 1
                    if cooldown_remaining == 0:
                        if cooldown_reentry_sharpe > 0:
                            if row["rolling_sharpe"] > cooldown_reentry_sharpe:
                                cooldown_active = False
                        else:
                            if row["rolling_sharpe"] > 0 or row["trend_state"] == "on":
                                cooldown_active = False

    if dd_hard_stop:
        df.loc[df["drawdown_state"] == "dd", "policy_action"] = "de-risk"

    if policy_fine_overrides:
        df = _apply_fine_policy_overrides(
            df,
            ticker=ticker,
            vol_tickers=vol_tickers,
            equity_only=policy_fine_overrides_equity_only,
            require_not_decreasing=policy_fine_overrides_require_not_decreasing,
            allowed_tickers=policy_fine_override_tickers,
            allowed_rs_bins=policy_fine_override_rs_bins,
        )

    if ticker not in weak_tickers and ticker not in vol_tickers:
        if equity_hold_boost > 1.0 and equity_hold_boost_regime:
            boost_mask = (df["regime_combo"] == equity_hold_boost_regime) & (
                df["policy_action"] == "hold"
            )
            df.loc[boost_mask, "policy_action"] = "hold_boost"
        if vol_floor > 0 and vol_floor_boost > 1.0:
            vol_mask = (
                (df["trend_state"] == "on")
                & (df["realized_vol"] <= vol_floor)
                & (df["policy_action"] == "hold")
            )
            df.loc[vol_mask, "policy_action"] = "vol_boost"

    df["fwd_5d"] = prices.shift(-5) / prices - 1.0
    df["fwd_10d"] = prices.shift(-10) / prices - 1.0

    def _action(row: pd.Series) -> str:
        if (
            row["sharpe_delta"] <= -warn_drop_sharpe
            or row["cagr_delta"] <= -warn_drop_cagr
        ):
            return "review exposure"
        return "monitor"

    df["action"] = df.apply(_action, axis=1)
    df.insert(0, "ticker", ticker)
    df.reset_index(inplace=True)
    if "index" in df.columns:
        df.rename(columns={"index": "date"}, inplace=True)
    elif "Date" in df.columns:
        df.rename(columns={"Date": "date"}, inplace=True)
    return df


def _build_decay_table(
    ticker: str,
    prices: pd.Series,
    window_days: int,
    min_drop_sharpe: float,
    min_drop_cagr: float,
    warn_drop_sharpe: float,
    warn_drop_cagr: float,
    start_date: str | None,
    use_slope: bool,
    slope_days: int,
    vix_series: pd.Series | None,
    vix_quantile: float,
    vol_quantile: float,
    vol_quantile_window: int,
    trend_ma_days: int,
    trend_strength_pct: float,
    drawdown_threshold: float,
    vol_tickers: set[str],
    weak_tickers: set[str],
    safeguard_negative: bool,
    cooldown_days: int,
    cooldown_dd_extra: int,
    cooldown_reentry_sharpe: float,
    equity_hold_boost: float,
    equity_hold_boost_regime: str,
    vol_floor: float,
    vol_floor_boost: float,
    dd_hard_stop: bool,
) -> pd.DataFrame:
    df = _build_base_table(
        ticker,
        prices,
        window_days,
        min_drop_sharpe,
        min_drop_cagr,
        warn_drop_sharpe,
        warn_drop_cagr,
        start_date,
        use_slope,
        slope_days,
        vix_series,
        vix_quantile,
        vol_quantile,
        vol_quantile_window,
        trend_ma_days,
        trend_strength_pct,
        drawdown_threshold,
        vol_tickers,
        weak_tickers,
        safeguard_negative,
        cooldown_days,
        cooldown_dd_extra,
        cooldown_reentry_sharpe,
        equity_hold_boost,
        equity_hold_boost_regime,
        vol_floor,
        vol_floor_boost,
        dd_hard_stop,
    )
    if df.empty:
        return df
    return df[df["decreasing"]].copy()


def _append_report(
    output_path: Path,
    config: DecayConfig,
    results: dict[str, pd.DataFrame],
    combo_results: dict[str, pd.DataFrame],
    overlay_results: dict[str, dict[str, dict[str, float]]],
    proxy_lines: list[str] | None = None,
) -> None:
    lines = []
    lines.append(f"## Daily Decay Checks: {config.run_label}")
    lines.append("")
    lines.append(
        "Rolling window: {window}d. Decreasing means Sharpe and CAGR both drop vs prior day.".format(
            window=config.window_days
        )
    )
    if config.use_slope:
        lines.append(
            "Slope mode: {days}d slope for Sharpe/CAGR.".format(days=config.slope_days)
        )
    lines.append(
        "Thresholds: sharpe_drop >= {sharpe}, cagr_drop >= {cagr}.".format(
            sharpe=config.min_drop_sharpe,
            cagr=config.min_drop_cagr,
        )
    )
    if config.start_date:
        lines.append(f"Start date filter: {config.start_date}+")
    if config.include_vix_regime:
        lines.append(f"VIX regime split: quantile >= {config.vix_quantile:.2f} = high")
    lines.append("")

    if proxy_lines:
        lines.extend(proxy_lines)

    for ticker, df in results.items():
        lines.append(f"### {ticker}")
        if df.empty:
            lines.append("- No decreasing dates found with current thresholds.")
            lines.append("")
            continue
        if config.use_slope:
            header = "| Date | Price | Rolling Sharpe | Rolling CAGR | Sharpe Slope | CAGR Slope |"
            if config.include_vix_regime:
                header += " VIX | Regime |"
            header += " Action | Policy |"
            lines.append(header)
            separator = "| --- | --- | --- | --- | --- | --- |"
            if config.include_vix_regime:
                separator += " --- | --- |"
            separator += " --- | --- |"
            lines.append(separator)
        else:
            header = "| Date | Price | Rolling Sharpe | Rolling CAGR | Sharpe Delta | CAGR Delta |"
            if config.include_vix_regime:
                header += " VIX | Regime |"
            header += " Action | Policy |"
            lines.append(header)
            separator = "| --- | --- | --- | --- | --- | --- |"
            if config.include_vix_regime:
                separator += " --- | --- |"
            separator += " --- | --- |"
            lines.append(separator)
        for _, row in df.iterrows():
            vix_info = ""
            if config.include_vix_regime:
                vix_info = " | {vix} | {regime}".format(
                    vix=f"{row.get('vix', np.nan):.2f}",
                    regime=row.get("vix_regime", "N/A"),
                )
            if config.use_slope:
                lines.append(
                    "| {date} | {price} | {sharpe} | {cagr} | {sd} | {cd}{vix} | {action} | {policy} |".format(
                        date=row["date"].strftime("%Y-%m-%d"),
                        price=f"{row['price']:.4f}",
                        sharpe=f"{row['rolling_sharpe']:.3f}",
                        cagr=f"{row['rolling_cagr']:.2%}",
                        sd=f"{row['sharpe_slope']:.3f}",
                        cd=f"{row['cagr_slope']:.2%}",
                        vix=vix_info,
                        action=row["action"],
                        policy=row.get("policy_action", ""),
                    )
                )
                continue
            lines.append(
                "| {date} | {price} | {sharpe} | {cagr} | {sd} | {cd}{vix} | {action} | {policy} |".format(
                    date=row["date"].strftime("%Y-%m-%d"),
                    price=f"{row['price']:.4f}",
                    sharpe=f"{row['rolling_sharpe']:.3f}",
                    cagr=f"{row['rolling_cagr']:.2%}",
                    sd=f"{row['sharpe_delta']:.3f}",
                    cd=f"{row['cagr_delta']:.2%}",
                    vix=vix_info,
                    action=row["action"],
                    policy=row.get("policy_action", ""),
                )
            )
        lines.append("")

    lines.append("### Regime Combo Summary")
    lines.append(
        "| Ticker | Regime Combo | Events | Avg Fwd 5D | Hit 5D | Avg Fwd 10D | Hit 10D |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for ticker, summary in combo_results.items():
        if summary.empty:
            continue
        for _, row in summary.iterrows():
            lines.append(
                "| {ticker} | {combo} | {events} | {avg5} | {hit5} | {avg10} | {hit10} |".format(
                    ticker=ticker,
                    combo=row["regime_combo"],
                    events=int(row["events"]),
                    avg5=f"{row['avg_fwd_5d']:.2%}",
                    hit5=f"{row['hit_5d']:.1%}",
                    avg10=f"{row['avg_fwd_10d']:.2%}",
                    hit10=f"{row['hit_10d']:.1%}",
                )
            )
    lines.append("")

    lines.append("### Policy Overlay Summary")
    lines.append("| Ticker | Sharpe | CAGR | Vol | Max DD |")
    lines.append("| --- | --- | --- | --- | --- |")
    for ticker, stats in overlay_results.items():
        for label in ("full", "in_sample", "oos"):
            if label not in stats:
                continue
            values = stats[label]
            lines.append(
                "| {ticker} ({label}) | {sharpe} | {cagr} | {vol} | {dd} |".format(
                    ticker=ticker,
                    label=label,
                    sharpe=f"{values.get('sharpe', float('nan')):.3f}",
                    cagr=f"{values.get('cagr', float('nan')):.2%}",
                    vol=f"{values.get('vol', float('nan')):.2%}",
                    dd=f"{values.get('max_dd', float('nan')):.2%}",
                )
            )
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        existing = output_path.read_text(encoding="utf-8")
        output_path.write_text(
            existing.rstrip() + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8"
        )
    else:
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_tickers(raw: str, default: list[str]) -> list[str]:
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


def _tickers_from_config(config: AppConfig, mode: str) -> list[str]:
    core = [config.ticker, config.growth_ticker]
    if mode == "core":
        return core
    if mode == "core_vol":
        return core + [config.vix_ticker, config.vvix_ticker]
    if mode == "all":
        return core + [
            config.vix_ticker,
            config.vvix_ticker,
            *config.reit_tickers.keys(),
        ]
    return core


def main() -> int:
    config = load_config()
    parser = argparse.ArgumentParser(
        description="Check per-date Sharpe/CAGR decay and write a report.",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Comma-separated tickers (defaults to config.ticker + growth_ticker).",
    )
    parser.add_argument(
        "--tickers-from-config",
        type=str,
        default="core",
        choices=["core", "core_vol", "all"],
        help="Ticker set from config: core, core_vol, or all.",
    )
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--min-drop-sharpe", type=float, default=0.0)
    parser.add_argument("--min-drop-cagr", type=float, default=0.0)
    parser.add_argument("--warn-drop-sharpe", type=float, default=0.1)
    parser.add_argument("--warn-drop-cagr", type=float, default=0.02)
    parser.add_argument("--start-date", type=str, default="")
    parser.add_argument(
        "--data-end-date",
        type=str,
        default="",
        help="Optional inclusive data cutoff date (YYYY-MM-DD) for deterministic runs.",
    )
    parser.add_argument("--use-slope", action="store_true")
    parser.add_argument("--slope-days", type=int, default=7)
    parser.add_argument("--include-vix-regime", action="store_true")
    parser.add_argument("--vix-quantile", type=float, default=0.75)
    parser.add_argument("--vol-quantile", type=float, default=0.75)
    parser.add_argument(
        "--vol-quantile-window",
        type=int,
        default=0,
        help="Rolling window for vol quantile (0 uses full history).",
    )
    parser.add_argument("--trend-ma-days", type=int, default=200)
    parser.add_argument(
        "--trend-strength-pct",
        type=float,
        default=0.0,
        help="Trend requires price >= MA * (1 + pct).",
    )
    parser.add_argument("--drawdown-threshold", type=float, default=0.1)
    parser.add_argument(
        "--run-label",
        type=str,
        default="run1_tight_weak_v8_dynvol",
    )
    parser.add_argument("--policy-min-events", type=int, default=50)
    parser.add_argument("--policy-rerisk", type=float, default=1.5)
    parser.add_argument("--policy-hold", type=float, default=1.0)
    parser.add_argument("--policy-cap", type=float, default=0.6)
    parser.add_argument("--policy-derisk", type=float, default=0.0)
    parser.add_argument("--policy-rerisk-vol", type=float, default=None)
    parser.add_argument("--policy-hold-vol", type=float, default=None)
    parser.add_argument("--policy-cap-vol", type=float, default=None)
    parser.add_argument("--policy-derisk-vol", type=float, default=None)
    parser.add_argument("--policy-rerisk-vol-trend-off", type=float, default=None)
    parser.add_argument("--policy-hold-vol-trend-off", type=float, default=None)
    parser.add_argument("--policy-cap-vol-trend-off", type=float, default=None)
    parser.add_argument("--policy-derisk-vol-trend-off", type=float, default=None)
    parser.add_argument("--policy-rerisk-vix-trend-off", type=float, default=None)
    parser.add_argument("--policy-hold-vix-trend-off", type=float, default=None)
    parser.add_argument("--policy-cap-vix-trend-off", type=float, default=None)
    parser.add_argument("--policy-derisk-vix-trend-off", type=float, default=None)
    parser.add_argument("--policy-rerisk-vvix-trend-off", type=float, default=None)
    parser.add_argument("--policy-hold-vvix-trend-off", type=float, default=None)
    parser.add_argument("--policy-cap-vvix-trend-off", type=float, default=None)
    parser.add_argument("--policy-derisk-vvix-trend-off", type=float, default=None)
    parser.add_argument(
        "--equity-hold-boost",
        type=float,
        default=1.0,
        help="Exposure multiplier for equity hold boosts.",
    )
    parser.add_argument(
        "--equity-hold-boost-regime",
        type=str,
        default="high|trend_off|dd_ok|vol_low",
        help="Regime combo to apply equity hold boost.",
    )
    parser.add_argument(
        "--vol-floor",
        type=float,
        default=0.0,
        help="If realized vol <= this, allow equity vol-boost in trend-on regimes.",
    )
    parser.add_argument(
        "--vol-floor-boost",
        type=float,
        default=1.0,
        help="Exposure multiplier for equity vol-boost.",
    )
    parser.add_argument("--oos-split-date", type=str, default="")
    parser.add_argument(
        "--vol-tickers",
        type=str,
        default="^VIX,^VVIX",
        help="Comma-separated tickers treated as vol indices for policy mapping.",
    )
    parser.add_argument(
        "--weak-tickers",
        type=str,
        default="DCRU.SI,M44U.SI,^VIX,^VVIX",
        help=(
            "Comma-separated tickers to restrict re-risk to low-vol trend-on; cap leverage -> hold."
        ),
    )
    parser.add_argument(
        "--policy-min-events-equity",
        type=int,
        default=0,
        help="Override min events for non-vol tickers (0 uses --policy-min-events).",
    )
    parser.add_argument(
        "--policy-min-events-vol",
        type=int,
        default=0,
        help="Override min events for vol tickers (0 uses --policy-min-events).",
    )
    parser.add_argument(
        "--safeguard-negative",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, force de-risk when rolling Sharpe and CAGR are negative.",
    )
    parser.add_argument(
        "--cooldown-days",
        type=int,
        default=0,
        help="Cooldown days after safeguard trigger before reopening exposure.",
    )
    parser.add_argument(
        "--cooldown-dd-extra",
        type=int,
        default=0,
        help="Extra cooldown days while drawdown_state=dd.",
    )
    parser.add_argument(
        "--dd-hard-stop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, force de-risk when drawdown_state=dd.",
    )
    parser.add_argument(
        "--cooldown-reentry-sharpe",
        type=float,
        default=0.0,
        help="Sharpe threshold required to exit cooldown (0 keeps old behavior).",
    )
    parser.add_argument(
        "--policy-fine-overrides",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable consensus fine-regime policy overrides (hold over de-risk in selected sub-regimes).",
    )
    parser.add_argument(
        "--policy-rs-quantiles",
        type=int,
        default=4,
        help="Quantile buckets for rolling_sharpe when fine overrides are enabled.",
    )
    parser.add_argument(
        "--policy-fine-overrides-equity-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When fine overrides are enabled, apply only to non-vol tickers.",
    )
    parser.add_argument(
        "--policy-fine-overrides-require-not-decreasing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When fine overrides are enabled, apply only when decreasing=False.",
    )
    parser.add_argument(
        "--policy-fine-override-tickers",
        type=str,
        default="",
        help="Optional comma-separated ticker whitelist for fine overrides (empty=all eligible tickers).",
    )
    parser.add_argument(
        "--policy-fine-override-rs-bins",
        type=str,
        default="rs_q3,rs_q4",
        help="Comma-separated rs_bin whitelist for fine overrides.",
    )
    parser.add_argument(
        "--export-regime-daily",
        action="store_true",
        help="Write a full daily regime CSV for each ticker.",
    )
    parser.add_argument(
        "--archive-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, archive the run outputs under reports/decay_runs/.",
    )
    parser.add_argument(
        "--archive-tag",
        type=str,
        default="",
        help="Optional archive tag (defaults to run label).",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default=f"reports/strategy_next_steps_{actual_time}-notes.md",
    )
    parser.add_argument(
        "--output-csv-dir",
        type=str,
        default="reports/selected_table_checks",
    )
    args = parser.parse_args()

    default_tickers = _tickers_from_config(config, args.tickers_from_config)
    tickers = _parse_tickers(args.tickers, default_tickers)

    decay_config = DecayConfig(
        tickers=tickers,
        window_days=args.window_days,
        min_drop_sharpe=args.min_drop_sharpe,
        min_drop_cagr=args.min_drop_cagr,
        warn_drop_sharpe=args.warn_drop_sharpe,
        warn_drop_cagr=args.warn_drop_cagr,
        start_date=args.start_date or None,
        data_end_date=args.data_end_date,
        use_slope=args.use_slope,
        slope_days=args.slope_days,
        include_vix_regime=args.include_vix_regime,
        vix_quantile=args.vix_quantile,
        vol_quantile=args.vol_quantile,
        vol_quantile_window=args.vol_quantile_window,
        trend_ma_days=args.trend_ma_days,
        trend_strength_pct=args.trend_strength_pct,
        drawdown_threshold=args.drawdown_threshold,
        run_label=args.run_label,
        policy_min_events=args.policy_min_events,
        policy_min_events_equity=(
            args.policy_min_events_equity
            if args.policy_min_events_equity > 0
            else args.policy_min_events
        ),
        policy_min_events_vol=(
            args.policy_min_events_vol
            if args.policy_min_events_vol > 0
            else args.policy_min_events
        ),
        policy_rerisk=args.policy_rerisk,
        policy_hold=args.policy_hold,
        policy_cap=args.policy_cap,
        policy_derisk=args.policy_derisk,
        policy_rerisk_vol=(
            args.policy_rerisk_vol
            if args.policy_rerisk_vol is not None
            else args.policy_rerisk
        ),
        policy_hold_vol=(
            args.policy_hold_vol
            if args.policy_hold_vol is not None
            else args.policy_hold
        ),
        policy_cap_vol=(
            args.policy_cap_vol if args.policy_cap_vol is not None else args.policy_cap
        ),
        policy_derisk_vol=(
            args.policy_derisk_vol
            if args.policy_derisk_vol is not None
            else args.policy_derisk
        ),
        policy_rerisk_vol_trend_off=(
            args.policy_rerisk_vol_trend_off
            if args.policy_rerisk_vol_trend_off is not None
            else (
                args.policy_rerisk_vol
                if args.policy_rerisk_vol is not None
                else args.policy_rerisk
            )
        ),
        policy_hold_vol_trend_off=(
            args.policy_hold_vol_trend_off
            if args.policy_hold_vol_trend_off is not None
            else (
                args.policy_hold_vol
                if args.policy_hold_vol is not None
                else args.policy_hold
            )
        ),
        policy_cap_vol_trend_off=(
            args.policy_cap_vol_trend_off
            if args.policy_cap_vol_trend_off is not None
            else (
                args.policy_cap_vol
                if args.policy_cap_vol is not None
                else args.policy_cap
            )
        ),
        policy_derisk_vol_trend_off=(
            args.policy_derisk_vol_trend_off
            if args.policy_derisk_vol_trend_off is not None
            else (
                args.policy_derisk_vol
                if args.policy_derisk_vol is not None
                else args.policy_derisk
            )
        ),
        policy_rerisk_vix_trend_off=args.policy_rerisk_vix_trend_off,
        policy_hold_vix_trend_off=args.policy_hold_vix_trend_off,
        policy_cap_vix_trend_off=args.policy_cap_vix_trend_off,
        policy_derisk_vix_trend_off=args.policy_derisk_vix_trend_off,
        policy_rerisk_vvix_trend_off=args.policy_rerisk_vvix_trend_off,
        policy_hold_vvix_trend_off=args.policy_hold_vvix_trend_off,
        policy_cap_vvix_trend_off=args.policy_cap_vvix_trend_off,
        policy_derisk_vvix_trend_off=args.policy_derisk_vvix_trend_off,
        equity_hold_boost=args.equity_hold_boost,
        equity_hold_boost_regime=args.equity_hold_boost_regime,
        vol_floor=args.vol_floor,
        vol_floor_boost=args.vol_floor_boost,
        oos_split_date=args.oos_split_date or None,
        vol_tickers=set(_parse_tickers(args.vol_tickers, [])),
        weak_tickers=set(_parse_tickers(args.weak_tickers, [])),
        safeguard_negative=args.safeguard_negative,
        cooldown_days=args.cooldown_days,
        cooldown_dd_extra=args.cooldown_dd_extra,
        cooldown_reentry_sharpe=args.cooldown_reentry_sharpe,
        policy_fine_overrides=args.policy_fine_overrides,
        policy_rs_quantiles=args.policy_rs_quantiles,
        policy_fine_overrides_equity_only=args.policy_fine_overrides_equity_only,
        policy_fine_overrides_require_not_decreasing=args.policy_fine_overrides_require_not_decreasing,
        policy_fine_override_tickers=set(_parse_tickers(args.policy_fine_override_tickers, [])),
        policy_fine_override_rs_bins=set(_parse_tickers(args.policy_fine_override_rs_bins, ["rs_q3", "rs_q4"])),
        dd_hard_stop=args.dd_hard_stop,
        archive_run=args.archive_run,
        archive_tag=args.archive_tag.strip() or None,
        output_md=Path(args.output_md),
        output_csv_dir=Path(args.output_csv_dir),
    )

    cache_dir = Path(config.cache_dir)
    max_age_hours = config.market_cache_max_age_hours

    results: dict[str, pd.DataFrame] = {}
    combo_results: dict[str, pd.DataFrame] = {}
    overlay_results: dict[str, dict[str, dict[str, float]]] = {}
    vix_series = None
    if decay_config.include_vix_regime:
        vix_series = fetch_close_series(
            config.vix_ticker,
            period="max",
            cache_dir=cache_dir,
            max_age_hours=max_age_hours,
        )
        if vix_series is not None:
            vix_series = vix_series.sort_index()
            vix_series = _apply_data_end_date(vix_series, decay_config.data_end_date)
    label = decay_config.run_label.replace(" ", "_").lower()
    export_regime_daily = args.export_regime_daily or label in V9_PROXY_LABELS
    for ticker in tickers:
        series = fetch_close_series(
            ticker,
            period="max",
            cache_dir=cache_dir,
            max_age_hours=max_age_hours,
        )
        if series is None or series.empty:
            print(f"No data for {ticker}.")
            results[ticker] = pd.DataFrame()
            continue
        series = series.sort_index()
        series = _apply_data_end_date(series, decay_config.data_end_date)
        if series.empty:
            print(f"No data for {ticker} after data-end-date filter.")
            results[ticker] = pd.DataFrame()
            continue
        base_df = _build_base_table(
            ticker,
            series,
            decay_config.window_days,
            decay_config.min_drop_sharpe,
            decay_config.min_drop_cagr,
            decay_config.warn_drop_sharpe,
            decay_config.warn_drop_cagr,
            decay_config.start_date,
            decay_config.use_slope,
            decay_config.slope_days,
            vix_series,
            decay_config.vix_quantile,
            decay_config.vol_quantile,
            decay_config.vol_quantile_window,
            decay_config.trend_ma_days,
            decay_config.trend_strength_pct,
            decay_config.drawdown_threshold,
            decay_config.vol_tickers,
            decay_config.weak_tickers,
            decay_config.safeguard_negative,
            decay_config.cooldown_days,
            decay_config.cooldown_dd_extra,
            decay_config.cooldown_reentry_sharpe,
            decay_config.policy_fine_overrides,
            decay_config.policy_rs_quantiles,
            decay_config.policy_fine_overrides_equity_only,
            decay_config.policy_fine_overrides_require_not_decreasing,
            decay_config.policy_fine_override_tickers,
            decay_config.policy_fine_override_rs_bins,
            decay_config.equity_hold_boost,
            decay_config.equity_hold_boost_regime,
            decay_config.vol_floor,
            decay_config.vol_floor_boost,
            decay_config.dd_hard_stop,
        )
        decay_config.output_csv_dir.mkdir(parents=True, exist_ok=True)

        if base_df.empty:
            results[ticker] = pd.DataFrame()
            continue

        if export_regime_daily:
            daily_path = (
                decay_config.output_csv_dir / f"regime_daily_{ticker}_{label}.csv"
            )
            base_df.to_csv(daily_path, index=False)

        df = base_df[base_df["decreasing"]].copy()
        results[ticker] = df
        if df.empty:
            combo_results[ticker] = pd.DataFrame()
        else:
            summary = (
                df.groupby("regime_combo")
                .agg(
                    events=("regime_combo", "size"),
                    avg_fwd_5d=("fwd_5d", "mean"),
                    hit_5d=("fwd_5d", lambda x: float((x > 0).mean())),
                    avg_fwd_10d=("fwd_10d", "mean"),
                    hit_10d=("fwd_10d", lambda x: float((x > 0).mean())),
                )
                .reset_index()
                .sort_values("events", ascending=False)
            )
            combo_results[ticker] = summary
            min_events = (
                decay_config.policy_min_events_vol
                if ticker in decay_config.vol_tickers
                else decay_config.policy_min_events_equity
            )
            df = _apply_policy_thresholds(
                df,
                summary,
                min_events,
            )
        csv_path = decay_config.output_csv_dir / f"decay_checks_{ticker}_{label}.csv"
        df.to_csv(csv_path, index=False)
        policy_path = decay_config.output_csv_dir / f"decay_policy_{ticker}_{label}.csv"
        policy_cols = [
            "date",
            "ticker",
            "regime_combo",
            "rs_bin",
            "trend_state",
            "policy_action",
            "rolling_sharpe",
            "rolling_cagr",
            "fwd_5d",
            "fwd_10d",
        ]
        df[policy_cols].to_csv(policy_path, index=False)
        overlay_results[ticker] = _simulate_overlay_split(
            series,
            df,
            decay_config,
            ticker,
        )

    if results:
        latest_rows = []
        for ticker, df in results.items():
            if df.empty:
                continue
            latest = df.sort_values("date").iloc[-1]
            latest_rows.append(
                {
                    "date": latest.get("date"),
                    "ticker": latest.get("ticker"),
                    "regime_combo": latest.get("regime_combo"),
                    "policy_action": latest.get("policy_action"),
                    "rolling_sharpe": latest.get("rolling_sharpe"),
                    "rolling_cagr": latest.get("rolling_cagr"),
                }
            )
        if latest_rows:
            daily_path = decay_config.output_csv_dir / f"decay_daily_action_{label}.csv"
            pd.DataFrame(latest_rows).to_csv(daily_path, index=False)

    _append_report(
        decay_config.output_md,
        decay_config,
        results,
        combo_results,
        overlay_results,
        proxy_lines=_build_proxy_section(
            decay_config,
            V9_PROXY_LABELS,
            hold_exposure=PROXY_HOLD_EXPOSURE,
        ),
    )
    summary_path = _write_overlay_summary(
        overlay_results,
        decay_config.output_csv_dir,
        label,
    )
    comparison_path = _write_overlay_comparison(
        overlay_results,
        decay_config.output_csv_dir,
        label,
        decay_config,
        cache_dir,
        max_age_hours,
    )
    if decay_config.archive_run:
        archive_label = decay_config.archive_tag or label
        _archive_run_outputs(
            decay_config.output_csv_dir,
            decay_config.output_md,
            archive_label,
            summary_path,
            comparison_path,
        )
    print(
        "Decay checks complete for {tickers} at {timestamp}.".format(
            tickers=", ".join(tickers),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        )
    )
    return 0


if __name__ == "__main__":
    actual_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    raise SystemExit(main())
