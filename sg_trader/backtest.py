from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from .config import AppConfig


@dataclass
class BacktestResult:
    stats: dict[str, float]
    geometric_mean: float
    exposure: float
    avg_holding_days: float
    sample_trades: list[dict[str, str]]


def _fetch_series(ticker: str) -> pd.Series:
    series = yf.Ticker(ticker).history(period="max")["Close"]
    return series.dropna()


def _align_series(*series: pd.Series) -> tuple[pd.Series, ...]:
    joined = pd.concat(series, axis=1, join="inner").dropna()
    return tuple(joined.iloc[:, idx] for idx in range(joined.shape[1]))


def run_backtest(config: AppConfig) -> BacktestResult:
    try:
        import vectorbt as vbt
    except ImportError as exc:
        raise RuntimeError(
            "vectorbt is required for backtesting. Install it with pip install vectorbt."
        ) from exc

    spx = _fetch_series(config.ticker)
    vix = _fetch_series(config.vix_ticker)
    vvix = _fetch_series(config.vvix_ticker)
    spx, vix, vvix = _align_series(spx, vix, vvix)

    log_returns = np.log(spx / spx.shift(1))
    rv = log_returns.rolling(window=30).std() * np.sqrt(252) * 100
    rv = rv.dropna()
    vix = vix.loc[rv.index]
    vvix = vvix.loc[rv.index]
    spx = spx.loc[rv.index]

    spread = vix - rv
    entries = (spread > config.alpha_spread_threshold) & (
        vvix < config.vvix_safe_threshold
    )
    exits = ~entries

    freq = "1D"
    pf = vbt.Portfolio.from_signals(
        spx,
        entries=entries,
        exits=exits,
        freq=freq,
        direction="longonly",
        size=1.0,
        size_type="percent",
        init_cash=1.0,
    )

    returns = pf.returns().dropna()
    if returns.empty:
        geometric_mean = 0.0
    else:
        cumulative = (1 + returns).prod()
        geometric_mean = cumulative ** (252 / len(returns)) - 1

    entry_count = int(entries.sum())
    trade_count = int(pf.trades.count())
    trade_records = pf.trades.records_readable
    in_position = False
    exposure_flags = []
    for entry, exit_signal in zip(entries.to_numpy(), exits.to_numpy()):
        if entry:
            in_position = True
        if exit_signal and in_position:
            in_position = False
        exposure_flags.append(in_position)
    exposure = float(sum(exposure_flags)) / len(exposure_flags) * 100
    if trade_records.empty:
        avg_holding_days = 0.0
    else:
        entry_times = pd.to_datetime(trade_records["Entry Timestamp"], errors="coerce")
        exit_times = pd.to_datetime(trade_records["Exit Timestamp"], errors="coerce")
        holding_days = (exit_times - entry_times).dt.total_seconds() / 86400
        avg_holding_days = float(holding_days.dropna().mean())
    sample_trades = []
    for _, row in trade_records.head(5).iterrows():
        try:
            pnl = float(row.get("PnL", 0.0))
        except (TypeError, ValueError):
            pnl = 0.0
        sample_trades.append(
            {
                "entry": str(row.get("Entry Timestamp", "")),
                "exit": str(row.get("Exit Timestamp", "")),
                "pnl": f"{pnl:.4f}",
            }
        )

    value_series = pf.value().dropna()
    if value_series.empty:
        computed_total_return = 0.0
        computed_annual_return = 0.0
        computed_annual_vol = 0.0
    else:
        computed_total_return = float(value_series.iloc[-1] / value_series.iloc[0] - 1)
        years = len(value_series) / 252
        if years <= 0:
            computed_annual_return = 0.0
        else:
            computed_annual_return = (1 + computed_total_return) ** (1 / years) - 1
        returns_series = value_series.pct_change().dropna()
        computed_annual_vol = float(returns_series.std() * np.sqrt(252))

    return BacktestResult(
        stats={
            "total_return": computed_total_return,
            "annualized_return": computed_annual_return,
            "annualized_volatility": computed_annual_vol,
            "entry_count": float(entry_count),
            "trade_count": float(trade_count),
            "exposure_pct": exposure,
            "avg_holding_days": avg_holding_days,
        },
        geometric_mean=float(geometric_mean),
        exposure=exposure,
        avg_holding_days=avg_holding_days,
        sample_trades=sample_trades,
    )
