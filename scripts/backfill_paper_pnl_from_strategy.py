import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from sg_trader.config import load_config


TRADING_CLOSE_TIME = "16:00:00"
TRADING_DAYS = 252
VOL_WINDOW_DAYS = 60
MOMENTUM_WINDOW_DAYS = 252
MOMENTUM_WINDOW_6M = 126
VIX_HYST_V2_HIGH = 32
VIX_HYST_V2_LOW = 20
VIX_HYST_CONFIRM_DAYS = 2
CRASH_VIX_THRESHOLD = 28
VOL_TARGET = 0.10
VOL_TARGET_WINDOW = 60
VOL_TARGET_MAX_LEVERAGE = 1.2
DRAWDOWN_WINDOW = 60
DRAWDOWN_THRESHOLD = 0.10
VVIX_THRESHOLD = 120
VIX_CRASH_THRESHOLD = 30
VIX_RECOVERY_THRESHOLD = 20
SMA_FAST_WINDOW = 50
SMA_SLOW_WINDOW = 200
RATE_SLOPE_WINDOW = 20
ZSCORE_WINDOW = 60
ZSCORE_THRESHOLD = -1.5
ZSCORE_CUT = 0.3
ZSCORE_TREND_ON_MIN_LEVERAGE = 0.9
MOMENTUM_DYNAMIC_VOL_WINDOW = 60
MOMENTUM_DYNAMIC_MEDIAN_WINDOW = 252
CARRY_TREND_LOOKBACK = 126
CARRY_TREND_VIX_MAX = 25
VOL_RISK_OFF_VIX = 30
VOL_RISK_OFF_VVIX = 120
REGIME_BLEND_RISK_OFF_VIX = 29
REGIME_BLEND_NON_TREND_IEF_SHARE = 0.5
RATE_REGIME_VIX_MAX = 30
BREADTH_RATIO_WINDOW = 50
MEAN_REVERSION_LOOKBACK = 5
ZSCORE_TREND_OFF_VIX = 32
ZSCORE_TREND_OFF_MAX_LEVERAGE = 0.5
DYNAMIC_IEF_CUT = 0.5
DYNAMIC_IEF_MIN_LEVERAGE = 0.9
DYNAMIC_IEF_VIX_CAP = 32
DYNAMIC_IEF_BOOST = 0.3
DYNAMIC_IEF_VIX_THRESHOLD = 32
DYNAMIC_IEF_LAG_DAYS = 0
MEAN_REVERSION_THRESHOLD = -0.02
DEFENSIVE_VIX_MAX = 25
MACRO_VOL_VIX_MAX = 28
MACRO_VOL_VVIX_MAX = 115
REVERSION_FAST_LOOKBACK = 3
REVERSION_FAST_THRESHOLD = -0.015
DRAWDOWN_LOOKBACK = 60
DRAWDOWN_CUTOFF = 0.12
CASH_TILT_VIX_MAX = 28
DEF_CARRY_V2_VIX_MAX = 22
VOL_BREAKOUT_LOOKBACK = 20
VOL_BREAKOUT_VOL_WINDOW = 20
VOL_BREAKOUT_VOL_THRESHOLD = 0.20
RISK_PARITY_CRASH_VIX = 30
DRAWDOWN_TREND_CUTOFF = 0.10
MACRO_CARRY_VIX_MAX = 25
MACRO_CARRY_VVIX_MAX = 120


@dataclass
class Position:
    quantity: float
    avg_price: float


def _timestamp_for_date(day: pd.Timestamp) -> str:
    return f"{day.strftime('%Y-%m-%d')} {TRADING_CLOSE_TIME}"


def _module_for_ticker(ticker: str, growth_ticker: str) -> str:
    upper = ticker.upper()
    if upper == "SPX_PUT":
        return "shield"
    if upper == "S-REIT_BASKET" or upper == "CLR.SI":
        return "fortress"
    if upper == "IEF":
        return "risk_off"
    if upper in {"^SPX", "SPX", "SPY"}:
        return "alpha"
    if upper == growth_ticker.upper() or upper == "QQQ":
        return "growth"
    return "unattributed"


def _download_prices(tickers: list[str], start: str, end: str) -> dict[str, pd.Series]:
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
    )
    if data.empty:
        raise RuntimeError("No price data returned from Yahoo Finance.")
    prices: dict[str, pd.Series] = {}
    for ticker in tickers:
        if ticker not in data.columns.get_level_values(0):
            raise RuntimeError(f"Missing ticker data for {ticker}.")
        series = data[ticker]["Close"].dropna()
        if series.empty:
            raise RuntimeError(f"No close prices available for {ticker}.")
        prices[ticker] = series
    return prices


def _common_dates(prices: dict[str, pd.Series], symbols: list[str]) -> pd.DatetimeIndex:
    common = None
    for symbol in symbols:
        series = prices.get(symbol)
        if series is None:
            raise RuntimeError(f"Missing price series for {symbol}.")
        index = series.index
        common = index if common is None else common.intersection(index)
    if common is None or common.empty:
        raise RuntimeError("No common dates found across required symbols.")
    return common


def _month_key(day: pd.Timestamp) -> str:
    return day.strftime("%Y-%m")


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def _returns(series: pd.Series) -> pd.Series:
    return series.pct_change().dropna()


def _append_entry(entries: list[dict[str, Any]], entry: dict[str, Any]) -> None:
    entries.append(entry)


def _pnl_entry(
    day: pd.Timestamp,
    ticker: str,
    position: Position,
    mark_price: float,
    module: str,
    tags: list[str],
    mark_symbol: str | None = None,
) -> dict[str, Any]:
    unrealized = (mark_price - position.avg_price) * position.quantity
    details: dict[str, Any] = {
        "quantity": position.quantity,
        "avg_price": position.avg_price,
        "mark_price": mark_price,
        "unrealized_pnl": unrealized,
        "module": module,
    }
    if mark_symbol:
        details["mark_symbol"] = mark_symbol
    return {
        "timestamp": _timestamp_for_date(day),
        "category": "Execution",
        "ticker": ticker,
        "action": "PAPER_PNL",
        "rationale": "Backfill strategy PnL snapshot.",
        "tags": tags,
        "details": details,
        "entry_type": "execution",
        "schema_version": 2,
    }


def _realized_entry(
    day: pd.Timestamp,
    ticker: str,
    realized_pnl: float,
    module: str,
    tags: list[str],
) -> dict[str, Any]:
    return {
        "timestamp": _timestamp_for_date(day),
        "category": "Execution",
        "ticker": ticker,
        "action": "PAPER_REALIZED",
        "rationale": "Backfill strategy realized PnL.",
        "tags": tags,
        "details": {
            "realized_pnl": realized_pnl,
            "module": module,
        },
        "entry_type": "execution",
        "schema_version": 2,
    }


def _buy_hold(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    symbols = ["SPY", "QQQ", "CLR.SI"]
    dates = _common_dates(prices, symbols)
    first_day = dates[0]
    per_symbol = initial_capital / len(symbols)
    positions: dict[str, Position] = {}
    for symbol in symbols:
        price = float(prices[symbol].loc[first_day])
        quantity = per_symbol / price
        positions[symbol] = Position(quantity=quantity, avg_price=price)

    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "buy_hold"]
    for day in dates:
        for symbol, position in positions.items():
            module = _module_for_ticker(symbol, growth_ticker)
            mark_price = float(prices[symbol].loc[day])
            _append_entry(
                entries,
                _pnl_entry(day, symbol, position, mark_price, module, tags),
            )
    return entries


def _momentum_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    spy = prices["SPY"]
    spx = prices["^SPX"]
    dates = _common_dates(prices, ["SPY", "^SPX"])
    sma200 = _sma(spy, 200)
    slope_window = 20
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "momentum"]

    position_symbol = None
    position = None
    cash = initial_capital

    for day in dates:
        if pd.isna(sma200.loc[day]):
            continue
        slope = sma200.loc[day] - sma200.shift(slope_window).loc[day]
        desired = "SPY" if slope > 0 else "SPX_PUT"

        if position_symbol is None:
            price = float(spy.loc[day])
            quantity = cash / price
            position_symbol = "SPY"
            position = Position(quantity=quantity, avg_price=price)
            cash = 0.0

        if desired != position_symbol:
            current_price = float(spx.loc[day]) if position_symbol == "SPX_PUT" else float(spy.loc[day])
            realized = (current_price - position.avg_price) * position.quantity
            _append_entry(
                entries,
                _realized_entry(
                    day,
                    position_symbol,
                    realized,
                    _module_for_ticker(position_symbol, growth_ticker),
                    tags,
                ),
            )
            cash = current_price * position.quantity
            new_price = float(spx.loc[day]) if desired == "SPX_PUT" else float(spy.loc[day])
            quantity = cash / new_price if new_price != 0 else 0.0
            position_symbol = desired
            position = Position(quantity=quantity, avg_price=new_price)
            cash = 0.0

        mark_price = float(spx.loc[day]) if position_symbol == "SPX_PUT" else float(spy.loc[day])
        mark_symbol = "^SPX" if position_symbol == "SPX_PUT" else None
        _append_entry(
            entries,
            _pnl_entry(
                day,
                position_symbol,
                position,
                mark_price,
                _module_for_ticker(position_symbol, growth_ticker),
                tags,
                mark_symbol=mark_symbol,
            ),
        )
    return entries


def _trend_vix_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    spy = prices["SPY"]
    spx = prices["^SPX"]
    vix = prices["^VIX"]
    dates = _common_dates(prices, ["SPY", "^SPX", "^VIX"])
    sma200 = _sma(spy, 200)
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "trend_vix"]

    position_symbol = None
    position = None
    cash = initial_capital

    for day in dates:
        if pd.isna(sma200.loc[day]):
            continue
        desired = "SPY" if (float(spy.loc[day]) > float(sma200.loc[day]) and float(vix.loc[day]) < 20) else "SPX_PUT"

        if position_symbol is None:
            price = float(spy.loc[day])
            quantity = cash / price
            position_symbol = "SPY"
            position = Position(quantity=quantity, avg_price=price)
            cash = 0.0

        if desired != position_symbol:
            current_price = float(spx.loc[day]) if position_symbol == "SPX_PUT" else float(spy.loc[day])
            realized = (current_price - position.avg_price) * position.quantity
            _append_entry(
                entries,
                _realized_entry(
                    day,
                    position_symbol,
                    realized,
                    _module_for_ticker(position_symbol, growth_ticker),
                    tags,
                ),
            )
            cash = current_price * position.quantity
            new_price = float(spx.loc[day]) if desired == "SPX_PUT" else float(spy.loc[day])
            quantity = cash / new_price if new_price != 0 else 0.0
            position_symbol = desired
            position = Position(quantity=quantity, avg_price=new_price)
            cash = 0.0

        mark_price = float(spx.loc[day]) if position_symbol == "SPX_PUT" else float(spy.loc[day])
        mark_symbol = "^SPX" if position_symbol == "SPX_PUT" else None
        _append_entry(
            entries,
            _pnl_entry(
                day,
                position_symbol,
                position,
                mark_price,
                _module_for_ticker(position_symbol, growth_ticker),
                tags,
                mark_symbol=mark_symbol,
            ),
        )
    return entries


def _risk_parity_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    symbols = ["SPY", "QQQ", "CLR.SI"]
    dates = _common_dates(prices, symbols)
    returns = {symbol: _returns(prices[symbol]) for symbol in symbols}
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "risk_parity"]

    positions: dict[str, Position] = {}
    cash = initial_capital
    current_month = None

    for day in dates:
        month_key = _month_key(day)
        if current_month != month_key:
            current_month = month_key
            realized_total = 0.0
            for symbol, position in positions.items():
                price = float(prices[symbol].loc[day])
                realized = (price - position.avg_price) * position.quantity
                realized_total += realized
                if position.quantity != 0:
                    _append_entry(
                        entries,
                        _realized_entry(
                            day,
                            symbol,
                            realized,
                            _module_for_ticker(symbol, growth_ticker),
                            tags,
                        ),
                    )
            cash += realized_total
            positions = {}

            vols = {}
            for symbol in symbols:
                window = returns[symbol].loc[:day].tail(VOL_WINDOW_DAYS)
                if len(window) < VOL_WINDOW_DAYS or window.std() == 0:
                    vols[symbol] = None
                else:
                    vols[symbol] = float(window.std())
            inv_vols = {
                symbol: (1.0 / vol) if vol else 0.0 for symbol, vol in vols.items()
            }
            total_inv = sum(inv_vols.values())
            weights = {symbol: (inv_vols[symbol] / total_inv) if total_inv > 0 else 0.0 for symbol in symbols}

            for symbol, weight in weights.items():
                if weight == 0.0:
                    continue
                price = float(prices[symbol].loc[day])
                quantity = (cash * weight) / price
                positions[symbol] = Position(quantity=quantity, avg_price=price)

        for symbol, position in positions.items():
            module = _module_for_ticker(symbol, growth_ticker)
            mark_price = float(prices[symbol].loc[day])
            _append_entry(
                entries,
                _pnl_entry(day, symbol, position, mark_price, module, tags),
            )
    return entries


def _dual_momentum_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    symbols = ["SPY", "QQQ", "CLR.SI"]
    dates = _common_dates(prices, symbols)
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "dual_momentum"]

    position_symbol = None
    position = None
    cash = initial_capital
    current_month = None

    for day in dates:
        month_key = _month_key(day)
        if current_month != month_key:
            current_month = month_key
            returns = {}
            for symbol in symbols:
                series = prices[symbol].loc[:day]
                if len(series) <= MOMENTUM_WINDOW_DAYS:
                    returns[symbol] = None
                else:
                    returns[symbol] = (series.iloc[-1] / series.iloc[-MOMENTUM_WINDOW_DAYS]) - 1.0
            valid = {k: v for k, v in returns.items() if v is not None}
            if not valid:
                continue
            desired = max(valid, key=valid.get)

            if position_symbol != desired:
                if position_symbol is not None:
                    price = float(prices[position_symbol].loc[day])
                    realized = (price - position.avg_price) * position.quantity
                    _append_entry(
                        entries,
                        _realized_entry(
                            day,
                            position_symbol,
                            realized,
                            _module_for_ticker(position_symbol, growth_ticker),
                            tags,
                        ),
                    )
                    cash = price * position.quantity
                price = float(prices[desired].loc[day])
                quantity = cash / price if price != 0 else 0.0
                position_symbol = desired
                position = Position(quantity=quantity, avg_price=price)
                cash = 0.0

        if position_symbol is None:
            continue
        module = _module_for_ticker(position_symbol, growth_ticker)
        mark_price = float(prices[position_symbol].loc[day])
        _append_entry(
            entries,
            _pnl_entry(day, position_symbol, position, mark_price, module, tags),
        )
    return entries


def _combined_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    half_capital = initial_capital * 0.5
    momentum_entries = _momentum_strategy(prices, half_capital, growth_ticker)
    risk_entries = _risk_parity_strategy(prices, half_capital, growth_ticker)

    combined = momentum_entries + risk_entries
    combined.sort(key=lambda entry: entry.get("timestamp", ""))
    for entry in combined:
        entry["rationale"] = "Backfill strategy PnL snapshot (combined)."
        tags = entry.get("tags", [])
        if "combined" not in tags:
            entry["tags"] = tags + ["combined"]
    return combined


def _vix_hysteresis_v2_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    vix = prices["^VIX"]
    spy = prices["SPY"]
    spx = prices["^SPX"]
    dates = _common_dates(prices, ["^VIX", "SPY", "^SPX"])
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "vix_hysteresis_v2"]

    position_symbol = None
    position = None
    cash = initial_capital
    high_count = 0
    low_count = 0

    for day in dates:
        vix_level = float(vix.loc[day])
        high_count = high_count + 1 if vix_level > VIX_HYST_V2_HIGH else 0
        low_count = low_count + 1 if vix_level < VIX_HYST_V2_LOW else 0

        if position_symbol is None:
            price = float(spy.loc[day])
            quantity = cash / price
            position_symbol = "SPY"
            position = Position(quantity=quantity, avg_price=price)
            cash = 0.0

        desired = position_symbol
        if position_symbol == "SPY" and high_count >= VIX_HYST_CONFIRM_DAYS:
            desired = "SPX_PUT"
        elif position_symbol == "SPX_PUT" and low_count >= VIX_HYST_CONFIRM_DAYS:
            desired = "SPY"

        if desired != position_symbol:
            current_price = float(spx.loc[day]) if position_symbol == "SPX_PUT" else float(spy.loc[day])
            realized = (current_price - position.avg_price) * position.quantity
            _append_entry(
                entries,
                _realized_entry(
                    day,
                    position_symbol,
                    realized,
                    _module_for_ticker(position_symbol, growth_ticker),
                    tags,
                ),
            )
            cash = current_price * position.quantity
            new_price = float(spx.loc[day]) if desired == "SPX_PUT" else float(spy.loc[day])
            quantity = cash / new_price if new_price != 0 else 0.0
            position_symbol = desired
            position = Position(quantity=quantity, avg_price=new_price)
            cash = 0.0

        mark_price = float(spx.loc[day]) if position_symbol == "SPX_PUT" else float(spy.loc[day])
        mark_symbol = "^SPX" if position_symbol == "SPX_PUT" else None
        _append_entry(
            entries,
            _pnl_entry(
                day,
                position_symbol,
                position,
                mark_price,
                _module_for_ticker(position_symbol, growth_ticker),
                tags,
                mark_symbol=mark_symbol,
            ),
        )
    return entries


def _trend_breadth_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    spy = prices["SPY"]
    spx = prices["^SPX"]
    rsp = prices["RSP"]
    dates = _common_dates(prices, ["SPY", "^SPX", "RSP"])
    sma200 = _sma(spy, 200)
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "trend_breadth"]

    position_symbol = None
    position = None
    cash = initial_capital

    for day in dates:
        if pd.isna(sma200.loc[day]):
            continue
        breadth_ok = float(rsp.loc[day]) / float(spy.loc[day]) > 1.0
        desired = "SPY" if (float(spy.loc[day]) > float(sma200.loc[day]) and breadth_ok) else "SPX_PUT"

        if position_symbol is None:
            price = float(spy.loc[day])
            quantity = cash / price
            position_symbol = "SPY"
            position = Position(quantity=quantity, avg_price=price)
            cash = 0.0

        if desired != position_symbol:
            current_price = float(spx.loc[day]) if position_symbol == "SPX_PUT" else float(spy.loc[day])
            realized = (current_price - position.avg_price) * position.quantity
            _append_entry(
                entries,
                _realized_entry(
                    day,
                    position_symbol,
                    realized,
                    _module_for_ticker(position_symbol, growth_ticker),
                    tags,
                ),
            )
            cash = current_price * position.quantity
            new_price = float(spx.loc[day]) if desired == "SPX_PUT" else float(spy.loc[day])
            quantity = cash / new_price if new_price != 0 else 0.0
            position_symbol = desired
            position = Position(quantity=quantity, avg_price=new_price)
            cash = 0.0

        mark_price = float(spx.loc[day]) if position_symbol == "SPX_PUT" else float(spy.loc[day])
        mark_symbol = "^SPX" if position_symbol == "SPX_PUT" else None
        _append_entry(
            entries,
            _pnl_entry(
                day,
                position_symbol,
                position,
                mark_price,
                _module_for_ticker(position_symbol, growth_ticker),
                tags,
                mark_symbol=mark_symbol,
            ),
        )
    return entries


def _carry_trend_hybrid_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    spy = prices["SPY"]
    vix = prices["^VIX"]
    dates = _common_dates(prices, ["SPY", "^VIX", "QQQ", "CLR.SI", "IEF"])
    sma200 = _sma(spy, SMA_SLOW_WINDOW)
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "carry_trend_hybrid"]

    position_symbol = None
    position = None
    cash = initial_capital
    current_month = None

    for day in dates:
        month_key = _month_key(day)
        if current_month != month_key:
            current_month = month_key
            if pd.isna(sma200.loc[day]):
                continue
            in_trend = float(spy.loc[day]) > float(sma200.loc[day])
            vix_ok = float(vix.loc[day]) <= CARRY_TREND_VIX_MAX
            if in_trend and vix_ok:
                returns_by_symbol = {}
                for symbol in ["QQQ", "CLR.SI"]:
                    series = prices[symbol].loc[:day]
                    if len(series) <= CARRY_TREND_LOOKBACK:
                        returns_by_symbol[symbol] = None
                    else:
                        returns_by_symbol[symbol] = (
                            series.iloc[-1] / series.iloc[-CARRY_TREND_LOOKBACK]
                        ) - 1.0
                valid = {k: v for k, v in returns_by_symbol.items() if v is not None}
                desired = max(valid, key=valid.get) if valid else "SPY"
            else:
                desired = "IEF"

            if position_symbol != desired:
                if position_symbol is not None:
                    price = float(prices[position_symbol].loc[day])
                    realized = (price - position.avg_price) * position.quantity
                    _append_entry(
                        entries,
                        _realized_entry(
                            day,
                            position_symbol,
                            realized,
                            _module_for_ticker(position_symbol, growth_ticker),
                            tags,
                        ),
                    )
                    cash = price * position.quantity
                price = float(prices[desired].loc[day])
                quantity = cash / price if price != 0 else 0.0
                position_symbol = desired
                position = Position(quantity=quantity, avg_price=price)
                cash = 0.0

        if position_symbol is None:
            continue
        mark_price = float(prices[position_symbol].loc[day])
        _append_entry(
            entries,
            _pnl_entry(
                day,
                position_symbol,
                position,
                mark_price,
                _module_for_ticker(position_symbol, growth_ticker),
                tags,
            ),
        )
    return entries


def _vol_risk_control_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    spy = prices["SPY"]
    ief = prices["IEF"]
    vix = prices["^VIX"]
    vvix = prices["^VVIX"]
    dates = _common_dates(prices, ["SPY", "IEF", "^VIX", "^VVIX"])
    returns = _returns(spy)
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "vol_risk_control"]

    position_symbol = None
    position = None
    cash = initial_capital

    for day in dates:
        spy_price = float(spy.loc[day])
        ief_price = float(ief.loc[day])
        equity = cash
        if position_symbol == "SPY":
            equity += position.quantity * spy_price
        elif position_symbol == "IEF":
            equity += position.quantity * ief_price

        risk_off = (
            float(vix.loc[day]) >= VOL_RISK_OFF_VIX
            or float(vvix.loc[day]) >= VOL_RISK_OFF_VVIX
        )

        if risk_off:
            if position_symbol != "IEF":
                if position_symbol == "SPY":
                    realized = (spy_price - position.avg_price) * position.quantity
                    _append_entry(
                        entries,
                        _realized_entry(
                            day,
                            position_symbol,
                            realized,
                            _module_for_ticker(position_symbol, growth_ticker),
                            tags,
                        ),
                    )
                elif position_symbol == "IEF":
                    realized = (ief_price - position.avg_price) * position.quantity
                    _append_entry(
                        entries,
                        _realized_entry(
                            day,
                            position_symbol,
                            realized,
                            _module_for_ticker(position_symbol, growth_ticker),
                            tags,
                        ),
                    )
                cash = equity
                quantity = cash / ief_price if ief_price != 0 else 0.0
                position_symbol = "IEF"
                position = Position(quantity=quantity, avg_price=ief_price)
                cash = cash - (quantity * ief_price)

            if position_symbol is not None:
                _append_entry(
                    entries,
                    _pnl_entry(
                        day,
                        position_symbol,
                        position,
                        ief_price,
                        _module_for_ticker(position_symbol, growth_ticker),
                        tags,
                    ),
                )
            continue

        window = returns.loc[:day].tail(VOL_TARGET_WINDOW)
        if len(window) < VOL_TARGET_WINDOW or window.std() == 0:
            target_leverage = 0.0
        else:
            realized_vol = float(window.std()) * math.sqrt(TRADING_DAYS)
            target_leverage = min(VOL_TARGET_MAX_LEVERAGE, VOL_TARGET / realized_vol)

        if target_leverage == 0.0:
            if position_symbol == "SPY":
                realized = (spy_price - position.avg_price) * position.quantity
                _append_entry(
                    entries,
                    _realized_entry(
                        day,
                        position_symbol,
                        realized,
                        _module_for_ticker(position_symbol, growth_ticker),
                        tags,
                    ),
                )
            elif position_symbol == "IEF":
                realized = (ief_price - position.avg_price) * position.quantity
                _append_entry(
                    entries,
                    _realized_entry(
                        day,
                        position_symbol,
                        realized,
                        _module_for_ticker(position_symbol, growth_ticker),
                        tags,
                    ),
                )
            cash = equity
            position_symbol = None
            position = None
            continue

        target_qty = (equity * target_leverage) / spy_price if spy_price != 0 else 0.0
        if position_symbol == "SPY":
            realized = (spy_price - position.avg_price) * position.quantity
            _append_entry(
                entries,
                _realized_entry(
                    day,
                    position_symbol,
                    realized,
                    _module_for_ticker(position_symbol, growth_ticker),
                    tags,
                ),
            )
        elif position_symbol == "IEF":
            realized = (ief_price - position.avg_price) * position.quantity
            _append_entry(
                entries,
                _realized_entry(
                    day,
                    position_symbol,
                    realized,
                    _module_for_ticker(position_symbol, growth_ticker),
                    tags,
                ),
            )

        position_symbol = "SPY"
        position = Position(quantity=target_qty, avg_price=spy_price)
        cash = equity - (target_qty * spy_price)
        _append_entry(
            entries,
            _pnl_entry(
                day,
                position_symbol,
                position,
                spy_price,
                _module_for_ticker(position_symbol, growth_ticker),
                tags,
            ),
        )
    return entries


def _macro_rate_regime_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    spy = prices["SPY"]
    qqq = prices["QQQ"]
    ief = prices["IEF"]
    tnx = prices["^TNX"]
    vix = prices["^VIX"]
    dates = _common_dates(prices, ["SPY", "QQQ", "IEF", "^TNX", "^VIX"])
    sma200 = _sma(spy, SMA_SLOW_WINDOW)
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "macro_rate_regime"]

    position_symbol = None
    position = None
    cash = initial_capital
    current_month = None

    for day in dates:
        month_key = _month_key(day)
        if current_month != month_key:
            current_month = month_key
            if pd.isna(sma200.loc[day]):
                continue
            slope = tnx.loc[day] - tnx.shift(RATE_SLOPE_WINDOW).loc[day]
            if pd.isna(slope):
                continue
            in_trend = float(spy.loc[day]) > float(sma200.loc[day])
            vix_ok = float(vix.loc[day]) <= RATE_REGIME_VIX_MAX
            if in_trend and vix_ok:
                desired = "QQQ" if slope < 0 else "SPY"
            else:
                desired = "IEF"

            if position_symbol != desired:
                if position_symbol is not None:
                    price = float(prices[position_symbol].loc[day])
                    realized = (price - position.avg_price) * position.quantity
                    _append_entry(
                        entries,
                        _realized_entry(
                            day,
                            position_symbol,
                            realized,
                            _module_for_ticker(position_symbol, growth_ticker),
                            tags,
                        ),
                    )
                    cash = price * position.quantity
                price = float(prices[desired].loc[day])
                quantity = cash / price if price != 0 else 0.0
                position_symbol = desired
                position = Position(quantity=quantity, avg_price=price)
                cash = 0.0

        if position_symbol is None:
            continue
        mark_price = float(prices[position_symbol].loc[day])
        _append_entry(
            entries,
            _pnl_entry(
                day,
                position_symbol,
                position,
                mark_price,
                _module_for_ticker(position_symbol, growth_ticker),
                tags,
            ),
        )
    return entries


def _breadth_quality_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    spy = prices["SPY"]
    qqq = prices["QQQ"]
    rsp = prices["RSP"]
    ief = prices["IEF"]
    dates = _common_dates(prices, ["SPY", "QQQ", "RSP", "IEF"])
    ratio = rsp / spy
    ratio_sma = _sma(ratio, BREADTH_RATIO_WINDOW)
    sma200 = _sma(spy, SMA_SLOW_WINDOW)
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "breadth_quality"]

    position_symbol = None
    position = None
    cash = initial_capital
    current_month = None

    for day in dates:
        month_key = _month_key(day)
        if current_month != month_key:
            current_month = month_key
            if pd.isna(sma200.loc[day]) or pd.isna(ratio_sma.loc[day]):
                continue
            in_trend = float(spy.loc[day]) > float(sma200.loc[day])
            breadth_ok = float(ratio.loc[day]) > float(ratio_sma.loc[day])
            if in_trend and breadth_ok:
                desired = "QQQ"
            elif in_trend:
                desired = "SPY"
            else:
                desired = "IEF"

            if position_symbol != desired:
                if position_symbol is not None:
                    price = float(prices[position_symbol].loc[day])
                    realized = (price - position.avg_price) * position.quantity
                    _append_entry(
                        entries,
                        _realized_entry(
                            day,
                            position_symbol,
                            realized,
                            _module_for_ticker(position_symbol, growth_ticker),
                            tags,
                        ),
                    )
                    cash = price * position.quantity
                price = float(prices[desired].loc[day])
                quantity = cash / price if price != 0 else 0.0
                position_symbol = desired
                position = Position(quantity=quantity, avg_price=price)
                cash = 0.0

        if position_symbol is None:
            continue
        mark_price = float(prices[position_symbol].loc[day])
        _append_entry(
            entries,
            _pnl_entry(
                day,
                position_symbol,
                position,
                mark_price,
                _module_for_ticker(position_symbol, growth_ticker),
                tags,
            ),
        )
    return entries


def _defensive_carry_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    spy = prices["SPY"]
    qqq = prices["QQQ"]
    ief = prices["IEF"]
    vix = prices["^VIX"]
    dates = _common_dates(prices, ["SPY", "QQQ", "IEF", "^VIX"])
    sma200 = _sma(spy, SMA_SLOW_WINDOW)
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "defensive_carry"]

    position_symbol = None
    position = None
    cash = initial_capital
    current_month = None

    for day in dates:
        month_key = _month_key(day)
        if current_month != month_key:
            current_month = month_key
            if pd.isna(sma200.loc[day]):
                continue
            in_trend = float(spy.loc[day]) > float(sma200.loc[day])
            risk_ok = float(vix.loc[day]) <= DEFENSIVE_VIX_MAX
            if in_trend and risk_ok:
                returns_by_symbol = {}
                for symbol in ["SPY", "QQQ"]:
                    series = prices[symbol].loc[:day]
                    if len(series) <= MOMENTUM_WINDOW_6M:
                        returns_by_symbol[symbol] = None
                    else:
                        returns_by_symbol[symbol] = (
                            series.iloc[-1] / series.iloc[-MOMENTUM_WINDOW_6M]
                        ) - 1.0
                valid = {k: v for k, v in returns_by_symbol.items() if v is not None}
                desired = max(valid, key=valid.get) if valid else "SPY"
            else:
                desired = "IEF"

            if position_symbol != desired:
                if position_symbol is not None:
                    price = float(prices[position_symbol].loc[day])
                    realized = (price - position.avg_price) * position.quantity
                    _append_entry(
                        entries,
                        _realized_entry(
                            day,
                            position_symbol,
                            realized,
                            _module_for_ticker(position_symbol, growth_ticker),
                            tags,
                        ),
                    )
                    cash = price * position.quantity
                price = float(prices[desired].loc[day])
                quantity = cash / price if price != 0 else 0.0
                position_symbol = desired
                position = Position(quantity=quantity, avg_price=price)
                cash = 0.0

        if position_symbol is None:
            continue
        mark_price = float(prices[position_symbol].loc[day])
        _append_entry(
            entries,
            _pnl_entry(
                day,
                position_symbol,
                position,
                mark_price,
                _module_for_ticker(position_symbol, growth_ticker),
                tags,
            ),
        )
    return entries


def _macro_vol_regime_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    spy = prices["SPY"]
    qqq = prices["QQQ"]
    ief = prices["IEF"]
    tnx = prices["^TNX"]
    vix = prices["^VIX"]
    vvix = prices["^VVIX"]
    dates = _common_dates(prices, ["SPY", "QQQ", "IEF", "^TNX", "^VIX", "^VVIX"])
    sma200 = _sma(spy, SMA_SLOW_WINDOW)
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "macro_vol_regime"]

    position_symbol = None
    position = None
    cash = initial_capital
    current_month = None

    for day in dates:
        month_key = _month_key(day)
        if current_month != month_key:
            current_month = month_key
            if pd.isna(sma200.loc[day]):
                continue
            slope = tnx.loc[day] - tnx.shift(RATE_SLOPE_WINDOW).loc[day]
            if pd.isna(slope):
                continue
            risk_off = (
                float(vix.loc[day]) >= MACRO_VOL_VIX_MAX
                or float(vvix.loc[day]) >= MACRO_VOL_VVIX_MAX
            )
            in_trend = float(spy.loc[day]) > float(sma200.loc[day])
            if risk_off or not in_trend:
                desired = "IEF"
            else:
                desired = "QQQ" if slope < 0 else "SPY"

            if position_symbol != desired:
                if position_symbol is not None:
                    price = float(prices[position_symbol].loc[day])
                    realized = (price - position.avg_price) * position.quantity
                    _append_entry(
                        entries,
                        _realized_entry(
                            day,
                            position_symbol,
                            realized,
                            _module_for_ticker(position_symbol, growth_ticker),
                            tags,
                        ),
                    )
                    cash = price * position.quantity
                price = float(prices[desired].loc[day])
                quantity = cash / price if price != 0 else 0.0
                position_symbol = desired
                position = Position(quantity=quantity, avg_price=price)
                cash = 0.0

        if position_symbol is None:
            continue
        mark_price = float(prices[position_symbol].loc[day])
        _append_entry(
            entries,
            _pnl_entry(
                day,
                position_symbol,
                position,
                mark_price,
                _module_for_ticker(position_symbol, growth_ticker),
                tags,
            ),
        )
    return entries


def _trend_breadth_hybrid_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    spy = prices["SPY"]
    qqq = prices["QQQ"]
    rsp = prices["RSP"]
    ief = prices["IEF"]
    vix = prices["^VIX"]
    dates = _common_dates(prices, ["SPY", "QQQ", "RSP", "IEF", "^VIX"])
    ratio = rsp / spy
    ratio_sma = _sma(ratio, BREADTH_RATIO_WINDOW)
    sma200 = _sma(spy, SMA_SLOW_WINDOW)
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "trend_breadth_hybrid"]

    position_symbol = None
    position = None
    cash = initial_capital
    current_month = None

    for day in dates:
        month_key = _month_key(day)
        if current_month != month_key:
            current_month = month_key
            if pd.isna(sma200.loc[day]) or pd.isna(ratio_sma.loc[day]):
                continue
            in_trend = float(spy.loc[day]) > float(sma200.loc[day])
            breadth_ok = float(ratio.loc[day]) > float(ratio_sma.loc[day])
            risk_ok = float(vix.loc[day]) <= DEFENSIVE_VIX_MAX
            if in_trend and breadth_ok and risk_ok:
                desired = "QQQ"
            elif in_trend and risk_ok:
                desired = "SPY"
            else:
                desired = "IEF"

            if position_symbol != desired:
                if position_symbol is not None:
                    price = float(prices[position_symbol].loc[day])
                    realized = (price - position.avg_price) * position.quantity
                    _append_entry(
                        entries,
                        _realized_entry(
                            day,
                            position_symbol,
                            realized,
                            _module_for_ticker(position_symbol, growth_ticker),
                            tags,
                        ),
                    )
                    cash = price * position.quantity
                price = float(prices[desired].loc[day])
                quantity = cash / price if price != 0 else 0.0
                position_symbol = desired
                position = Position(quantity=quantity, avg_price=price)
                cash = 0.0

        if position_symbol is None:
            continue
        mark_price = float(prices[position_symbol].loc[day])
        _append_entry(
            entries,
            _pnl_entry(
                day,
                position_symbol,
                position,
                mark_price,
                _module_for_ticker(position_symbol, growth_ticker),
                tags,
            ),
        )
    return entries


def _short_term_reversion_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    spy = prices["SPY"]
    ief = prices["IEF"]
    vix = prices["^VIX"]
    dates = _common_dates(prices, ["SPY", "IEF", "^VIX"])
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "short_term_reversion"]

    position_symbol = None
    position = None
    cash = initial_capital

    for day in dates:
        lookback_price = spy.shift(REVERSION_FAST_LOOKBACK).loc[day]
        if pd.isna(lookback_price):
            continue
        return_lookback = (float(spy.loc[day]) / float(lookback_price)) - 1.0
        vix_ok = float(vix.loc[day]) < MACRO_VOL_VIX_MAX
        desired = "SPY" if (return_lookback <= REVERSION_FAST_THRESHOLD and vix_ok) else "IEF"

        if position_symbol is None:
            price = float(prices[desired].loc[day])
            quantity = cash / price if price != 0 else 0.0
            position_symbol = desired
            position = Position(quantity=quantity, avg_price=price)
            cash = 0.0

        if desired != position_symbol:
            current_price = float(prices[position_symbol].loc[day])
            realized = (current_price - position.avg_price) * position.quantity
            _append_entry(
                entries,
                _realized_entry(
                    day,
                    position_symbol,
                    realized,
                    _module_for_ticker(position_symbol, growth_ticker),
                    tags,
                ),
            )
            cash = current_price * position.quantity
            new_price = float(prices[desired].loc[day])
            quantity = cash / new_price if new_price != 0 else 0.0
            position_symbol = desired
            position = Position(quantity=quantity, avg_price=new_price)
            cash = 0.0

        mark_price = float(prices[position_symbol].loc[day])
        _append_entry(
            entries,
            _pnl_entry(
                day,
                position_symbol,
                position,
                mark_price,
                _module_for_ticker(position_symbol, growth_ticker),
                tags,
            ),
        )
    return entries


def _drawdown_control_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    spy = prices["SPY"]
    ief = prices["IEF"]
    dates = _common_dates(prices, ["SPY", "IEF"])
    sma200 = _sma(spy, SMA_SLOW_WINDOW)
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "drawdown_control"]

    position_symbol = None
    position = None
    cash = initial_capital
    current_month = None

    for day in dates:
        month_key = _month_key(day)
        if current_month != month_key:
            current_month = month_key
            if pd.isna(sma200.loc[day]):
                continue
            window = spy.loc[:day].tail(DRAWDOWN_LOOKBACK)
            if window.empty:
                continue
            peak = float(window.max())
            drawdown = (float(spy.loc[day]) / peak) - 1.0 if peak != 0 else 0.0
            in_trend = float(spy.loc[day]) > float(sma200.loc[day])
            desired = "SPY" if (drawdown > -DRAWDOWN_CUTOFF and in_trend) else "IEF"

            if position_symbol != desired:
                if position_symbol is not None:
                    price = float(prices[position_symbol].loc[day])
                    realized = (price - position.avg_price) * position.quantity
                    _append_entry(
                        entries,
                        _realized_entry(
                            day,
                            position_symbol,
                            realized,
                            _module_for_ticker(position_symbol, growth_ticker),
                            tags,
                        ),
                    )
                    cash = price * position.quantity
                price = float(prices[desired].loc[day])
                quantity = cash / price if price != 0 else 0.0
                position_symbol = desired
                position = Position(quantity=quantity, avg_price=price)
                cash = 0.0

        if position_symbol is None:
            continue
        mark_price = float(prices[position_symbol].loc[day])
        _append_entry(
            entries,
            _pnl_entry(
                day,
                position_symbol,
                position,
                mark_price,
                _module_for_ticker(position_symbol, growth_ticker),
                tags,
            ),
        )
    return entries


def _cash_tilt_regime_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    spy = prices["SPY"]
    ief = prices["IEF"]
    vix = prices["^VIX"]
    dates = _common_dates(prices, ["SPY", "IEF", "^VIX"])
    sma200 = _sma(spy, SMA_SLOW_WINDOW)
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "cash_tilt_regime"]

    position_symbol = None
    position = None
    cash = initial_capital
    current_month = None

    for day in dates:
        month_key = _month_key(day)
        if current_month != month_key:
            current_month = month_key
            if pd.isna(sma200.loc[day]):
                continue
            risk_off = float(vix.loc[day]) >= CASH_TILT_VIX_MAX
            in_trend = float(spy.loc[day]) > float(sma200.loc[day])
            desired = "SPY" if (in_trend and not risk_off) else "IEF"

            if position_symbol != desired:
                if position_symbol is not None:
                    price = float(prices[position_symbol].loc[day])
                    realized = (price - position.avg_price) * position.quantity
                    _append_entry(
                        entries,
                        _realized_entry(
                            day,
                            position_symbol,
                            realized,
                            _module_for_ticker(position_symbol, growth_ticker),
                            tags,
                        ),
                    )
                    cash = price * position.quantity
                price = float(prices[desired].loc[day])
                quantity = cash / price if price != 0 else 0.0
                position_symbol = desired
                position = Position(quantity=quantity, avg_price=price)
                cash = 0.0

        if position_symbol is None:
            continue
        mark_price = float(prices[position_symbol].loc[day])
        _append_entry(
            entries,
            _pnl_entry(
                day,
                position_symbol,
                position,
                mark_price,
                _module_for_ticker(position_symbol, growth_ticker),
                tags,
            ),
        )
    return entries


def _defensive_carry_v2_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    spy = prices["SPY"]
    qqq = prices["QQQ"]
    ief = prices["IEF"]
    vix = prices["^VIX"]
    tnx = prices["^TNX"]
    dates = _common_dates(prices, ["SPY", "QQQ", "IEF", "^VIX", "^TNX"])
    sma200 = _sma(spy, SMA_SLOW_WINDOW)
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "defensive_carry_v2"]

    position_symbol = None
    position = None
    cash = initial_capital
    current_month = None

    for day in dates:
        month_key = _month_key(day)
        if current_month != month_key:
            current_month = month_key
            if pd.isna(sma200.loc[day]):
                continue
            slope = tnx.loc[day] - tnx.shift(RATE_SLOPE_WINDOW).loc[day]
            if pd.isna(slope):
                continue
            in_trend = float(spy.loc[day]) > float(sma200.loc[day])
            risk_ok = float(vix.loc[day]) <= DEF_CARRY_V2_VIX_MAX
            if in_trend and risk_ok and float(slope) <= 0:
                returns_by_symbol = {}
                for symbol in ["SPY", "QQQ"]:
                    series = prices[symbol].loc[:day]
                    if len(series) <= MOMENTUM_WINDOW_6M:
                        returns_by_symbol[symbol] = None
                    else:
                        returns_by_symbol[symbol] = (
                            series.iloc[-1] / series.iloc[-MOMENTUM_WINDOW_6M]
                        ) - 1.0
                valid = {k: v for k, v in returns_by_symbol.items() if v is not None}
                desired = max(valid, key=valid.get) if valid else "SPY"
            else:
                desired = "IEF"

            if position_symbol != desired:
                if position_symbol is not None:
                    price = float(prices[position_symbol].loc[day])
                    realized = (price - position.avg_price) * position.quantity
                    _append_entry(
                        entries,
                        _realized_entry(
                            day,
                            position_symbol,
                            realized,
                            _module_for_ticker(position_symbol, growth_ticker),
                            tags,
                        ),
                    )
                    cash = price * position.quantity
                price = float(prices[desired].loc[day])
                quantity = cash / price if price != 0 else 0.0
                position_symbol = desired
                position = Position(quantity=quantity, avg_price=price)
                cash = 0.0

        if position_symbol is None:
            continue
        mark_price = float(prices[position_symbol].loc[day])
        _append_entry(
            entries,
            _pnl_entry(
                day,
                position_symbol,
                position,
                mark_price,
                _module_for_ticker(position_symbol, growth_ticker),
                tags,
            ),
        )
    return entries


def _vol_breakout_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    spy = prices["SPY"]
    ief = prices["IEF"]
    dates = _common_dates(prices, ["SPY", "IEF"])
    returns = _returns(spy)
    sma200 = _sma(spy, SMA_SLOW_WINDOW)
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "vol_breakout"]

    position_symbol = None
    position = None
    cash = initial_capital

    for day in dates:
        breakout_window = spy.loc[:day].tail(VOL_BREAKOUT_LOOKBACK + 1)
        if len(breakout_window) <= VOL_BREAKOUT_LOOKBACK:
            continue
        prior_high = float(breakout_window.iloc[:-1].max())
        in_trend = not pd.isna(sma200.loc[day]) and float(spy.loc[day]) > float(sma200.loc[day])

        vol_window = returns.loc[:day].tail(VOL_BREAKOUT_VOL_WINDOW)
        if len(vol_window) < VOL_BREAKOUT_VOL_WINDOW or vol_window.std() == 0:
            vol_ok = False
        else:
            realized_vol = float(vol_window.std()) * math.sqrt(TRADING_DAYS)
            vol_ok = realized_vol >= VOL_BREAKOUT_VOL_THRESHOLD

        desired = "SPY" if (float(spy.loc[day]) > prior_high and in_trend and vol_ok) else "IEF"

        if position_symbol is None:
            price = float(prices[desired].loc[day])
            quantity = cash / price if price != 0 else 0.0
            position_symbol = desired
            position = Position(quantity=quantity, avg_price=price)
            cash = 0.0

        if desired != position_symbol:
            current_price = float(prices[position_symbol].loc[day])
            realized = (current_price - position.avg_price) * position.quantity
            _append_entry(
                entries,
                _realized_entry(
                    day,
                    position_symbol,
                    realized,
                    _module_for_ticker(position_symbol, growth_ticker),
                    tags,
                ),
            )
            cash = current_price * position.quantity
            new_price = float(prices[desired].loc[day])
            quantity = cash / new_price if new_price != 0 else 0.0
            position_symbol = desired
            position = Position(quantity=quantity, avg_price=new_price)
            cash = 0.0

        mark_price = float(prices[position_symbol].loc[day])
        _append_entry(
            entries,
            _pnl_entry(
                day,
                position_symbol,
                position,
                mark_price,
                _module_for_ticker(position_symbol, growth_ticker),
                tags,
            ),
        )
    return entries


def _risk_parity_crash_guard_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    symbols = ["SPY", "QQQ", "CLR.SI"]
    dates = _common_dates(prices, symbols + ["^VIX", "IEF"])
    spy = prices["SPY"]
    vix = prices["^VIX"]
    ief = prices["IEF"]
    sma200 = _sma(spy, SMA_SLOW_WINDOW)
    returns = {symbol: _returns(prices[symbol]) for symbol in symbols}
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "risk_parity_crash_guard"]

    positions: dict[str, Position] = {}
    cash = initial_capital
    current_month = None

    for day in dates:
        month_key = _month_key(day)
        if current_month != month_key:
            current_month = month_key
            realized_total = 0.0
            for symbol, position in positions.items():
                price = float(prices[symbol].loc[day])
                realized = (price - position.avg_price) * position.quantity
                realized_total += realized
                if position.quantity != 0:
                    _append_entry(
                        entries,
                        _realized_entry(
                            day,
                            symbol,
                            realized,
                            _module_for_ticker(symbol, growth_ticker),
                            tags,
                        ),
                    )
            cash += realized_total
            positions = {}

            risk_off = float(vix.loc[day]) >= RISK_PARITY_CRASH_VIX
            if not pd.isna(sma200.loc[day]) and float(spy.loc[day]) < float(sma200.loc[day]):
                risk_off = True

            if risk_off:
                price = float(ief.loc[day])
                quantity = cash / price if price != 0 else 0.0
                positions["IEF"] = Position(quantity=quantity, avg_price=price)
            else:
                vols = {}
                for symbol in symbols:
                    window = returns[symbol].loc[:day].tail(VOL_WINDOW_DAYS)
                    if len(window) < VOL_WINDOW_DAYS or window.std() == 0:
                        vols[symbol] = None
                    else:
                        vols[symbol] = float(window.std())
                inv_vols = {symbol: (1.0 / vol) if vol else 0.0 for symbol, vol in vols.items()}
                total_inv = sum(inv_vols.values())
                weights = {symbol: (inv_vols[symbol] / total_inv) if total_inv > 0 else 0.0 for symbol in symbols}
                for symbol, weight in weights.items():
                    if weight == 0.0:
                        continue
                    price = float(prices[symbol].loc[day])
                    quantity = (cash * weight) / price
                    positions[symbol] = Position(quantity=quantity, avg_price=price)

        for symbol, position in positions.items():
            module = _module_for_ticker(symbol, growth_ticker)
            mark_price = float(prices[symbol].loc[day])
            _append_entry(
                entries,
                _pnl_entry(day, symbol, position, mark_price, module, tags),
            )
    return entries


def _trend_vol_target_drawdown_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    spy = prices["SPY"]
    ief = prices["IEF"]
    dates = _common_dates(prices, ["SPY", "IEF"])
    sma200 = _sma(spy, SMA_SLOW_WINDOW)
    returns = _returns(spy)
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "trend_vol_drawdown"]

    position_symbol = None
    position = None
    cash = initial_capital

    for day in dates:
        window = spy.loc[:day].tail(DRAWDOWN_LOOKBACK)
        if window.empty or pd.isna(sma200.loc[day]):
            continue
        peak = float(window.max())
        drawdown = (float(spy.loc[day]) / peak) - 1.0 if peak != 0 else 0.0
        in_trend = float(spy.loc[day]) > float(sma200.loc[day])

        vol_window = returns.loc[:day].tail(VOL_TARGET_WINDOW)
        if len(vol_window) < VOL_TARGET_WINDOW or vol_window.std() == 0:
            target_leverage = 0.0
        else:
            realized_vol = float(vol_window.std()) * math.sqrt(TRADING_DAYS)
            target_leverage = min(VOL_TARGET_MAX_LEVERAGE, VOL_TARGET / realized_vol)

        if not in_trend or drawdown < -DRAWDOWN_TREND_CUTOFF:
            desired = "IEF"
            target_leverage = 1.0
        else:
            desired = "SPY"

        spy_price = float(spy.loc[day])
        ief_price = float(ief.loc[day])
        equity = cash
        if position_symbol == "SPY":
            equity += position.quantity * spy_price
        elif position_symbol == "IEF":
            equity += position.quantity * ief_price

        if desired == "IEF":
            if position_symbol != "IEF":
                if position_symbol is not None:
                    price = spy_price if position_symbol == "SPY" else ief_price
                    realized = (price - position.avg_price) * position.quantity
                    _append_entry(
                        entries,
                        _realized_entry(
                            day,
                            position_symbol,
                            realized,
                            _module_for_ticker(position_symbol, growth_ticker),
                            tags,
                        ),
                    )
                cash = equity
                quantity = cash / ief_price if ief_price != 0 else 0.0
                position_symbol = "IEF"
                position = Position(quantity=quantity, avg_price=ief_price)
                cash = cash - (quantity * ief_price)

            _append_entry(
                entries,
                _pnl_entry(
                    day,
                    position_symbol,
                    position,
                    ief_price,
                    _module_for_ticker(position_symbol, growth_ticker),
                    tags,
                ),
            )
            continue

        if target_leverage == 0.0:
            continue

        target_qty = (equity * target_leverage) / spy_price if spy_price != 0 else 0.0
        if position_symbol == "SPY":
            realized = (spy_price - position.avg_price) * position.quantity
            _append_entry(
                entries,
                _realized_entry(
                    day,
                    position_symbol,
                    realized,
                    _module_for_ticker(position_symbol, growth_ticker),
                    tags,
                ),
            )
        elif position_symbol == "IEF":
            realized = (ief_price - position.avg_price) * position.quantity
            _append_entry(
                entries,
                _realized_entry(
                    day,
                    position_symbol,
                    realized,
                    _module_for_ticker(position_symbol, growth_ticker),
                    tags,
                ),
            )

        position_symbol = "SPY"
        position = Position(quantity=target_qty, avg_price=spy_price)
        cash = equity - (target_qty * spy_price)
        _append_entry(
            entries,
            _pnl_entry(
                day,
                position_symbol,
                position,
                spy_price,
                _module_for_ticker(position_symbol, growth_ticker),
                tags,
            ),
        )
    return entries


def _carry_macro_regime_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    spy = prices["SPY"]
    qqq = prices["QQQ"]
    ief = prices["IEF"]
    tnx = prices["^TNX"]
    vix = prices["^VIX"]
    vvix = prices["^VVIX"]
    dates = _common_dates(prices, ["SPY", "QQQ", "IEF", "^TNX", "^VIX", "^VVIX"])
    sma200 = _sma(spy, SMA_SLOW_WINDOW)
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "carry_macro_regime"]

    position_symbol = None
    position = None
    cash = initial_capital
    current_month = None

    for day in dates:
        month_key = _month_key(day)
        if current_month != month_key:
            current_month = month_key
            if pd.isna(sma200.loc[day]):
                continue
            slope = tnx.loc[day] - tnx.shift(RATE_SLOPE_WINDOW).loc[day]
            if pd.isna(slope):
                continue
            risk_off = (
                float(vix.loc[day]) >= MACRO_CARRY_VIX_MAX
                or float(vvix.loc[day]) >= MACRO_CARRY_VVIX_MAX
            )
            in_trend = float(spy.loc[day]) > float(sma200.loc[day])
            if risk_off or not in_trend:
                desired = "IEF"
            else:
                desired = "QQQ" if float(slope) <= 0 else "SPY"

            if position_symbol != desired:
                if position_symbol is not None:
                    price = float(prices[position_symbol].loc[day])
                    realized = (price - position.avg_price) * position.quantity
                    _append_entry(
                        entries,
                        _realized_entry(
                            day,
                            position_symbol,
                            realized,
                            _module_for_ticker(position_symbol, growth_ticker),
                            tags,
                        ),
                    )
                    cash = price * position.quantity
                price = float(prices[desired].loc[day])
                quantity = cash / price if price != 0 else 0.0
                position_symbol = desired
                position = Position(quantity=quantity, avg_price=price)
                cash = 0.0

        if position_symbol is None:
            continue
        mark_price = float(prices[position_symbol].loc[day])
        _append_entry(
            entries,
            _pnl_entry(
                day,
                position_symbol,
                position,
                mark_price,
                _module_for_ticker(position_symbol, growth_ticker),
                tags,
            ),
        )
    return entries


def _mean_reversion_vol_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    spy = prices["SPY"]
    ief = prices["IEF"]
    vix = prices["^VIX"]
    dates = _common_dates(prices, ["SPY", "IEF", "^VIX"])
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "mean_reversion_vol"]

    position_symbol = None
    position = None
    cash = initial_capital

    for day in dates:
        window_price = spy.shift(MEAN_REVERSION_LOOKBACK).loc[day]
        if pd.isna(window_price):
            continue
        return_lookback = (float(spy.loc[day]) / float(window_price)) - 1.0
        vix_ok = float(vix.loc[day]) < VOL_RISK_OFF_VIX
        desired = "SPY" if (return_lookback <= MEAN_REVERSION_THRESHOLD and vix_ok) else "IEF"

        if position_symbol is None:
            price = float(prices[desired].loc[day])
            quantity = cash / price if price != 0 else 0.0
            position_symbol = desired
            position = Position(quantity=quantity, avg_price=price)
            cash = 0.0

        if desired != position_symbol:
            current_price = float(prices[position_symbol].loc[day])
            realized = (current_price - position.avg_price) * position.quantity
            _append_entry(
                entries,
                _realized_entry(
                    day,
                    position_symbol,
                    realized,
                    _module_for_ticker(position_symbol, growth_ticker),
                    tags,
                ),
            )
            cash = current_price * position.quantity
            new_price = float(prices[desired].loc[day])
            quantity = cash / new_price if new_price != 0 else 0.0
            position_symbol = desired
            position = Position(quantity=quantity, avg_price=new_price)
            cash = 0.0

        mark_price = float(prices[position_symbol].loc[day])
        _append_entry(
            entries,
            _pnl_entry(
                day,
                position_symbol,
                position,
                mark_price,
                _module_for_ticker(position_symbol, growth_ticker),
                tags,
            ),
        )
    return entries


def _dual_momentum_crash_filter(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    symbols = ["SPY", "QQQ", "CLR.SI"]
    dates = _common_dates(prices, symbols + ["^VIX", "^SPX"])
    spy = prices["SPY"]
    spx = prices["^SPX"]
    vix = prices["^VIX"]
    sma200 = _sma(spy, 200)
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "dual_momentum_crash"]

    position_symbol = None
    position = None
    cash = initial_capital
    current_month = None

    for day in dates:
        month_key = _month_key(day)
        if current_month != month_key:
            current_month = month_key
            crash = float(vix.loc[day]) > CRASH_VIX_THRESHOLD or (
                not pd.isna(sma200.loc[day]) and float(spy.loc[day]) < float(sma200.loc[day])
            )
            if crash:
                desired = "SPX_PUT"
            else:
                returns = {}
                for symbol in symbols:
                    series = prices[symbol].loc[:day]
                    if len(series) <= MOMENTUM_WINDOW_DAYS:
                        returns[symbol] = None
                    else:
                        returns[symbol] = (series.iloc[-1] / series.iloc[-MOMENTUM_WINDOW_DAYS]) - 1.0
                valid = {k: v for k, v in returns.items() if v is not None}
                if not valid:
                    continue
                desired = max(valid, key=valid.get)

            if position_symbol != desired:
                if position_symbol is not None:
                    price = float(spx.loc[day]) if position_symbol == "SPX_PUT" else float(prices[position_symbol].loc[day])
                    realized = (price - position.avg_price) * position.quantity
                    _append_entry(
                        entries,
                        _realized_entry(
                            day,
                            position_symbol,
                            realized,
                            _module_for_ticker(position_symbol, growth_ticker),
                            tags,
                        ),
                    )
                    cash = price * position.quantity
                price = float(spx.loc[day]) if desired == "SPX_PUT" else float(prices[desired].loc[day])
                quantity = cash / price if price != 0 else 0.0
                position_symbol = desired
                position = Position(quantity=quantity, avg_price=price)
                cash = 0.0

        if position_symbol is None:
            continue
        mark_price = float(spx.loc[day]) if position_symbol == "SPX_PUT" else float(prices[position_symbol].loc[day])
        mark_symbol = "^SPX" if position_symbol == "SPX_PUT" else None
        _append_entry(
            entries,
            _pnl_entry(
                day,
                position_symbol,
                position,
                mark_price,
                _module_for_ticker(position_symbol, growth_ticker),
                tags,
                mark_symbol=mark_symbol,
            ),
        )
    return entries


def _vol_targeting_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    spy = prices["SPY"]
    vix = prices["^VIX"]
    dates = _common_dates(prices, ["SPY", "^VIX"])
    returns = _returns(spy)
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "vol_targeting"]

    position_symbol = None
    position = None
    cash = initial_capital

    for day in dates:
        window = returns.loc[:day].tail(VOL_TARGET_WINDOW)
        if len(window) < VOL_TARGET_WINDOW or window.std() == 0:
            target_leverage = 0.0
        else:
            realized_vol = float(window.std()) * math.sqrt(TRADING_DAYS)
            target_leverage = min(VOL_TARGET_MAX_LEVERAGE, VOL_TARGET / realized_vol)

        if float(vix.loc[day]) > 30:
            target_leverage = 0.0

        price = float(spy.loc[day])
        equity = cash + (position.quantity * price if position_symbol == "SPY" else 0.0)

        if target_leverage == 0.0:
            if position_symbol == "SPY":
                realized = (price - position.avg_price) * position.quantity
                _append_entry(
                    entries,
                    _realized_entry(
                        day,
                        position_symbol,
                        realized,
                        _module_for_ticker(position_symbol, growth_ticker),
                        tags,
                    ),
                )
                cash = equity
                position_symbol = None
                position = None
            continue

        target_qty = (equity * target_leverage) / price if price != 0 else 0.0
        if position_symbol != "SPY":
            position_symbol = "SPY"
            position = Position(quantity=target_qty, avg_price=price)
            cash = equity - (target_qty * price)
        else:
            realized = (price - position.avg_price) * position.quantity
            _append_entry(
                entries,
                _realized_entry(
                    day,
                    position_symbol,
                    realized,
                    _module_for_ticker(position_symbol, growth_ticker),
                    tags,
                ),
            )
            position = Position(quantity=target_qty, avg_price=price)
            cash = equity - (target_qty * price)

        _append_entry(
            entries,
            _pnl_entry(
                day,
                position_symbol,
                position,
                price,
                _module_for_ticker(position_symbol, growth_ticker),
                tags,
            ),
        )
    return entries


def _risk_off_ief_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    spy = prices["SPY"]
    ief = prices["IEF"]
    vix = prices["^VIX"]
    dates = _common_dates(prices, ["SPY", "IEF", "^VIX"])
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "risk_off_ief"]

    position_symbol = None
    position = None
    cash = initial_capital

    for day in dates:
        vix_level = float(vix.loc[day])
        if vix_level < 20:
            desired = "SPY"
        elif vix_level > 30:
            desired = "IEF"
        else:
            desired = position_symbol or "SPY"

        if position_symbol is None:
            price = float(spy.loc[day])
            quantity = cash / price
            position_symbol = "SPY"
            position = Position(quantity=quantity, avg_price=price)
            cash = 0.0

        if desired != position_symbol:
            current_price = float(ief.loc[day]) if position_symbol == "IEF" else float(spy.loc[day])
            realized = (current_price - position.avg_price) * position.quantity
            _append_entry(
                entries,
                _realized_entry(
                    day,
                    position_symbol,
                    realized,
                    _module_for_ticker(position_symbol, growth_ticker),
                    tags,
                ),
            )
            cash = current_price * position.quantity
            new_price = float(ief.loc[day]) if desired == "IEF" else float(spy.loc[day])
            quantity = cash / new_price if new_price != 0 else 0.0
            position_symbol = desired
            position = Position(quantity=quantity, avg_price=new_price)
            cash = 0.0

        mark_price = float(ief.loc[day]) if position_symbol == "IEF" else float(spy.loc[day])
        _append_entry(
            entries,
            _pnl_entry(
                day,
                position_symbol,
                position,
                mark_price,
                _module_for_ticker(position_symbol, growth_ticker),
                tags,
            ),
        )
    return entries


def _regime_blend_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    symbols = ["SPY", "QQQ", "CLR.SI"]
    risk_off_symbol = "IEF"
    dates = _common_dates(prices, symbols + [risk_off_symbol, "^VIX"])
    spy = prices["SPY"]
    vix = prices["^VIX"]
    sma200 = _sma(spy, 200)
    returns = {symbol: _returns(prices[symbol]) for symbol in symbols + [risk_off_symbol]}
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "regime_blend"]

    positions: dict[str, Position] = {}
    cash = initial_capital
    current_month = None

    for day in dates:
        month_key = _month_key(day)
        if current_month != month_key:
            current_month = month_key
            realized_total = 0.0
            for symbol, position in positions.items():
                price = float(prices[symbol].loc[day])
                realized = (price - position.avg_price) * position.quantity
                realized_total += realized
                if position.quantity != 0:
                    _append_entry(
                        entries,
                        _realized_entry(
                            day,
                            symbol,
                            realized,
                            _module_for_ticker(symbol, growth_ticker),
                            tags,
                        ),
                    )
            cash += realized_total
            positions = {}

            in_trend = not pd.isna(sma200.loc[day]) and float(spy.loc[day]) > float(sma200.loc[day])
            risk_off = float(vix.loc[day]) >= REGIME_BLEND_RISK_OFF_VIX
            if risk_off:
                price = float(prices[risk_off_symbol].loc[day])
                quantity = cash / price if price != 0 else 0.0
                positions[risk_off_symbol] = Position(quantity=quantity, avg_price=price)
            elif in_trend:
                momentum = {}
                for symbol in symbols:
                    series = prices[symbol].loc[:day]
                    if len(series) <= MOMENTUM_WINDOW_DAYS:
                        momentum[symbol] = None
                    else:
                        momentum[symbol] = (series.iloc[-1] / series.iloc[-MOMENTUM_WINDOW_DAYS]) - 1.0
                valid = {k: v for k, v in momentum.items() if v is not None}
                if valid:
                    chosen = max(valid, key=valid.get)
                    price = float(prices[chosen].loc[day])
                    quantity = cash / price if price != 0 else 0.0
                    positions[chosen] = Position(quantity=quantity, avg_price=price)
            else:
                vols = {}
                for symbol in symbols:
                    window = returns[symbol].loc[:day].tail(VOL_WINDOW_DAYS)
                    if len(window) < VOL_WINDOW_DAYS or window.std() == 0:
                        vols[symbol] = None
                    else:
                        vols[symbol] = float(window.std())
                inv_vols = {symbol: (1.0 / vol) if vol else 0.0 for symbol, vol in vols.items()}
                total_inv = sum(inv_vols.values())
                weights = {
                    symbol: (inv_vols[symbol] / total_inv) if total_inv > 0 else 0.0
                    for symbol in symbols
                }
                risk_off_price = float(prices[risk_off_symbol].loc[day])
                risk_off_cash = cash * REGIME_BLEND_NON_TREND_IEF_SHARE
                risk_on_cash = cash - risk_off_cash
                risk_off_qty = risk_off_cash / risk_off_price if risk_off_price != 0 else 0.0
                positions[risk_off_symbol] = Position(
                    quantity=risk_off_qty,
                    avg_price=risk_off_price,
                )
                for symbol, weight in weights.items():
                    if weight == 0.0:
                        continue
                    price = float(prices[symbol].loc[day])
                    quantity = (risk_on_cash * weight) / price
                    positions[symbol] = Position(quantity=quantity, avg_price=price)

        for symbol, position in positions.items():
            module = _module_for_ticker(symbol, growth_ticker)
            mark_price = float(prices[symbol].loc[day])
            _append_entry(
                entries,
                _pnl_entry(day, symbol, position, mark_price, module, tags),
            )
    return entries


def _drawdown_throttle_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    spy = prices["SPY"]
    dates = _common_dates(prices, ["SPY"])
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "drawdown_throttle"]

    position_symbol = None
    position = None
    cash = initial_capital
    rolling_prices: list[float] = []

    for day in dates:
        price = float(spy.loc[day])
        rolling_prices.append(price)
        if len(rolling_prices) > DRAWDOWN_WINDOW:
            rolling_prices.pop(0)
        peak = max(rolling_prices) if rolling_prices else price
        drawdown = (price / peak) - 1.0 if peak else 0.0
        target_leverage = 0.5 if drawdown < -DRAWDOWN_THRESHOLD else 1.0

        equity = cash + (position.quantity * price if position_symbol == "SPY" else 0.0)
        target_qty = (equity * target_leverage) / price if price != 0 else 0.0

        if position_symbol != "SPY":
            position_symbol = "SPY"
            position = Position(quantity=target_qty, avg_price=price)
            cash = equity - (target_qty * price)
        else:
            realized = (price - position.avg_price) * position.quantity
            _append_entry(
                entries,
                _realized_entry(
                    day,
                    position_symbol,
                    realized,
                    _module_for_ticker(position_symbol, growth_ticker),
                    tags,
                ),
            )
            position = Position(quantity=target_qty, avg_price=price)
            cash = equity - (target_qty * price)

        _append_entry(
            entries,
            _pnl_entry(
                day,
                position_symbol,
                position,
                price,
                _module_for_ticker(position_symbol, growth_ticker),
                tags,
            ),
        )
    return entries


def _vol_of_vol_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    vix = prices["^VIX"]
    vvix = prices["^VVIX"]
    spy = prices["SPY"]
    spx = prices["^SPX"]
    dates = _common_dates(prices, ["^VIX", "^VVIX", "SPY", "^SPX"])
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "vol_of_vol"]

    position_symbol = None
    position = None
    cash = initial_capital

    for day in dates:
        guard = float(vvix.loc[day]) > VVIX_THRESHOLD or float(vix.loc[day]) > VIX_CRASH_THRESHOLD
        desired = "SPX_PUT" if guard else "SPY"

        if position_symbol is None:
            price = float(spy.loc[day])
            quantity = cash / price
            position_symbol = "SPY"
            position = Position(quantity=quantity, avg_price=price)
            cash = 0.0

        if desired != position_symbol:
            current_price = float(spx.loc[day]) if position_symbol == "SPX_PUT" else float(spy.loc[day])
            realized = (current_price - position.avg_price) * position.quantity
            _append_entry(
                entries,
                _realized_entry(
                    day,
                    position_symbol,
                    realized,
                    _module_for_ticker(position_symbol, growth_ticker),
                    tags,
                ),
            )
            cash = current_price * position.quantity
            new_price = float(spx.loc[day]) if desired == "SPX_PUT" else float(spy.loc[day])
            quantity = cash / new_price if new_price != 0 else 0.0
            position_symbol = desired
            position = Position(quantity=quantity, avg_price=new_price)
            cash = 0.0

        mark_price = float(spx.loc[day]) if position_symbol == "SPX_PUT" else float(spy.loc[day])
        mark_symbol = "^SPX" if position_symbol == "SPX_PUT" else None
        _append_entry(
            entries,
            _pnl_entry(
                day,
                position_symbol,
                position,
                mark_price,
                _module_for_ticker(position_symbol, growth_ticker),
                tags,
                mark_symbol=mark_symbol,
            ),
        )
    return entries


def _sma_crossover_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    spy = prices["SPY"]
    spx = prices["^SPX"]
    dates = _common_dates(prices, ["SPY", "^SPX"])
    sma_fast = _sma(spy, SMA_FAST_WINDOW)
    sma_slow = _sma(spy, SMA_SLOW_WINDOW)
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "sma_crossover"]

    position_symbol = None
    position = None
    cash = initial_capital

    for day in dates:
        if pd.isna(sma_fast.loc[day]) or pd.isna(sma_slow.loc[day]):
            continue
        desired = "SPY" if float(sma_fast.loc[day]) > float(sma_slow.loc[day]) else "SPX_PUT"

        if position_symbol is None:
            price = float(spy.loc[day])
            quantity = cash / price
            position_symbol = "SPY"
            position = Position(quantity=quantity, avg_price=price)
            cash = 0.0

        if desired != position_symbol:
            current_price = float(spx.loc[day]) if position_symbol == "SPX_PUT" else float(spy.loc[day])
            realized = (current_price - position.avg_price) * position.quantity
            _append_entry(
                entries,
                _realized_entry(
                    day,
                    position_symbol,
                    realized,
                    _module_for_ticker(position_symbol, growth_ticker),
                    tags,
                ),
            )
            cash = current_price * position.quantity
            new_price = float(spx.loc[day]) if desired == "SPX_PUT" else float(spy.loc[day])
            quantity = cash / new_price if new_price != 0 else 0.0
            position_symbol = desired
            position = Position(quantity=quantity, avg_price=new_price)
            cash = 0.0

        mark_price = float(spx.loc[day]) if position_symbol == "SPX_PUT" else float(spy.loc[day])
        mark_symbol = "^SPX" if position_symbol == "SPX_PUT" else None
        _append_entry(
            entries,
            _pnl_entry(
                day,
                position_symbol,
                position,
                mark_price,
                _module_for_ticker(position_symbol, growth_ticker),
                tags,
                mark_symbol=mark_symbol,
            ),
        )
    return entries


def _trend_rate_regime_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    spy = prices["SPY"]
    spx = prices["^SPX"]
    tnx = prices["^TNX"]
    dates = _common_dates(prices, ["SPY", "^SPX", "^TNX"])
    sma_slow = _sma(spy, SMA_SLOW_WINDOW)
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "trend_rate"]

    position_symbol = None
    position = None
    cash = initial_capital

    for day in dates:
        if pd.isna(sma_slow.loc[day]):
            continue
        slope = tnx.loc[day] - tnx.shift(RATE_SLOPE_WINDOW).loc[day]
        if pd.isna(slope):
            continue
        desired = "SPY" if (float(spy.loc[day]) > float(sma_slow.loc[day]) and slope < 0) else "SPX_PUT"

        if position_symbol is None:
            price = float(spy.loc[day])
            quantity = cash / price
            position_symbol = "SPY"
            position = Position(quantity=quantity, avg_price=price)
            cash = 0.0

        if desired != position_symbol:
            current_price = float(spx.loc[day]) if position_symbol == "SPX_PUT" else float(spy.loc[day])
            realized = (current_price - position.avg_price) * position.quantity
            _append_entry(
                entries,
                _realized_entry(
                    day,
                    position_symbol,
                    realized,
                    _module_for_ticker(position_symbol, growth_ticker),
                    tags,
                ),
            )
            cash = current_price * position.quantity
            new_price = float(spx.loc[day]) if desired == "SPX_PUT" else float(spy.loc[day])
            quantity = cash / new_price if new_price != 0 else 0.0
            position_symbol = desired
            position = Position(quantity=quantity, avg_price=new_price)
            cash = 0.0

        mark_price = float(spx.loc[day]) if position_symbol == "SPX_PUT" else float(spy.loc[day])
        mark_symbol = "^SPX" if position_symbol == "SPX_PUT" else None
        _append_entry(
            entries,
            _pnl_entry(
                day,
                position_symbol,
                position,
                mark_price,
                _module_for_ticker(position_symbol, growth_ticker),
                tags,
                mark_symbol=mark_symbol,
            ),
        )
    return entries


def _carry_momentum_blend_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    half_capital = initial_capital * 0.5
    momentum_entries = _dual_momentum_strategy(prices, half_capital, growth_ticker)
    risk_entries = _risk_parity_strategy(prices, half_capital, growth_ticker)

    combined = momentum_entries + risk_entries
    combined.sort(key=lambda entry: entry.get("timestamp", ""))
    for entry in combined:
        entry["rationale"] = "Backfill strategy PnL snapshot (carry momentum blend)."
        tags = entry.get("tags", [])
        if "carry_momentum_blend" not in tags:
            entry["tags"] = tags + ["carry_momentum_blend"]
    return combined


def _zscore_throttle_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    spy = prices["SPY"]
    ief = prices["IEF"]
    dates = _common_dates(prices, ["SPY", "IEF"])
    returns = _returns(spy)
    sma200 = _sma(spy, SMA_SLOW_WINDOW)
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "zscore_throttle"]

    spy_position = None
    ief_position = None
    cash = initial_capital

    for day in dates:
        window = returns.loc[:day].tail(ZSCORE_WINDOW)
        if len(window) < ZSCORE_WINDOW or window.std() == 0:
            target_leverage = 1.0
        else:
            zscore = (window.iloc[-1] - window.mean()) / window.std()
            target_leverage = ZSCORE_CUT if zscore < ZSCORE_THRESHOLD else 1.0

        if not pd.isna(sma200.loc[day]):
            in_trend = float(spy.loc[day]) > float(sma200.loc[day])
            if in_trend and target_leverage < ZSCORE_TREND_ON_MIN_LEVERAGE:
                target_leverage = ZSCORE_TREND_ON_MIN_LEVERAGE
            if not in_trend and "^VIX" in prices:
                vix_level = float(prices["^VIX"].loc[day])
                if vix_level >= ZSCORE_TREND_OFF_VIX:
                    target_leverage = min(
                        target_leverage,
                        ZSCORE_TREND_OFF_MAX_LEVERAGE,
                    )

        spy_price = float(spy.loc[day])
        ief_price = float(ief.loc[day])
        equity = cash
        if spy_position is not None:
            equity += spy_position.quantity * spy_price
        if ief_position is not None:
            equity += ief_position.quantity * ief_price

        spy_weight = target_leverage
        ief_weight = max(0.0, 1.0 - spy_weight)
        target_spy_qty = (equity * spy_weight) / spy_price if spy_price != 0 else 0.0
        target_ief_qty = (equity * ief_weight) / ief_price if ief_price != 0 else 0.0

        if spy_position is not None:
            realized = (spy_price - spy_position.avg_price) * spy_position.quantity
            _append_entry(
                entries,
                _realized_entry(
                    day,
                    "SPY",
                    realized,
                    _module_for_ticker("SPY", growth_ticker),
                    tags,
                ),
            )
        if ief_position is not None and ief_position.quantity != 0:
            realized = (ief_price - ief_position.avg_price) * ief_position.quantity
            _append_entry(
                entries,
                _realized_entry(
                    day,
                    "IEF",
                    realized,
                    _module_for_ticker("IEF", growth_ticker),
                    tags,
                ),
            )

        spy_position = Position(quantity=target_spy_qty, avg_price=spy_price)
        ief_position = Position(quantity=target_ief_qty, avg_price=ief_price)
        cash = equity - (target_spy_qty * spy_price) - (target_ief_qty * ief_price)

        if spy_position.quantity != 0:
            _append_entry(
                entries,
                _pnl_entry(
                    day,
                    "SPY",
                    spy_position,
                    spy_price,
                    _module_for_ticker("SPY", growth_ticker),
                    tags,
                ),
            )
        if ief_position.quantity != 0:
            _append_entry(
                entries,
                _pnl_entry(
                    day,
                    "IEF",
                    ief_position,
                    ief_price,
                    _module_for_ticker("IEF", growth_ticker),
                    tags,
                ),
            )
    return entries


def _dynamic_ief_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
    lag_days: int = DYNAMIC_IEF_LAG_DAYS,
    tag_suffix: str | None = None,
) -> list[dict[str, Any]]:
    spy = prices["SPY"]
    ief = prices["IEF"]
    dates = _common_dates(prices, ["SPY", "IEF", "^VIX"])
    returns = _returns(spy)
    sma200 = _sma(spy, SMA_SLOW_WINDOW)
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "dynamic_ief"]
    if tag_suffix:
        tags.append(tag_suffix)

    spy_position = None
    ief_position = None
    cash = initial_capital

    target_ief_weights: list[float] = []
    for day in dates:
        window = returns.loc[:day].tail(ZSCORE_WINDOW)
        if len(window) < ZSCORE_WINDOW or window.std() == 0:
            target_leverage = 1.0
        else:
            zscore = (window.iloc[-1] - window.mean()) / window.std()
            target_leverage = DYNAMIC_IEF_CUT if zscore < ZSCORE_THRESHOLD else 1.0

        if not pd.isna(sma200.loc[day]):
            in_trend = float(spy.loc[day]) > float(sma200.loc[day])
            if in_trend and target_leverage < DYNAMIC_IEF_MIN_LEVERAGE:
                target_leverage = DYNAMIC_IEF_MIN_LEVERAGE
            if not in_trend and "^VIX" in prices:
                vix_level = float(prices["^VIX"].loc[day])
                if vix_level >= DYNAMIC_IEF_VIX_CAP:
                    target_leverage = min(target_leverage, ZSCORE_TREND_OFF_MAX_LEVERAGE)

        base_ief = max(0.0, 1.0 - target_leverage)
        if not pd.isna(sma200.loc[day]):
            if float(spy.loc[day]) > float(sma200.loc[day]):
                vix_level = float(prices["^VIX"].loc[day])
                if vix_level >= DYNAMIC_IEF_VIX_THRESHOLD:
                    base_ief = max(base_ief, DYNAMIC_IEF_BOOST)
        target_ief_weights.append(min(1.0, base_ief))

    for idx, day in enumerate(dates):
        weight_idx = max(0, idx - lag_days)
        ief_weight = target_ief_weights[weight_idx]
        spy_weight = max(0.0, 1.0 - ief_weight)

        spy_price = float(spy.loc[day])
        ief_price = float(ief.loc[day])
        equity = cash
        if spy_position is not None:
            equity += spy_position.quantity * spy_price
        if ief_position is not None:
            equity += ief_position.quantity * ief_price

        target_spy_qty = (equity * spy_weight) / spy_price if spy_price != 0 else 0.0
        target_ief_qty = (equity * ief_weight) / ief_price if ief_price != 0 else 0.0

        if spy_position is not None:
            realized = (spy_price - spy_position.avg_price) * spy_position.quantity
            _append_entry(
                entries,
                _realized_entry(
                    day,
                    "SPY",
                    realized,
                    _module_for_ticker("SPY", growth_ticker),
                    tags,
                ),
            )
        if ief_position is not None and ief_position.quantity != 0:
            realized = (ief_price - ief_position.avg_price) * ief_position.quantity
            _append_entry(
                entries,
                _realized_entry(
                    day,
                    "IEF",
                    realized,
                    _module_for_ticker("IEF", growth_ticker),
                    tags,
                ),
            )

        spy_position = Position(quantity=target_spy_qty, avg_price=spy_price)
        ief_position = Position(quantity=target_ief_qty, avg_price=ief_price)
        cash = equity - (target_spy_qty * spy_price) - (target_ief_qty * ief_price)

        if spy_position.quantity != 0:
            _append_entry(
                entries,
                _pnl_entry(
                    day,
                    "SPY",
                    spy_position,
                    spy_price,
                    _module_for_ticker("SPY", growth_ticker),
                    tags,
                ),
            )
        if ief_position.quantity != 0:
            _append_entry(
                entries,
                _pnl_entry(
                    day,
                    "IEF",
                    ief_position,
                    ief_price,
                    _module_for_ticker("IEF", growth_ticker),
                    tags,
                ),
            )
    return entries


def _crash_guard_rebound_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    spy = prices["SPY"]
    spx = prices["^SPX"]
    vix = prices["^VIX"]
    dates = _common_dates(prices, ["SPY", "^SPX", "^VIX"])
    sma_slow = _sma(spy, SMA_SLOW_WINDOW)
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "crash_guard_rebound"]

    position_symbol = None
    position = None
    cash = initial_capital
    in_risk_off = False

    for day in dates:
        vix_level = float(vix.loc[day])
        if vix_level > VIX_CRASH_THRESHOLD:
            in_risk_off = True
        elif in_risk_off:
            if vix_level < VIX_RECOVERY_THRESHOLD and not pd.isna(sma_slow.loc[day]):
                if float(spy.loc[day]) > float(sma_slow.loc[day]):
                    in_risk_off = False

        desired = "SPX_PUT" if in_risk_off else "SPY"

        if position_symbol is None:
            price = float(spy.loc[day])
            quantity = cash / price
            position_symbol = "SPY"
            position = Position(quantity=quantity, avg_price=price)
            cash = 0.0

        if desired != position_symbol:
            current_price = float(spx.loc[day]) if position_symbol == "SPX_PUT" else float(spy.loc[day])
            realized = (current_price - position.avg_price) * position.quantity
            _append_entry(
                entries,
                _realized_entry(
                    day,
                    position_symbol,
                    realized,
                    _module_for_ticker(position_symbol, growth_ticker),
                    tags,
                ),
            )
            cash = current_price * position.quantity
            new_price = float(spx.loc[day]) if desired == "SPX_PUT" else float(spy.loc[day])
            quantity = cash / new_price if new_price != 0 else 0.0
            position_symbol = desired
            position = Position(quantity=quantity, avg_price=new_price)
            cash = 0.0

        mark_price = float(spx.loc[day]) if position_symbol == "SPX_PUT" else float(spy.loc[day])
        mark_symbol = "^SPX" if position_symbol == "SPX_PUT" else None
        _append_entry(
            entries,
            _pnl_entry(
                day,
                position_symbol,
                position,
                mark_price,
                _module_for_ticker(position_symbol, growth_ticker),
                tags,
                mark_symbol=mark_symbol,
            ),
        )
    return entries


def _dynamic_lookback_momentum_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    symbols = ["SPY", "QQQ", "CLR.SI"]
    dates = _common_dates(prices, symbols)
    spy = prices["SPY"]
    returns = _returns(spy)
    vol_series = returns.rolling(window=MOMENTUM_DYNAMIC_VOL_WINDOW).std() * math.sqrt(TRADING_DAYS)
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "dynamic_lookback"]

    position_symbol = None
    position = None
    cash = initial_capital
    current_month = None

    for day in dates:
        month_key = _month_key(day)
        if current_month != month_key:
            current_month = month_key
            current_vol = vol_series.loc[day] if day in vol_series.index else None
            if current_vol is None or pd.isna(current_vol):
                continue
            vol_window = vol_series.loc[:day].tail(MOMENTUM_DYNAMIC_MEDIAN_WINDOW)
            if vol_window.empty:
                continue
            median_vol = float(vol_window.median())
            lookback = MOMENTUM_WINDOW_6M if current_vol > median_vol else MOMENTUM_WINDOW_DAYS

            returns_by_symbol = {}
            for symbol in symbols:
                series = prices[symbol].loc[:day]
                if len(series) <= lookback:
                    returns_by_symbol[symbol] = None
                else:
                    returns_by_symbol[symbol] = (series.iloc[-1] / series.iloc[-lookback]) - 1.0
            valid = {k: v for k, v in returns_by_symbol.items() if v is not None}
            if not valid:
                continue
            desired = max(valid, key=valid.get)

            if position_symbol != desired:
                if position_symbol is not None:
                    price = float(prices[position_symbol].loc[day])
                    realized = (price - position.avg_price) * position.quantity
                    _append_entry(
                        entries,
                        _realized_entry(
                            day,
                            position_symbol,
                            realized,
                            _module_for_ticker(position_symbol, growth_ticker),
                            tags,
                        ),
                    )
                    cash = price * position.quantity
                price = float(prices[desired].loc[day])
                quantity = cash / price if price != 0 else 0.0
                position_symbol = desired
                position = Position(quantity=quantity, avg_price=price)
                cash = 0.0

        if position_symbol is None:
            continue
        module = _module_for_ticker(position_symbol, growth_ticker)
        mark_price = float(prices[position_symbol].loc[day])
        _append_entry(
            entries,
            _pnl_entry(day, position_symbol, position, mark_price, module, tags),
        )
    return entries


def _vix_hysteresis_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    vix = prices["^VIX"]
    spy = prices["SPY"]
    spx = prices["^SPX"]
    dates = _common_dates(prices, ["^VIX", "SPY", "^SPX"])
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "vix_hysteresis"]

    position_symbol = None
    position = None
    cash = initial_capital

    for day in dates:
        vix_level = float(vix.loc[day])
        if position_symbol is None:
            price = float(spy.loc[day])
            quantity = cash / price
            position_symbol = "SPY"
            position = Position(quantity=quantity, avg_price=price)
            cash = 0.0

        if position_symbol == "SPY" and vix_level > 30:
            desired = "SPX_PUT"
        elif position_symbol == "SPX_PUT" and vix_level < 22:
            desired = "SPY"
        else:
            desired = position_symbol

        if desired != position_symbol:
            current_price = float(spx.loc[day]) if position_symbol == "SPX_PUT" else float(spy.loc[day])
            realized = (current_price - position.avg_price) * position.quantity
            _append_entry(
                entries,
                _realized_entry(
                    day,
                    position_symbol,
                    realized,
                    _module_for_ticker(position_symbol, growth_ticker),
                    tags,
                ),
            )
            cash = current_price * position.quantity
            new_price = float(spx.loc[day]) if desired == "SPX_PUT" else float(spy.loc[day])
            quantity = cash / new_price if new_price != 0 else 0.0
            position_symbol = desired
            position = Position(quantity=quantity, avg_price=new_price)
            cash = 0.0

        mark_price = float(spx.loc[day]) if position_symbol == "SPX_PUT" else float(spy.loc[day])
        mark_symbol = "^SPX" if position_symbol == "SPX_PUT" else None
        _append_entry(
            entries,
            _pnl_entry(
                day,
                position_symbol,
                position,
                mark_price,
                _module_for_ticker(position_symbol, growth_ticker),
                tags,
                mark_symbol=mark_symbol,
            ),
        )
    return entries


def _trend_vix_blend_strategy(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    spy = prices["SPY"]
    spx = prices["^SPX"]
    vix = prices["^VIX"]
    dates = _common_dates(prices, ["SPY", "^SPX", "^VIX"])
    sma200 = _sma(spy, 200)
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "trend_vix_blend"]

    position_symbol = None
    position = None
    cash = initial_capital

    for day in dates:
        if pd.isna(sma200.loc[day]):
            continue
        desired = "SPY" if (float(spy.loc[day]) > float(sma200.loc[day]) and float(vix.loc[day]) < 25) else "SPX_PUT"

        if position_symbol is None:
            price = float(spy.loc[day])
            quantity = cash / price
            position_symbol = "SPY"
            position = Position(quantity=quantity, avg_price=price)
            cash = 0.0

        if desired != position_symbol:
            current_price = float(spx.loc[day]) if position_symbol == "SPX_PUT" else float(spy.loc[day])
            realized = (current_price - position.avg_price) * position.quantity
            _append_entry(
                entries,
                _realized_entry(
                    day,
                    position_symbol,
                    realized,
                    _module_for_ticker(position_symbol, growth_ticker),
                    tags,
                ),
            )
            cash = current_price * position.quantity
            new_price = float(spx.loc[day]) if desired == "SPX_PUT" else float(spy.loc[day])
            quantity = cash / new_price if new_price != 0 else 0.0
            position_symbol = desired
            position = Position(quantity=quantity, avg_price=new_price)
            cash = 0.0

        mark_price = float(spx.loc[day]) if position_symbol == "SPX_PUT" else float(spy.loc[day])
        mark_symbol = "^SPX" if position_symbol == "SPX_PUT" else None
        _append_entry(
            entries,
            _pnl_entry(
                day,
                position_symbol,
                position,
                mark_price,
                _module_for_ticker(position_symbol, growth_ticker),
                tags,
                mark_symbol=mark_symbol,
            ),
        )
    return entries


def _dual_momentum_risk_off(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
    window_days: int,
) -> list[dict[str, Any]]:
    symbols = ["SPY", "QQQ", "CLR.SI"]
    dates = _common_dates(prices, symbols)
    entries: list[dict[str, Any]] = []
    tag_name = "dual_momentum_6m" if window_days == MOMENTUM_WINDOW_6M else "dual_momentum_risk_off"
    tags = ["paper", "backfill", "strategy", tag_name]

    position_symbol = None
    position = None
    cash = initial_capital
    current_month = None

    for day in dates:
        month_key = _month_key(day)
        if current_month != month_key:
            current_month = month_key
            returns = {}
            for symbol in symbols:
                series = prices[symbol].loc[:day]
                if len(series) <= window_days:
                    returns[symbol] = None
                else:
                    returns[symbol] = (series.iloc[-1] / series.iloc[-window_days]) - 1.0
            valid = {k: v for k, v in returns.items() if v is not None}
            if not valid:
                continue
            best_symbol = max(valid, key=valid.get)
            desired = best_symbol if valid[best_symbol] > 0 else None

            if desired is None and position_symbol is not None:
                price = float(prices[position_symbol].loc[day])
                realized = (price - position.avg_price) * position.quantity
                _append_entry(
                    entries,
                    _realized_entry(
                        day,
                        position_symbol,
                        realized,
                        _module_for_ticker(position_symbol, growth_ticker),
                        tags,
                    ),
                )
                cash = price * position.quantity
                position_symbol = None
                position = None

            if desired and position_symbol != desired:
                if position_symbol is not None:
                    price = float(prices[position_symbol].loc[day])
                    realized = (price - position.avg_price) * position.quantity
                    _append_entry(
                        entries,
                        _realized_entry(
                            day,
                            position_symbol,
                            realized,
                            _module_for_ticker(position_symbol, growth_ticker),
                            tags,
                        ),
                    )
                    cash = price * position.quantity
                price = float(prices[desired].loc[day])
                quantity = cash / price if price != 0 else 0.0
                position_symbol = desired
                position = Position(quantity=quantity, avg_price=price)
                cash = 0.0

        if position_symbol is None:
            continue
        module = _module_for_ticker(position_symbol, growth_ticker)
        mark_price = float(prices[position_symbol].loc[day])
        _append_entry(
            entries,
            _pnl_entry(day, position_symbol, position, mark_price, module, tags),
        )
    return entries


def _monthly_rebalance(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    weights = {"SPY": 0.5, "QQQ": 0.3, "CLR.SI": 0.2}
    dates = _common_dates(prices, list(weights.keys()))
    positions: dict[str, Position] = {}
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "monthly_rebalance"]

    current_equity = initial_capital
    current_month = None

    for day in dates:
        month_key = day.strftime("%Y-%m")
        if current_month != month_key:
            current_month = month_key
            realized_total = 0.0
            for symbol, position in positions.items():
                price = float(prices[symbol].loc[day])
                realized_total += (price - position.avg_price) * position.quantity
                if position.quantity != 0:
                    _append_entry(
                        entries,
                        _realized_entry(
                            day,
                            symbol,
                            (price - position.avg_price) * position.quantity,
                            _module_for_ticker(symbol, growth_ticker),
                            tags,
                        ),
                    )
            current_equity += realized_total
            positions = {}
            for symbol, weight in weights.items():
                price = float(prices[symbol].loc[day])
                quantity = (current_equity * weight) / price
                positions[symbol] = Position(quantity=quantity, avg_price=price)

        for symbol, position in positions.items():
            module = _module_for_ticker(symbol, growth_ticker)
            mark_price = float(prices[symbol].loc[day])
            _append_entry(
                entries,
                _pnl_entry(day, symbol, position, mark_price, module, tags),
            )
    return entries


def _volatility_timing(
    prices: dict[str, pd.Series],
    initial_capital: float,
    growth_ticker: str,
) -> list[dict[str, Any]]:
    dates = _common_dates(prices, ["SPY", "^VIX", "^SPX"])
    entries: list[dict[str, Any]] = []
    tags = ["paper", "backfill", "strategy", "volatility_timing"]
    position_symbol = None
    position = None
    cash = initial_capital

    for day in dates:
        vix = float(prices["^VIX"].loc[day])
        if vix < 20:
            desired = "SPY"
        elif vix > 30:
            desired = "SPX_PUT"
        else:
            desired = position_symbol or "SPY"

        if position_symbol is None:
            price = float(prices["SPY"].loc[day])
            quantity = cash / price
            position_symbol = "SPY"
            position = Position(quantity=quantity, avg_price=price)
            cash = 0.0

        if desired != position_symbol:
            current_price = float(
                prices["^SPX"].loc[day]
                if position_symbol == "SPX_PUT"
                else prices[position_symbol].loc[day]
            )
            realized = (current_price - position.avg_price) * position.quantity
            _append_entry(
                entries,
                _realized_entry(
                    day,
                    position_symbol,
                    realized,
                    _module_for_ticker(position_symbol, growth_ticker),
                    tags,
                ),
            )
            cash = current_price * position.quantity

            new_price = float(
                prices["^SPX"].loc[day]
                if desired == "SPX_PUT"
                else prices[desired].loc[day]
            )
            quantity = cash / new_price if new_price != 0 else 0.0
            position_symbol = desired
            position = Position(quantity=quantity, avg_price=new_price)
            cash = 0.0

        mark_price = float(
            prices["^SPX"].loc[day]
            if position_symbol == "SPX_PUT"
            else prices[position_symbol].loc[day]
        )
        module = _module_for_ticker(position_symbol, growth_ticker)
        mark_symbol = "^SPX" if position_symbol == "SPX_PUT" else None
        _append_entry(
            entries,
            _pnl_entry(
                day,
                position_symbol,
                position,
                mark_price,
                module,
                tags,
                mark_symbol=mark_symbol,
            ),
        )

    return entries


def _write_entries(entries: list[dict[str, Any]], output_path: Path) -> None:
    output_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def _append_to_ledger(entries: list[dict[str, Any]], ledger_path: Path) -> None:
    data: list[dict[str, Any]] = []
    if ledger_path.exists():
        data = json.loads(ledger_path.read_text(encoding="utf-8"))
    data.extend(entries)
    ledger_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _summarize_entries(entries: list[dict[str, Any]]) -> dict[str, Any]:
    actions: dict[str, int] = {}
    timestamps = [entry.get("timestamp") for entry in entries if entry.get("timestamp")]
    for entry in entries:
        action = str(entry.get("action", ""))
        actions[action] = actions.get(action, 0) + 1
    summary = {
        "entries": len(entries),
        "actions": actions,
        "first_timestamp": min(timestamps) if timestamps else None,
        "last_timestamp": max(timestamps) if timestamps else None,
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill PAPER_PNL entries from simple strategy rules.",
    )
    parser.add_argument(
        "--strategy",
        choices=[
            "buy_hold",
            "monthly_rebalance",
            "volatility_timing",
            "momentum",
            "dual_momentum_6m",
            "trend_vix",
            "trend_vix_blend",
            "vix_hysteresis",
            "vix_hysteresis_v2",
            "trend_breadth",
            "dual_momentum_crash",
            "vol_targeting",
            "risk_off_ief",
            "regime_blend",
            "drawdown_throttle",
            "vol_of_vol",
            "sma_crossover",
            "trend_rate",
            "carry_momentum_blend",
            "carry_trend_hybrid",
            "vol_risk_control",
            "macro_rate_regime",
            "breadth_quality",
            "mean_reversion_vol",
            "defensive_carry",
            "macro_vol_regime",
            "trend_breadth_hybrid",
            "short_term_reversion",
            "drawdown_control",
            "cash_tilt_regime",
            "defensive_carry_v2",
            "vol_breakout",
            "risk_parity_crash_guard",
            "trend_vol_drawdown",
            "carry_macro_regime",
            "zscore_throttle",
            "dynamic_ief",
            "dynamic_ief_lag1",
            "dynamic_ief_lag2",
            "dynamic_ief_lag3",
            "crash_guard_rebound",
            "dynamic_lookback",
            "risk_parity",
            "dual_momentum",
            "dual_momentum_risk_off",
            "combined",
        ],
        required=True,
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument(
        "--dynamic-ief-lag",
        type=int,
        default=0,
        help="Lag days for dynamic_ief strategy (applies only to --strategy dynamic_ief).",
    )
    parser.add_argument(
        "--output-ledger",
        default="",
        help="Write output entries to a new ledger JSON file.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append entries to the live ledger file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print entry summary only; do not write output.",
    )
    args = parser.parse_args()

    config = load_config()
    tickers = [
        "^SPX",
        "SPY",
        "QQQ",
        "^VIX",
        "^VVIX",
        "^TNX",
        "CLR.SI",
        "IEF",
        "RSP",
    ]
    prices = _download_prices(tickers, args.start, args.end)
    if args.strategy == "buy_hold":
        entries = _buy_hold(prices, config.paper_initial_capital, config.growth_ticker)
    elif args.strategy == "monthly_rebalance":
        entries = _monthly_rebalance(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "momentum":
        entries = _momentum_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "dual_momentum_6m":
        entries = _dual_momentum_risk_off(
            prices,
            config.paper_initial_capital,
            config.growth_ticker,
            MOMENTUM_WINDOW_6M,
        )
    elif args.strategy == "trend_vix":
        entries = _trend_vix_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "trend_vix_blend":
        entries = _trend_vix_blend_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "vix_hysteresis":
        entries = _vix_hysteresis_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "vix_hysteresis_v2":
        entries = _vix_hysteresis_v2_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "trend_breadth":
        entries = _trend_breadth_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "dual_momentum_crash":
        entries = _dual_momentum_crash_filter(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "vol_targeting":
        entries = _vol_targeting_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "risk_off_ief":
        entries = _risk_off_ief_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "regime_blend":
        entries = _regime_blend_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "drawdown_throttle":
        entries = _drawdown_throttle_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "vol_of_vol":
        entries = _vol_of_vol_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "sma_crossover":
        entries = _sma_crossover_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "trend_rate":
        entries = _trend_rate_regime_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "carry_momentum_blend":
        entries = _carry_momentum_blend_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "carry_trend_hybrid":
        entries = _carry_trend_hybrid_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "vol_risk_control":
        entries = _vol_risk_control_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "macro_rate_regime":
        entries = _macro_rate_regime_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "breadth_quality":
        entries = _breadth_quality_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "mean_reversion_vol":
        entries = _mean_reversion_vol_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "defensive_carry":
        entries = _defensive_carry_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "macro_vol_regime":
        entries = _macro_vol_regime_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "trend_breadth_hybrid":
        entries = _trend_breadth_hybrid_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "short_term_reversion":
        entries = _short_term_reversion_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "drawdown_control":
        entries = _drawdown_control_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "cash_tilt_regime":
        entries = _cash_tilt_regime_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "defensive_carry_v2":
        entries = _defensive_carry_v2_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "vol_breakout":
        entries = _vol_breakout_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "risk_parity_crash_guard":
        entries = _risk_parity_crash_guard_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "trend_vol_drawdown":
        entries = _trend_vol_target_drawdown_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "carry_macro_regime":
        entries = _carry_macro_regime_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "zscore_throttle":
        entries = _zscore_throttle_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "dynamic_ief":
        tag_suffix = None
        if args.dynamic_ief_lag > 0:
            tag_suffix = f"lag_{args.dynamic_ief_lag}d"
        entries = _dynamic_ief_strategy(
            prices,
            config.paper_initial_capital,
            config.growth_ticker,
            lag_days=args.dynamic_ief_lag,
            tag_suffix=tag_suffix,
        )
    elif args.strategy == "dynamic_ief_lag1":
        entries = _dynamic_ief_strategy(
            prices,
            config.paper_initial_capital,
            config.growth_ticker,
            lag_days=1,
            tag_suffix="lag_1d",
        )
    elif args.strategy == "dynamic_ief_lag2":
        entries = _dynamic_ief_strategy(
            prices,
            config.paper_initial_capital,
            config.growth_ticker,
            lag_days=2,
            tag_suffix="lag_2d",
        )
    elif args.strategy == "dynamic_ief_lag3":
        entries = _dynamic_ief_strategy(
            prices,
            config.paper_initial_capital,
            config.growth_ticker,
            lag_days=3,
            tag_suffix="lag_3d",
        )
    elif args.strategy == "crash_guard_rebound":
        entries = _crash_guard_rebound_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "dynamic_lookback":
        entries = _dynamic_lookback_momentum_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "risk_parity":
        entries = _risk_parity_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "dual_momentum":
        entries = _dual_momentum_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    elif args.strategy == "dual_momentum_risk_off":
        entries = _dual_momentum_risk_off(
            prices,
            config.paper_initial_capital,
            config.growth_ticker,
            MOMENTUM_WINDOW_DAYS,
        )
    elif args.strategy == "combined":
        entries = _combined_strategy(
            prices, config.paper_initial_capital, config.growth_ticker
        )
    else:
        entries = _volatility_timing(
            prices, config.paper_initial_capital, config.growth_ticker
        )

    if args.dry_run:
        summary = _summarize_entries(entries)
        print(json.dumps(summary, indent=2))
        return 0

    if args.append:
        _append_to_ledger(entries, Path(config.log_file))
        print(f"Appended {len(entries)} entries to {config.log_file}")
        return 0

    output_path = Path(
        args.output_ledger
        or f"reports/pnl_backfill_{args.strategy}_{args.start}_{args.end}.json"
    )
    _write_entries(entries, output_path)
    print(f"Wrote {len(entries)} entries to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
