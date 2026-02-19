from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time
from typing import Any
import re

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

from .cache_utils import (
    cache_age_hours,
    read_cache,
    read_series_cache,
    series_cache_age_hours,
    write_cache,
    write_series_cache,
)
from .config import AppConfig

DEFAULT_MAS_TBILL_ESERVICES_URL = (
    "https://eservices.mas.gov.sg/statistics/fdanet/BenchmarkPricesAndYields.aspx"
)

LONG_HISTORY_TICKERS = {"SPY", "IEF", "^VIX"}


@dataclass
class MarketSignals:
    iv: float
    rv: float
    vvix: float
    spread: float
    spx: float


@dataclass
class GrowthSignal:
    ticker: str
    price: float
    sma: float
    ma_days: int
    above: bool


def _series_cache_key(ticker: str) -> str:
    return "market_" + re.sub(r"[^a-zA-Z0-9_-]", "_", ticker)


def _series_long_cache_key(ticker: str) -> str:
    return "market_long_" + re.sub(r"[^a-zA-Z0-9_-]", "_", ticker)


def _fetch_yahoo_series(
    ticker: str,
    period: str,
    retries: int = 3,
    backoff_seconds: float = 1.0,
) -> tuple[pd.Series | None, str | None]:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            series = yf.Ticker(ticker).history(period=period)["Close"].dropna()
            if not series.empty:
                return series, None
            last_error = "empty_series"
        except Exception as exc:
            last_error = str(exc)
        if attempt < retries:
            print(f"Yahoo fetch failed ({ticker}) attempt {attempt}/{retries}: {last_error}")
            time.sleep(backoff_seconds * (2 ** (attempt - 1)))
    return None, last_error


def fetch_close_series(
    ticker: str,
    period: str = "1y",
    cache_dir: Path | None = None,
    max_age_hours: int = 48,
) -> pd.Series | None:
    series, error = _fetch_yahoo_series(ticker, period)
    if series is not None and not series.empty:
        if cache_dir is not None:
            if ticker in LONG_HISTORY_TICKERS:
                write_series_cache(cache_dir, _series_cache_key(ticker), series)
                write_series_cache(cache_dir, _series_long_cache_key(ticker), series)
            else:
                write_series_cache(cache_dir, _series_cache_key(ticker), series.tail(120))
        return series
    if error:
        print(f"Failed to fetch {ticker}: {error}")
    if cache_dir is None:
        return None
    cached = read_series_cache(cache_dir, _series_cache_key(ticker), max_age_hours)
    if cached is None or cached.empty:
        return None
    return cached


def fetch_close_series_status(
    ticker: str,
    period: str,
    cache_dir: Path,
    max_age_hours: int,
) -> tuple[pd.Series | None, dict[str, Any]]:
    series, error = _fetch_yahoo_series(ticker, period)
    if series is not None and not series.empty:
        if ticker in LONG_HISTORY_TICKERS:
            write_series_cache(cache_dir, _series_cache_key(ticker), series)
            write_series_cache(cache_dir, _series_long_cache_key(ticker), series)
        else:
            write_series_cache(cache_dir, _series_cache_key(ticker), series.tail(120))
        return series, {
            "source": "live",
            "age_days": _data_recency_days(series),
            "cache_age_hours": None,
        }
    if error:
        print(f"Failed to fetch {ticker}: {error}")

    cache_age = series_cache_age_hours(cache_dir, _series_cache_key(ticker))
    cached = read_series_cache(cache_dir, _series_cache_key(ticker), max_age_hours)
    if cached is not None and not cached.empty:
        series = cached
        return series, {
            "source": "cache",
            "age_days": _data_recency_days(series),
            "cache_age_hours": cache_age,
        }

    return None, {
        "source": "missing",
        "age_days": None,
        "cache_age_hours": cache_age,
    }


def _data_recency_days(series: pd.Series) -> float | None:
    if series.empty:
        return None
    last_ts = series.index[-1]
    try:
        last_dt = pd.to_datetime(last_ts).to_pydatetime()
    except Exception:
        return None
    if getattr(last_dt, "tzinfo", None) is not None:
        last_dt = last_dt.replace(tzinfo=None)
    delta = datetime.now() - last_dt
    return delta.total_seconds() / 86400.0


def check_market_data_quality(config: AppConfig) -> dict[str, float | None]:
    cache_dir = Path(config.cache_dir)
    max_age = config.market_cache_max_age_hours
    spx, spx_status = fetch_close_series_status(
        config.ticker, "1y", cache_dir, max_age
    )
    vix, vix_status = fetch_close_series_status(
        config.vix_ticker, "1y", cache_dir, max_age
    )
    vvix, vvix_status = fetch_close_series_status(
        config.vvix_ticker, "1y", cache_dir, max_age
    )

    data = {
        "spx_age_days": _data_recency_days(spx) if spx is not None else None,
        "vix_age_days": _data_recency_days(vix) if vix is not None else None,
        "vvix_age_days": _data_recency_days(vvix) if vvix is not None else None,
        "spx_source": spx_status.get("source"),
        "vix_source": vix_status.get("source"),
        "vvix_source": vvix_status.get("source"),
    }

    for key, status in (
        ("spx", spx_status),
        ("vix", vix_status),
        ("vvix", vvix_status),
    ):
        cache_age = status.get("cache_age_hours")
        if cache_age is not None and status.get("source") == "cache":
            data[f"{key}_cache_age_days"] = cache_age / 24.0

    return data


def get_growth_signal(config: AppConfig) -> GrowthSignal | None:
    cache_dir = Path(config.cache_dir)
    max_age = config.market_cache_max_age_hours
    series = fetch_close_series(
        config.growth_ticker, cache_dir=cache_dir, max_age_hours=max_age
    )
    if series is None or series.empty:
        return None
    ma_days = config.growth_ma_days
    if len(series) < ma_days:
        return None
    sma = series.rolling(window=ma_days).mean().iloc[-1]
    price = float(series.iloc[-1])
    return GrowthSignal(
        ticker=config.growth_ticker,
        price=price,
        sma=float(sma),
        ma_days=ma_days,
        above=price >= float(sma),
    )


def get_market_signals(config: AppConfig) -> MarketSignals | None:
    cache_dir = Path(config.cache_dir)
    max_age = config.market_cache_max_age_hours
    spx = fetch_close_series(config.ticker, cache_dir=cache_dir, max_age_hours=max_age)
    vix = fetch_close_series(config.vix_ticker, cache_dir=cache_dir, max_age_hours=max_age)
    vvix = fetch_close_series(config.vvix_ticker, cache_dir=cache_dir, max_age_hours=max_age)
    if spx is None or vix is None or vvix is None:
        return None
    if spx.empty or vix.empty or vvix.empty:
        return None

    log_returns = np.log(spx / spx.shift(1))
    rv = log_returns.rolling(window=30).std() * np.sqrt(252) * 100
    current_rv = rv.dropna().iloc[-1]
    current_iv = vix.iloc[-1]
    current_vvix = vvix.iloc[-1]

    spread = current_iv - current_rv

    return MarketSignals(
        iv=float(current_iv),
        rv=float(current_rv),
        vvix=float(current_vvix),
        spread=float(spread),
        spx=float(spx.iloc[-1]),
    )


def get_reit_signals(
    config: AppConfig, risk_free_rate: float | None = None
) -> pd.DataFrame:
    rows = []
    rate = config.risk_free_rate if risk_free_rate is None else risk_free_rate
    for ticker, name in config.reit_tickers.items():
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}
            yield_pct = float(info.get("dividendYield", 0) or 0) * 100
            price = float(info.get("regularMarketPrice", 0) or 0)
        except Exception:
            yield_pct = 0.0
            price = 0.0
        spread = yield_pct - rate
        rows.append(
            {
                "Ticker": ticker,
                "Name": name,
                "Yield": yield_pct,
                "Spread": spread,
                "Price": price,
            }
        )
    return pd.DataFrame(rows)


def _coerce_numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _extract_numeric(payload: Any) -> float | None:
    if payload is None:
        return None
    numeric = _coerce_numeric(payload)
    if numeric is not None:
        return numeric
    if isinstance(payload, dict):
        for key in (
            "value",
            "yield",
            "tbill",
            "tbill_6m",
            "tbill_6_month",
            "t_bill_6m",
            "t_bill_6_month",
        ):
            if key in payload:
                candidate = _coerce_numeric(payload[key])
                if candidate is not None:
                    return candidate
        for value in payload.values():
            candidate = _extract_numeric(value)
            if candidate is not None:
                return candidate
    if isinstance(payload, list):
        for item in reversed(payload):
            candidate = _extract_numeric(item)
            if candidate is not None:
                return candidate
    return None


def _parse_numeric_from_text(text: str) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _extract_percent_value(text: str) -> float | None:
    percent_match = re.search(r"(\d+(?:\.\d+)?)%", text)
    if percent_match:
        try:
            return float(percent_match.group(1))
        except ValueError:
            return None
    numbers = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", text)]
    for value in numbers:
        if 0 < value < 20:
            return value
    return None


def _find_cutoff_yield_from_table(soup: BeautifulSoup) -> float | None:
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if not rows:
            continue
        header_cells = [
            cell.get_text(" ", strip=True).lower()
            for cell in rows[0].find_all(["th", "td"])
        ]
        cutoff_idx = None
        for idx, header in enumerate(header_cells):
            if ("cut-off" in header or "cutoff" in header) and (
                "yield" in header or "%" in header
            ):
                cutoff_idx = idx
                break
        for row in rows[1:]:
            cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
            if not cells:
                continue
            row_text = " ".join(cells).lower()
            if "6-month" not in row_text and "6 month" not in row_text:
                continue
            if cutoff_idx is not None and cutoff_idx < len(cells):
                candidate = _parse_numeric_from_text(cells[cutoff_idx])
                if candidate is not None:
                    return candidate
            for cell in cells:
                if "6-month" in cell.lower() or "6 month" in cell.lower():
                    continue
                candidate = _parse_numeric_from_text(cell)
                if candidate is not None:
                    return candidate
    return None


def _find_cutoff_yield_in_eservices_table(soup: BeautifulSoup) -> float | None:
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if not rows:
            continue
        header_cells = [
            cell.get_text(" ", strip=True).lower()
            for cell in rows[0].find_all(["th", "td"])
        ]
        cutoff_idx = None
        tenor_idx = None
        for idx, header in enumerate(header_cells):
            if "cut" in header and "yield" in header:
                cutoff_idx = idx
            if "tenor" in header or "term" in header:
                tenor_idx = idx
        for row in rows[1:]:
            cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
            if not cells:
                continue
            row_text = " ".join(cells).lower()
            is_six_month = "6" in row_text and "month" in row_text
            if tenor_idx is not None and tenor_idx < len(cells):
                tenor_text = cells[tenor_idx].lower()
                is_six_month = "6" in tenor_text and "month" in tenor_text
            if not is_six_month:
                continue
            if cutoff_idx is not None and cutoff_idx < len(cells):
                candidate = _parse_numeric_from_text(cells[cutoff_idx])
                if candidate is not None:
                    return candidate
            for cell in cells:
                if "yield" not in cell.lower():
                    continue
                candidate = _parse_numeric_from_text(cell)
                if candidate is not None:
                    return candidate
    return None


def _extract_latest_series_value(soup: BeautifulSoup) -> float | None:
    for table in soup.find_all("table"):
        headers = [
            cell.get_text(" ", strip=True).lower()
            for cell in table.find_all("th")
        ]
        if not headers:
            continue
        has_six_month = any("six month" in h or "6-month" in h for h in headers)
        has_yield = any("yield" in h for h in headers)
        if not (has_six_month and has_yield):
            continue
        rows = table.find_all("tr")
        for row in reversed(rows[1:]):
            cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
            for cell in reversed(cells):
                candidate = _parse_numeric_from_text(cell)
                if candidate is not None:
                    return candidate
    return None


def _find_benchmark_6m_yield(soup: BeautifulSoup) -> float | None:
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            row_text = row.get_text(" ", strip=True).lower()
            if "6-month t-bill" not in row_text:
                continue
            candidate = _extract_percent_value(row_text)
            if candidate is not None:
                return candidate
    return None


def _find_benchmark_6m_yield_in_text(text: str) -> float | None:
    normalized = text.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    match = re.search(r"6\W*month\W*t\W*-?\W*bills?", normalized, re.IGNORECASE)
    if not match:
        return None
    window = normalized[match.start() : match.start() + 400]
    candidate = _extract_percent_value(window)
    if candidate is not None:
        return candidate
    return None


def _shift_month(year: int, month: int, delta: int) -> tuple[int, int]:
    total = year * 12 + (month - 1) + delta
    new_year = total // 12
    new_month = total % 12 + 1
    return new_year, new_month


def _request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    headers: dict[str, str],
    data: dict[str, str] | None = None,
    retries: int = 3,
    backoff_seconds: float = 1.0,
) -> requests.Response:
    for attempt in range(1, retries + 1):
        try:
            if method == "GET":
                response = session.get(url, headers=headers, timeout=10)
            else:
                response = session.post(url, headers=headers, data=data, timeout=10)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            if attempt == retries:
                raise
            print(f"MAS fetch failed (attempt {attempt}/{retries}): {exc}")
            time.sleep(backoff_seconds * (2 ** (attempt - 1)))
    raise requests.RequestException("MAS fetch failed after retries")


def fetch_mas_tbill_yield_from_eservices_page(
    url: str, year: int, month: int
) -> tuple[float | None, str]:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml",
    }
    session = requests.Session()
    try:
        response = _request_with_retry(session, "GET", url, headers)
    except requests.RequestException as exc:
        return None, f"eservices_error:initial:{exc}"

    soup = BeautifulSoup(response.text, "html.parser")
    hidden_fields = {
        inp.get("name"): inp.get("value", "")
        for inp in soup.find_all("input", {"type": "hidden"})
        if inp.get("name")
    }
    payload = {
        **hidden_fields,
        "ctl00$ContentPlaceHolder1$StartYearDropDownList": str(year),
        "ctl00$ContentPlaceHolder1$StartMonthDropDownList": str(month),
        "ctl00$ContentPlaceHolder1$EndYearDropDownList": str(year),
        "ctl00$ContentPlaceHolder1$EndMonthDropDownList": str(month),
        "ctl00$ContentPlaceHolder1$FrequencyDropDownList": "D",
        "ctl00$ContentPlaceHolder1$SixMonthTreasuryBillYieldCheckBox": "on",
        "ctl00$ContentPlaceHolder1$DisplayButton": "Display",
    }
    try:
        response = _request_with_retry(
            session,
            "POST",
            url,
            headers,
            data=payload,
        )
    except requests.RequestException as exc:
        return None, f"eservices_error:post:{exc}"

    soup = BeautifulSoup(response.text, "html.parser")
    candidate = _extract_latest_series_value(soup)
    if candidate is not None:
        return candidate, "eservices_ok"
    candidate = _find_cutoff_yield_in_eservices_table(soup)
    if candidate is not None:
        return candidate, "eservices_ok"
    return None, "eservices_no_value"


def fetch_mas_tbill_yield(config: AppConfig) -> tuple[float | None, str]:
    now = datetime.now()
    last_status = "eservices_no_value"
    months_back = max(0, int(config.mas_months_back))
    for offset in range(0, months_back + 1):
        year, month = _shift_month(now.year, now.month, -offset)
        rate, status = fetch_mas_tbill_yield_from_eservices_page(
            DEFAULT_MAS_TBILL_ESERVICES_URL,
            year,
            month,
        )
        if rate is not None:
            write_cache(Path(config.cache_dir), "mas_tbill_6m", rate)
            return rate, status
        last_status = status
    cache_entry = read_cache(Path(config.cache_dir), "mas_tbill_6m")
    if cache_entry is not None:
        age_hours = cache_age_hours(cache_entry)
        if age_hours is not None and age_hours <= config.mas_cache_max_age_hours:
            return cache_entry.value, f"cache_ok:{age_hours:.2f}h"
        return None, "cache_stale"
    return None, f"{last_status}:months_back_{months_back}"


def check_fortress_rebalance(
    config: AppConfig,
    reit_spread_threshold: float | None = None,
    risk_free_rate: float | None = None,
) -> tuple[str | None, pd.DataFrame]:
    threshold = (
        config.reit_spread_threshold
        if reit_spread_threshold is None
        else reit_spread_threshold
    )
    df = get_reit_signals(config, risk_free_rate=risk_free_rate)
    opportunities = df[df["Spread"] > threshold]
    if opportunities.empty:
        return None, opportunities

    lines = ["FORTRESS REBALANCE SIGNAL"]
    for _, row in opportunities.iterrows():
        lines.append(
            f"- {row['Name']} ({row['Ticker']}): "
            f"Yield {row['Yield']:.2f}%, Spread {row['Spread']:.2f}%"
        )
    return "\n".join(lines), opportunities


def calculate_shield_strike(current_spx: float, vix: float, dte: int = 180) -> float:
    sigma_3_move = 3 * (vix / 100) * np.sqrt(dte / 252)
    return current_spx * (1 - sigma_3_move)
