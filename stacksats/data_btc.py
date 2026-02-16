"""BTC-only data provider services."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

from .btc_price_fetcher import fetch_btc_price_historical, fetch_btc_price_robust


class DataLoadError(RuntimeError):
    """Raised when BTC source data cannot be loaded safely."""


def _is_cache_usable(csv_bytes: bytes, backtest_start: pd.Timestamp, today: pd.Timestamp) -> bool:
    """Return True when cached CoinMetrics data appears complete and usable."""
    try:
        cached_df = pd.read_csv(BytesIO(csv_bytes), usecols=["time", "PriceUSD"])
        if cached_df.empty:
            return False

        cached_df["time"] = pd.to_datetime(cached_df["time"], errors="coerce")
        cached_df["PriceUSD"] = pd.to_numeric(cached_df["PriceUSD"], errors="coerce")
        cached_df = cached_df.dropna(subset=["time"]).sort_values("time")
        if cached_df.empty:
            return False

        latest_date = cached_df["time"].max().normalize()
        if latest_date < (today - pd.Timedelta(days=3)):
            return False

        in_window = (cached_df["time"] >= backtest_start) & (cached_df["time"] <= today)
        return bool(cached_df.loc[in_window, "PriceUSD"].notna().any())
    except Exception:
        return False


@dataclass
class BTCDataProvider:
    """BTC-only data provider with cache/fetch/gap-fill behavior."""

    cache_dir: str | None = "~/.stacksats/cache"
    max_age_hours: int = 24
    clock: Callable[[], pd.Timestamp] = pd.Timestamp.now

    def load(self, *, backtest_start: str = "2018-01-01") -> pd.DataFrame:
        url = "https://raw.githubusercontent.com/coinmetrics/data/refs/heads/master/csv/btc.csv"
        use_cache = self.cache_dir is not None

        logging.info("Loading CoinMetrics BTC data...")
        csv_bytes: bytes
        cache_path: Path | None = None
        now = self.clock()
        today = now.normalize()
        backtest_start_ts = pd.to_datetime(backtest_start)

        def _download_csv(*, cache_available: bool) -> bytes:
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                return response.content
            except requests.RequestException as exc:
                cache_note = (
                    " A local cache file exists but was not usable."
                    if cache_available
                    else " No local cache file was available."
                )
                raise DataLoadError(
                    "Unable to download CoinMetrics BTC data."
                    " Check internet/proxy settings and retry."
                    f"{cache_note}"
                ) from exc

        if use_cache:
            cache_path = Path(self.cache_dir).expanduser() / "coinmetrics_btc.csv"
            if cache_path.exists():
                age_hours = (now.timestamp() - cache_path.stat().st_mtime) / 3600.0
                if age_hours <= self.max_age_hours:
                    try:
                        cached_bytes = cache_path.read_bytes()
                    except OSError:
                        cached_bytes = b""
                    if _is_cache_usable(cached_bytes, backtest_start_ts, today):
                        csv_bytes = cached_bytes
                    else:
                        csv_bytes = _download_csv(cache_available=True)
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            cache_path.write_bytes(csv_bytes)
                        except OSError as exc:
                            logging.warning("Could not write cache file %s: %s", cache_path, exc)
                else:
                    csv_bytes = _download_csv(cache_available=True)
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        cache_path.write_bytes(csv_bytes)
                    except OSError as exc:
                        logging.warning("Could not write cache file %s: %s", cache_path, exc)
            else:
                csv_bytes = _download_csv(cache_available=False)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    cache_path.write_bytes(csv_bytes)
                except OSError as exc:
                    logging.warning("Could not write cache file %s: %s", cache_path, exc)
        else:
            csv_bytes = _download_csv(cache_available=False)

        try:
            df = pd.read_csv(BytesIO(csv_bytes))
        except Exception as exc:
            raise DataLoadError(
                "Downloaded CoinMetrics data could not be parsed as CSV."
                " If the issue persists, remove the local cache file and retry."
            ) from exc
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
        df.index = df.index.normalize().tz_localize(None)
        df = df.loc[~df.index.duplicated(keep="last")].sort_index()

        if "PriceUSD" not in df.columns:
            raise ValueError("PriceUSD column not found in CoinMetrics data")

        price_col = "PriceUSD_coinmetrics"
        mvrv_col = "CapMVRVCur"
        df[price_col] = df["PriceUSD"]

        full_date_range = pd.date_range(start=backtest_start_ts, end=today, freq="D")
        df = df.reindex(df.index.union(full_date_range)).sort_index()
        backtest_mask = (df.index >= backtest_start_ts) & (df.index <= today)
        missing_dates = df.index[backtest_mask & df[price_col].isna()]
        for date in missing_dates:
            previous_price = None
            prev_date = date - pd.Timedelta(days=1)
            if prev_date in df.index and pd.notna(df.loc[prev_date, price_col]):
                previous_price = float(df.loc[prev_date, price_col])
            if date == today:
                price_usd = fetch_btc_price_robust(previous_price=previous_price)
            else:
                price_usd = fetch_btc_price_historical(date, previous_price=previous_price)
            if price_usd is not None:
                df.loc[date, price_col] = price_usd
            elif previous_price is not None:
                df.loc[date, price_col] = previous_price

        if mvrv_col in df.columns and today in df.index and pd.isna(df.loc[today, mvrv_col]):
            yesterday = today - pd.Timedelta(days=1)
            if yesterday in df.index and pd.notna(df.loc[yesterday, mvrv_col]):
                df.loc[today, mvrv_col] = df.loc[yesterday, mvrv_col]
                logging.info(
                    "Used yesterday's MVRV value (%s) for %s",
                    f"{df.loc[yesterday, mvrv_col]:.4f}",
                    today.date(),
                )
            else:
                logging.warning(
                    "Could not find valid MVRV for %s. Yesterday (%s) not available "
                    "or also missing MVRV.",
                    today.date(),
                    yesterday.date(),
                )

        remaining_missing = df.loc[backtest_start_ts:today, price_col].isnull()
        if remaining_missing.any():
            num_missing = int(remaining_missing.sum())
            first_missing = df.loc[backtest_start_ts:today][remaining_missing].index.min().date()
            raise AssertionError(
                "Critical error: "
                f"{num_missing} dates still missing BTC-USD prices. "
                f"First missing: {first_missing}"
            )
        return df
