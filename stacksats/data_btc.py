"""BTC-only data provider services."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

class DataLoadError(RuntimeError):
    """Raised when BTC source data cannot be loaded safely."""


def _is_cache_usable(
    csv_bytes: bytes,
    backtest_start: pd.Timestamp,
    today: pd.Timestamp,
    *,
    target_end: pd.Timestamp | None = None,
    require_recent: bool = True,
) -> bool:
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
        window_end = (target_end or today).normalize()

        if require_recent and latest_date < (today - pd.Timedelta(days=3)):
            return False

        if latest_date < window_end:
            return False

        in_window = (cached_df["time"] >= backtest_start) & (cached_df["time"] <= window_end)
        return bool(cached_df.loc[in_window, "PriceUSD"].notna().any())
    except Exception:
        return False


@dataclass
class BTCDataProvider:
    """BTC-only data provider with cache/fetch/gap-fill behavior."""

    cache_dir: str | None = "~/.stacksats/cache"
    max_age_hours: int = 24
    clock: Callable[[], pd.Timestamp] = pd.Timestamp.now

    def load(self, *, backtest_start: str = "2018-01-01", end_date: str | None = None) -> pd.DataFrame:
        url = "https://raw.githubusercontent.com/coinmetrics/data/refs/heads/master/csv/btc.csv"
        use_cache = self.cache_dir is not None

        logging.info("Loading CoinMetrics BTC data...")
        csv_bytes: bytes
        cache_path: Path | None = None
        now = self.clock()
        today = now.normalize()
        backtest_start_ts = pd.to_datetime(backtest_start)
        target_end = pd.to_datetime(end_date).normalize() if end_date is not None else today
        if pd.isna(target_end):
            raise ValueError(f"Invalid end_date value: {end_date!r}")
        target_end = min(target_end, today)
        if target_end < backtest_start_ts:
            raise ValueError(
                "end_date must be on or after backtest_start. "
                f"Received backtest_start={backtest_start_ts.date()} "
                f"and end_date={target_end.date()}."
            )
        past_only_window = target_end < today

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
                try:
                    cached_bytes = cache_path.read_bytes()
                except OSError:
                    cached_bytes = b""

                cache_usable = _is_cache_usable(
                    cached_bytes,
                    backtest_start_ts,
                    today,
                    target_end=target_end,
                    require_recent=not past_only_window,
                )

                if past_only_window:
                    if cache_usable:
                        csv_bytes = cached_bytes
                    else:
                        raise DataLoadError(
                            "Past-only backtest requested but local CoinMetrics cache "
                            "does not cover the requested window. "
                            f"Requested end_date={target_end.date()}."
                        )
                else:
                    if age_hours <= self.max_age_hours and cache_usable:
                        csv_bytes = cached_bytes
                    else:
                        csv_bytes = _download_csv(cache_available=True)
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            cache_path.write_bytes(csv_bytes)
                        except OSError as exc:
                            logging.warning("Could not write cache file %s: %s", cache_path, exc)
            else:
                if past_only_window:
                    raise DataLoadError(
                        "Past-only backtest requested but no local CoinMetrics cache file "
                        "was found. Set STACKSATS_COINMETRICS_CSV/cache first or run with "
                        "an end_date that includes today."
                    )
                csv_bytes = _download_csv(cache_available=False)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    cache_path.write_bytes(csv_bytes)
                except OSError as exc:
                    logging.warning("Could not write cache file %s: %s", cache_path, exc)
        else:
            if past_only_window:
                raise DataLoadError(
                    "Past-only backtest requested with cache disabled (cache_dir=None). "
                    "Enable cache_dir or provide in-memory btc_df."
                )
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
        df[price_col] = df["PriceUSD"]

        if df.index.empty:
            raise DataLoadError("CoinMetrics CSV contained no rows.")

        valid_price_index = df.index[df[price_col].notna()]
        if len(valid_price_index) == 0:
            raise DataLoadError("CoinMetrics CSV contains no valid PriceUSD values.")
        latest_price_date = valid_price_index.max().normalize()
        available_end = min(target_end, latest_price_date)
        if available_end < backtest_start_ts:
            raise DataLoadError(
                "CoinMetrics CSV does not cover the requested start date. "
                f"Requested start={backtest_start_ts.date()}, "
                f"latest available price={latest_price_date.date()}."
            )

        window = df.loc[backtest_start_ts:available_end].copy()
        if window.empty:
            raise DataLoadError(
                "No CoinMetrics rows available in the requested backtest window."
            )

        missing_price = window[price_col].isna()
        if missing_price.any():
            first_missing = window.index[missing_price][0].date()
            raise DataLoadError(
                "CoinMetrics CSV has missing PriceUSD values in the requested window. "
                f"First missing date: {first_missing}."
            )

        expected_index = pd.date_range(start=window.index.min(), end=window.index.max(), freq="D")
        if not window.index.equals(expected_index):
            missing_dates = expected_index.difference(window.index)
            first_missing = missing_dates[0].date()
            raise DataLoadError(
                "CoinMetrics CSV has missing dates in the requested window. "
                f"First missing date: {first_missing}."
            )

        return window
