"""BTC data provider backed by local BRK-generated parquet metrics."""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import polars as pl

PARQUET_DEFAULT_PATH = "./bitcoin_analytics.parquet"
PARQUET_ENV_VAR = "STACKSATS_ANALYTICS_PARQUET"
DATE_COL = "date"


class DataLoadError(RuntimeError):
    """Raised when BTC source data cannot be loaded safely."""


def _resolve_parquet_path(path_override: str | None) -> Path:
    path = Path(os.getenv(PARQUET_ENV_VAR) or path_override or PARQUET_DEFAULT_PATH).expanduser()
    if not path.exists():
        raise DataLoadError(
            f"Parquet file not found at '{path}'. Set {PARQUET_ENV_VAR} or provide parquet_path."
        )
    return path


def _norm_dt(value: dt.datetime) -> dt.datetime:
    out = value.replace(hour=0, minute=0, second=0, microsecond=0)
    if value.tzinfo is not None:
        out = value.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return out


def _require_daily_index(
    df: pl.DataFrame,
    *,
    backtest_start_ts: dt.datetime,
    target_end: dt.datetime,
) -> None:
    if DATE_COL not in df.columns:
        raise DataLoadError("DataFrame must have a 'date' column.")
    dates = df[DATE_COL]
    if dates.null_count() > 0:
        raise DataLoadError("Date column must not contain nulls.")
    start_ts = _norm_dt(backtest_start_ts)
    end_ts = _norm_dt(target_end)
    min_d = dates.min()
    max_d = dates.max()
    if min_d is None or max_d is None:
        raise DataLoadError("No valid dates in DataFrame.")
    min_dt = min_d if isinstance(min_d, dt.datetime) else dt.datetime.fromisoformat(str(min_d)[:10])
    max_dt = max_d if isinstance(max_d, dt.datetime) else dt.datetime.fromisoformat(str(max_d)[:10])
    if min_dt > start_ts or max_dt < end_ts:
        raise DataLoadError(
            "BRK parquet data does not cover requested window. "
            f"Data: {min_dt.date()} to {max_dt.date()}, requested: {start_ts.date()} to {end_ts.date()}."
        )
    # Check for missing dates in range
    expected_count = (end_ts - start_ts).days + 1
    in_range = df.filter(
        (pl.col(DATE_COL) >= start_ts) & (pl.col(DATE_COL) <= end_ts)
    )
    if in_range.height != expected_count:
        raise DataLoadError(
            "BRK parquet data has missing dates in requested window. "
            f"Expected {expected_count} days, got {in_range.height}."
        )


def _load_btc_from_parquet(path: Path) -> pl.DataFrame:
    frame = pl.read_parquet(path)
    if frame.is_empty():
        raise DataLoadError("Parquet file is empty.")
    if DATE_COL not in frame.columns:
        # Try first column as date or index
        cols = frame.columns
        if len(cols) >= 1 and cols[0].lower() in ("date", "index", "timestamp"):
            frame = frame.rename({cols[0]: DATE_COL})
        else:
            raise DataLoadError("Parquet must have a 'date' column.")
    if frame[DATE_COL].dtype == pl.Utf8:
        frame = frame.with_columns(pl.col(DATE_COL).str.to_datetime())
    if "Datetime" in str(frame[DATE_COL].dtype):
        frame = frame.with_columns(
            pl.col(DATE_COL).dt.replace_time_zone(None).dt.truncate("1d")
        )
    frame = frame.unique(subset=[DATE_COL], keep="last").sort(DATE_COL)
    if "price_usd" not in frame.columns:
        raise DataLoadError("Parquet must contain a 'price_usd' column.")
    frame = frame.with_columns(pl.col("price_usd").cast(pl.Float64, strict=False))
    if "mvrv" in frame.columns:
        frame = frame.with_columns(pl.col("mvrv").cast(pl.Float64, strict=False))
    return frame


@dataclass
class BTCDataProvider:
    """BTC-only provider using BRK-generated local parquet metrics."""

    parquet_path: str | None = None
    clock: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.timezone.utc)
    max_staleness_days: int = 3

    def load(
        self,
        *,
        backtest_start: str = "2018-01-01",
        end_date: str | None = None,
    ) -> pl.DataFrame:
        now = _norm_dt(self.clock())
        backtest_start_ts = _norm_dt(dt.datetime.strptime(backtest_start[:10], "%Y-%m-%d"))
        if end_date is not None:
            try:
                target_end = _norm_dt(dt.datetime.strptime(end_date[:10], "%Y-%m-%d"))
            except Exception as exc:
                raise ValueError(f"Invalid end_date value: {end_date!r}") from exc
        else:
            target_end = now
        if target_end > now:
            target_end = now
        if target_end < backtest_start_ts:
            raise ValueError(
                "end_date must be on or after backtest_start. "
                f"Received backtest_start={backtest_start_ts.date()} and end_date={target_end.date()}."
            )

        frame = _load_btc_from_parquet(_resolve_parquet_path(self.parquet_path))
        if "price_usd" not in frame.columns:
            raise DataLoadError("Required price_usd series missing from BRK data.")

        valid_price = frame.filter(pl.col("price_usd").is_finite())
        if valid_price.is_empty():
            raise DataLoadError("BRK data contains no valid price_usd values.")
        latest_price_date = valid_price[DATE_COL].max()
        if isinstance(latest_price_date, dt.datetime):
            latest_dt = latest_price_date
        else:
            latest_dt = dt.datetime.fromisoformat(str(latest_price_date)[:10])
        latest_dt = _norm_dt(latest_dt)
        if latest_dt < target_end:
            raise DataLoadError(
                "BRK data does not cover requested end_date. "
                f"Latest available={latest_dt.date()}, requested={target_end.date()}."
            )
        if latest_dt < (now - dt.timedelta(days=int(self.max_staleness_days))):
            raise DataLoadError(
                "BRK data is stale for runtime usage. "
                f"Latest available={latest_dt.date()}, now={now.date()}."
            )

        window = frame.filter(
            (pl.col(DATE_COL) >= backtest_start_ts) & (pl.col(DATE_COL) <= target_end)
        )
        if window.is_empty():
            raise DataLoadError("No BRK rows available in requested backtest window.")
        if window.filter(pl.col("price_usd").is_null()).height > 0:
            first_null = window.filter(pl.col("price_usd").is_null())[DATE_COL][0]
            first_str = str(first_null)[:10] if first_null is not None else "unknown"
            raise DataLoadError(
                "BRK data has missing price_usd values in requested window. "
                f"First missing date: {first_str}."
            )
        _require_daily_index(
            window,
            backtest_start_ts=backtest_start_ts,
            target_end=target_end,
        )
        return window
