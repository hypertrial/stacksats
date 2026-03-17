"""BTC data provider backed by local BRK-generated parquet metrics."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import polars as pl

from .data_setup import resolve_runtime_parquet

PARQUET_DEFAULT_PATH = "./bitcoin_analytics.parquet"
PARQUET_ENV_VAR = "STACKSATS_ANALYTICS_PARQUET"
DATE_COL = "date"


class DataLoadError(RuntimeError):
    """Raised when BTC source data cannot be loaded safely."""


def _resolve_parquet_path(path_override: str | None) -> Path:
    try:
        return resolve_runtime_parquet(path_override).path
    except FileNotFoundError as exc:
        raise DataLoadError(str(exc)) from exc


def _norm_dt(value: dt.datetime) -> dt.datetime:
    out = value
    if value.tzinfo is not None:
        out = value.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return out.replace(hour=0, minute=0, second=0, microsecond=0)


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


def _daily_date_expr(dtype: pl.DataType, *, column: str = DATE_COL) -> pl.Expr:
    if dtype == pl.Utf8:
        base = pl.col(column).str.to_datetime(strict=False)
    elif dtype == pl.Date:
        base = pl.col(column).cast(pl.Datetime)
    else:
        base = pl.col(column).cast(pl.Datetime, strict=False)
    return base.dt.replace_time_zone(None).dt.truncate("1d")


def _scan_btc_from_parquet(path: Path) -> pl.LazyFrame:
    frame = pl.scan_parquet(path)
    schema = frame.collect_schema()
    names = schema.names()
    if DATE_COL not in names:
        if names and names[0].lower() in ("date", "index", "timestamp"):
            frame = frame.rename({names[0]: DATE_COL})
            schema = frame.collect_schema()
            names = schema.names()
        else:
            raise DataLoadError("Parquet must have a 'date' column.")
    if "price_usd" not in names:
        raise DataLoadError("Runtime parquet must contain a 'price_usd' column.")
    frame = frame.with_columns(
        _daily_date_expr(schema[DATE_COL]).alias(DATE_COL),
        pl.col("price_usd").cast(pl.Float64, strict=False).alias("price_usd"),
        *(
            [pl.col("mvrv").cast(pl.Float64, strict=False).alias("mvrv")]
            if "mvrv" in names
            else []
        ),
    )
    return frame.unique(subset=[DATE_COL], keep="last").sort(DATE_COL)


@dataclass
class BTCDataProvider:
    """BTC-only provider using BRK-generated local parquet metrics."""

    parquet_path: str | None = None
    clock: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.timezone.utc)
    # Retained for API compatibility; runtime now defaults to available parquet horizon.
    max_staleness_days: int = 3

    def load(
        self,
        *,
        backtest_start: str = "2018-01-01",
        end_date: str | None = None,
        include_warmup: bool = True,
    ) -> pl.DataFrame:
        """Return an eager runtime BTC frame with strict source validation."""
        window = self.load_lazy(
            backtest_start=backtest_start,
            end_date=end_date,
            include_warmup=include_warmup,
        ).collect()
        target_end = window[DATE_COL].max()
        if target_end is None:
            raise DataLoadError("No BRK rows available in requested backtest window.")
        _require_daily_index(
            window,
            backtest_start_ts=_norm_dt(dt.datetime.strptime(backtest_start[:10], "%Y-%m-%d")),
            target_end=_norm_dt(target_end),
        )
        return window

    def load_lazy(
        self,
        *,
        backtest_start: str = "2018-01-01",
        end_date: str | None = None,
        include_warmup: bool = True,
    ) -> pl.LazyFrame:
        """Return a lazy runtime BTC frame with strict source validation."""
        now = _norm_dt(self.clock())
        backtest_start_ts = _norm_dt(dt.datetime.strptime(backtest_start[:10], "%Y-%m-%d"))
        if end_date is not None:
            try:
                requested_end = _norm_dt(dt.datetime.strptime(end_date[:10], "%Y-%m-%d"))
            except Exception as exc:
                raise ValueError(f"Invalid end_date value: {end_date!r}") from exc
            if requested_end > now:
                requested_end = now
        else:
            requested_end = None

        frame = _scan_btc_from_parquet(_resolve_parquet_path(self.parquet_path))
        stats = frame.select(
            pl.len().alias("row_count"),
            pl.when(pl.col("price_usd").is_finite())
            .then(pl.col(DATE_COL))
            .otherwise(None)
            .max()
            .alias("latest_valid_price_date"),
        ).collect().row(0, named=True)
        if int(stats["row_count"]) == 0:
            raise DataLoadError("Parquet file is empty.")
        latest_price_date = stats["latest_valid_price_date"]
        if latest_price_date is None:
            raise DataLoadError("BRK data contains no valid price_usd values.")
        if isinstance(latest_price_date, dt.datetime):
            latest_dt = latest_price_date
        else:
            latest_dt = dt.datetime.fromisoformat(str(latest_price_date)[:10])
        latest_dt = _norm_dt(latest_dt)
        # Default to available parquet horizon when no explicit end date is provided.
        target_end = requested_end if requested_end is not None else min(now, latest_dt)

        if target_end < backtest_start_ts:
            raise ValueError(
                "end_date must be on or after backtest_start. "
                f"Received backtest_start={backtest_start_ts.date()} and end_date={target_end.date()}."
            )
        if latest_dt < target_end:
            raise DataLoadError(
                "BRK data does not cover requested end_date. "
                f"Latest available={latest_dt.date()}, requested={target_end.date()}."
            )

        if include_warmup:
            window = frame.filter(pl.col(DATE_COL) <= target_end)
        else:
            window = frame.filter(
                (pl.col(DATE_COL) >= backtest_start_ts) & (pl.col(DATE_COL) <= target_end)
            )
        validation = window.select(
            pl.len().filter(pl.col(DATE_COL) >= backtest_start_ts).alias("scored_rows"),
            pl.when(
                pl.col("price_usd").is_null() | ~pl.col("price_usd").is_finite()
            )
            .then(pl.col(DATE_COL))
            .otherwise(None)
            .min()
            .alias("first_invalid_price_date"),
        ).collect().row(0, named=True)
        if int(validation["scored_rows"]) == 0:
            raise DataLoadError("No BRK rows available in requested backtest window.")
        first_invalid = validation["first_invalid_price_date"]
        if first_invalid is not None:
            first_str = str(first_invalid)[:10]
            raise DataLoadError(
                "BRK data has missing price_usd values in requested window. "
                f"First missing date: {first_str}."
            )
        return window
