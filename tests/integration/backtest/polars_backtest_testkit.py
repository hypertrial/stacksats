"""Shared Polars-native helpers for backtest integration tests."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable

import numpy as np
import polars as pl

from tests.test_helpers import btc_frame, daily_dates


def dt_at(value: str | dt.datetime) -> dt.datetime:
    """Normalize a date-like value to naive midnight datetime."""
    if isinstance(value, dt.datetime):
        out = value
    else:
        out = dt.datetime.strptime(str(value)[:10], "%Y-%m-%d")
    if out.tzinfo is not None:
        out = out.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return out.replace(hour=0, minute=0, second=0, microsecond=0)


def day_range(start: str | dt.datetime, end: str | dt.datetime) -> list[dt.datetime]:
    """Return daily datetimes from start through end inclusive."""
    start_dt = dt_at(start)
    end_dt = dt_at(end)
    days = (end_dt - start_dt).days
    return [start_dt + dt.timedelta(days=offset) for offset in range(days + 1)]


def slice_dates(
    df: pl.DataFrame,
    start: str | dt.datetime,
    end: str | dt.datetime,
) -> pl.DataFrame:
    """Return a date-bounded slice of a canonical dataframe."""
    start_dt = dt_at(start)
    end_dt = dt_at(end)
    return df.filter((pl.col("date") >= start_dt) & (pl.col("date") <= end_dt))


def frame_has_date(df: pl.DataFrame, value: str | dt.datetime) -> bool:
    """Check whether a canonical dataframe contains a given date."""
    target = dt_at(value)
    return df.filter(pl.col("date") == target).height > 0


def make_btc_df(
    *,
    start: str = "2020-01-01",
    days: int = 2500,
    price_start: float = 10000.0,
    price_step: float = 25.0,
    mvrv_start: float = 1.0,
    mvrv_end: float = 2.0,
) -> pl.DataFrame:
    """Build a canonical synthetic BTC dataframe with price and MVRV columns."""
    df = btc_frame(
        start=start,
        days=days,
        include_mvrv=False,
        price_start=price_start,
        price_step=price_step,
    )
    mvrv = np.linspace(mvrv_start, mvrv_end, num=days)
    return df.with_columns(
        pl.Series("mvrv", mvrv),
        pl.col("price_usd").alias("PriceUSD"),
    )


def make_btc_df_from_prices(
    prices: Iterable[float],
    *,
    start: str = "2020-01-01",
    mvrv_low: float = 0.8,
    mvrv_high: float = 2.2,
) -> pl.DataFrame:
    """Build a canonical BTC dataframe from an explicit daily price path."""
    price_list = [float(price) for price in prices]
    days = len(price_list)
    return pl.DataFrame({
        "date": daily_dates(start, days),
        "price_usd": price_list,
        "PriceUSD": price_list,
        "mvrv": np.linspace(mvrv_low, mvrv_high, num=days),
    })


def normalize_weight_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Return a canonical sorted weight frame with datetime dates."""
    if df.is_empty():
        return pl.DataFrame(schema={"date": pl.Datetime("us"), "weight": pl.Float64})
    date_expr = (
        pl.col("date").str.to_datetime().dt.replace_time_zone(None)
        if df["date"].dtype == pl.Utf8
        else pl.col("date").cast(pl.Datetime).dt.replace_time_zone(None)
    )
    return df.with_columns(date_expr.alias("date")).sort("date").select(["date", "weight"])


def weight_lookup(df: pl.DataFrame) -> dict[dt.datetime, float]:
    """Map canonical dates to weights."""
    normalized = normalize_weight_frame(df)
    return dict(zip(normalized["date"].to_list(), normalized["weight"].to_list(), strict=True))
