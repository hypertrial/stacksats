"""Shared test helpers and constants for validation tests."""

from datetime import datetime, timedelta

import polars as pl

# Constants
FLOAT_TOLERANCE = 1e-12
WEIGHT_SUM_TOLERANCE = 1e-6  # Tolerance for weight sum validation
PRICE_COL = "price_usd"
DATE_COLS = ["start_date", "end_date", "date"]
PRIMARY_KEY_COLS = ["day_index", "start_date", "end_date", "date"]

# Sample data bounds
SAMPLE_START = "2024-01-01"
SAMPLE_END = "2025-12-31"


def pl_date_range(start: str, end: str) -> pl.Series:
    """Polars datetime series from start to end (inclusive)."""
    from stacksats.prelude import date_range_series

    return date_range_series(start, end)


def daily_dates(start: str, periods: int) -> list[datetime]:
    """Return a Python list of daily datetimes."""
    start_dt = datetime.strptime(start[:10], "%Y-%m-%d")
    return [start_dt + timedelta(days=offset) for offset in range(periods)]


def btc_frame(
    *,
    start: str = "2024-01-01",
    days: int = 365,
    include_mvrv: bool = True,
    price_start: float = 10000.0,
    price_step: float = 25.0,
) -> pl.DataFrame:
    """Build a canonical Polars BTC dataframe with a date column."""
    dates = daily_dates(start, days)
    data: dict[str, object] = {
        "date": dates,
        "price_usd": [price_start + (offset * price_step) for offset in range(days)],
    }
    if include_mvrv:
        data["mvrv"] = [1.0 + (offset / max(days, 1)) for offset in range(days)]
    return pl.DataFrame(data)


def weight_frame(values: list[float], *, start: str = "2024-01-01") -> pl.DataFrame:
    """Build a canonical weight dataframe for validation tests."""
    return pl.DataFrame({"date": daily_dates(start, len(values)), "weight": values})


def iter_date_ranges(df: pl.DataFrame):
    """Iterate over (start_date, end_date) groups in a DataFrame.

    Yields ((start_date, end_date), group_df) for each unique (start_date, end_date).
    """
    for row in df.unique(["start_date", "end_date"]).iter_rows(named=True):
        start, end = row["start_date"], row["end_date"]
        group = df.filter(
            (pl.col("start_date") == start) & (pl.col("end_date") == end)
        )
        yield (start, end), group


def get_range_days(start: str, end: str) -> int:
    """Calculate number of days in a date range (inclusive)."""
    start_dt = datetime.strptime(start[:10], "%Y-%m-%d")
    end_dt = datetime.strptime(end[:10], "%Y-%m-%d")
    return (end_dt - start_dt).days + 1
