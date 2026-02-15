"""Shared test helpers and constants for validation tests."""

import pandas as pd

# Constants
FLOAT_TOLERANCE = 1e-12
WEIGHT_SUM_TOLERANCE = 1e-6  # Tolerance for weight sum validation
PRICE_COL = "PriceUSD_coinmetrics"
DATE_COLS = ["start_date", "end_date", "date"]
PRIMARY_KEY_COLS = ["day_index", "start_date", "end_date", "date"]

# Sample data bounds
SAMPLE_START = "2024-01-01"
SAMPLE_END = "2025-12-31"


def iter_date_ranges(df: pd.DataFrame):
    """Iterate over (start_date, end_date) groups in a DataFrame."""
    return df.groupby(["start_date", "end_date"])


def get_range_days(start: str, end: str) -> int:
    """Calculate number of days in a date range (inclusive)."""
    return (pd.Timestamp(end) - pd.Timestamp(start)).days + 1
