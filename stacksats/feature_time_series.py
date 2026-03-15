"""Feature time-series input object for strategy computation."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import polars as pl

DATE_COL = "date"


def _validate_date_column(df: pl.DataFrame) -> None:
    """Ensure date column exists, is sorted, unique, and has no nulls."""
    if DATE_COL not in df.columns:
        raise ValueError(f"FeatureTimeSeries must have a '{DATE_COL}' column.")
    dates = df[DATE_COL]
    if dates.null_count() > 0:
        raise ValueError(f"Column '{DATE_COL}' must not contain nulls.")
    if dates.is_duplicated().any():
        raise ValueError(f"Column '{DATE_COL}' must not contain duplicates.")
    sorted_dates = df.sort(DATE_COL)[DATE_COL]
    if not dates.equals(sorted_dates):
        raise ValueError(f"Column '{DATE_COL}' must be sorted ascending.")


def _validate_no_future_data(df: pl.DataFrame, as_of_date: dt.datetime) -> None:
    """Ensure no row has date after as_of_date."""
    max_date = df[DATE_COL].max()
    if max_date is None:
        return
    if isinstance(max_date, dt.datetime):
        max_d = max_date.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        max_d = dt.datetime.fromisoformat(str(max_date)[:10])
    ref = as_of_date.replace(hour=0, minute=0, second=0, microsecond=0)
    if as_of_date.tzinfo is not None:
        ref = as_of_date.astimezone(dt.timezone.utc).replace(tzinfo=None)
    if max_d > ref:
        raise ValueError(
            "FeatureTimeSeries must not contain data after as_of_date "
            f"(as_of_date={as_of_date!s}, max date in data={max_date!s})."
        )


def _validate_finite_numeric(df: pl.DataFrame, columns: tuple[str, ...]) -> None:
    """Ensure specified numeric columns contain only finite values."""
    for col in columns:
        if col not in df.columns:
            continue
        s = df[col]
        if s.null_count() > 0:
            raise ValueError(f"Column '{col}' must not contain nulls.")
        if s.dtype in (pl.Float32, pl.Float64) and not s.is_finite().all():
            raise ValueError(f"Column '{col}' must contain finite values (no inf).")


@dataclass(frozen=True, slots=True)
class FeatureTimeSeries:
    """
    Immutable feature time-series input to a strategy.

    Polars-backed: datetime as a column (name 'date'), each other column is a feature.
    Validates schema (required columns) and time-series invariants (sorted, unique
    dates; optional no-future-data and finite-numeric checks).
    """

    _frame: pl.DataFrame

    def __post_init__(self) -> None:
        if not isinstance(self._frame, pl.DataFrame):
            raise TypeError("FeatureTimeSeries._frame must be a polars DataFrame.")
        _validate_date_column(self._frame)

    @classmethod
    def from_dataframe(
        cls,
        df: pl.DataFrame,
        *,
        required_columns: tuple[str, ...] = (),
        as_of_date: dt.datetime | None = None,
        require_finite: tuple[str, ...] | None = None,
    ) -> FeatureTimeSeries:
        """
        Build a FeatureTimeSeries from a Polars DataFrame.

        The DataFrame must have a 'date' column (sorted, unique, no nulls).
        Other columns are feature columns.

        Parameters
        ----------
        df : polars DataFrame
            Feature matrix with 'date' column and feature columns.
        required_columns : tuple of str
            Column names that must be present (excluding 'date').
        as_of_date : datetime or None
            If set, validates that no row has date after this date (no forward-looking).
        require_finite : tuple of str or None
            If set, these numeric columns must contain only finite values.
        """
        if not isinstance(df, pl.DataFrame):
            raise TypeError("FeatureTimeSeries.from_dataframe requires a polars DataFrame.")
        if df.is_empty():
            pl_df = pl.DataFrame(schema={DATE_COL: pl.Datetime("us")})
        else:
            pl_df = df.clone()
            if DATE_COL not in pl_df.columns and pl_df.width > 0:
                pl_df = pl_df.rename({pl_df.columns[0]: DATE_COL})
        for col in required_columns:
            if col not in pl_df.columns:
                raise ValueError(
                    f"FeatureTimeSeries missing required column: {col!r}. "
                    f"Available: {tuple(pl_df.columns)!r}."
                )
        _validate_date_column(pl_df)
        if as_of_date is not None and not pl_df.is_empty():
            _validate_no_future_data(pl_df, as_of_date)
        if require_finite:
            _validate_finite_numeric(pl_df, tuple(require_finite))
        return cls(_frame=pl_df)

    @property
    def data(self) -> pl.DataFrame:
        """Return the underlying Polars DataFrame (read-only; do not mutate)."""
        return self._frame

    @property
    def columns(self) -> tuple[str, ...]:
        """Return feature column names (including 'date')."""
        return tuple(self._frame.columns)

    @property
    def row_count(self) -> int:
        """Return number of rows."""
        return self._frame.height

    def validate_schema(self, required_columns: tuple[str, ...]) -> None:
        """Raise if any required column is missing."""
        for col in required_columns:
            if col not in self._frame.columns:
                raise ValueError(
                    f"FeatureTimeSeries missing required column: {col!r}. "
                    f"Available: {self.columns!r}."
                )
