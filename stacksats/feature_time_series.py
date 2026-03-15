"""Feature time-series input object for strategy computation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
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


def _validate_no_future_data(df: pl.DataFrame, as_of_date: pd.Timestamp) -> None:
    """Ensure no row has date after as_of_date."""
    max_date = df[DATE_COL].max()
    if max_date is None:
        return
    if pd.Timestamp(max_date) > pd.Timestamp(as_of_date).normalize():
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
    def from_pandas(
        cls,
        df: pd.DataFrame,
        *,
        required_columns: tuple[str, ...] = (),
        as_of_date: pd.Timestamp | None = None,
        require_finite: tuple[str, ...] | None = None,
    ) -> FeatureTimeSeries:
        """
        Build a FeatureTimeSeries from a pandas DataFrame.

        The DataFrame must have a DatetimeIndex or a 'date' column; it will be
        converted to Polars with a 'date' column. If the index is a DatetimeIndex,
        it is reset into a column named 'date'.

        Parameters
        ----------
        df : pandas DataFrame
            Feature matrix (index or column = dates, columns = features).
        required_columns : tuple of str
            Column names that must be present (excluding 'date').
        as_of_date : timestamp or None
            If set, validates that no row has date after this date (no forward-looking).
        require_finite : tuple of str or None
            If set, these numeric columns must contain only finite values.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("FeatureTimeSeries.from_pandas requires a pandas DataFrame.")
        if df.empty:
            pl_df = pl.DataFrame(schema={DATE_COL: pl.Datetime("us")})
        else:
            pdf = df.copy()
            if isinstance(pdf.index, pd.DatetimeIndex):
                pdf = pdf.reset_index()
                first_col = pdf.columns[0]
                if first_col != DATE_COL:
                    if DATE_COL in pdf.columns:
                        pdf = pdf.drop(columns=[first_col])
                    else:
                        pdf = pdf.rename(columns={first_col: DATE_COL})
            elif DATE_COL not in pdf.columns:
                raise ValueError(
                    "FeatureTimeSeries.from_pandas: DataFrame must have "
                    "DatetimeIndex or a 'date' column."
                )
            pl_df = pl.from_pandas(pdf)
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

    def to_pandas(self) -> pd.DataFrame:
        """Return a pandas DataFrame with datetime index (date column becomes index)."""
        pdf = self._frame.to_pandas()
        if DATE_COL in pdf.columns:
            if pdf.empty:
                pdf = pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
            else:
                pdf = pdf.set_index(pd.to_datetime(pdf[DATE_COL])).drop(columns=[DATE_COL])
        return pdf

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
