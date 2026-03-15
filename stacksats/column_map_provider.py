"""Column-mapping data provider for flexible data ingestion without a BRK parquet file.

Allows users to supply any Polars DataFrame by declaring a column map that
maps library-canonical column names (e.g. ``price_usd``, ``mvrv``) to the
actual column names in their DataFrame.

Example usage::

    import polars as pl
    from stacksats.column_map_provider import ColumnMapDataProvider
    from stacksats.runner import StrategyRunner

    df = pl.read_csv("my_data.csv").with_columns(pl.col("date").str.to_datetime())
    runner = StrategyRunner(
        data_provider=ColumnMapDataProvider(
            df=df,
            column_map={"price_usd": "Close", "mvrv": "MVRV_Ratio"},
        )
    )
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import polars as pl

#: The only column truly required by the strategy framework.
_REQUIRED_COLUMNS: tuple[str, ...] = ("price_usd",)
DATE_COL = "date"


class ColumnMapError(ValueError):
    """Raised when the column map or supplied DataFrame is invalid."""


@dataclass
class ColumnMapDataProvider:
    """BTC data provider backed by any user-supplied Polars DataFrame.

    Parameters
    ----------
    df:
        A Polars DataFrame with a ``date`` column (or column that can be
        interpreted as date) at daily frequency (or finer — it will be
        normalized to daily).
    column_map:
        Mapping from **library column names** → **user DataFrame column names**.
        Only the columns you need map to; unmapped library columns that already
        exist in df by their canonical name are used as-is.

        Example::

            {"price_usd": "Close", "mvrv": "MVRV"}

    Raises
    ------
    ColumnMapError
        If required library columns cannot be resolved from the DataFrame after
        applying the map.
    """

    df: pl.DataFrame
    column_map: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Public interface (same as BTCDataProvider)
    # ------------------------------------------------------------------ #

    def load(
        self,
        *,
        backtest_start: str = "2018-01-01",
        end_date: str | None = None,
    ) -> pl.DataFrame:
        """Return the canonical BTC DataFrame for the requested window.

        Applies the column map, enforces a daily date column, and slices
        to ``[backtest_start, end_date]``.
        """
        frame = self._apply_column_map(self.df)
        frame = self._to_daily_date(frame)
        self._validate_required_columns(frame)

        start_ts = dt.datetime.strptime(backtest_start[:10], "%Y-%m-%d")
        if end_date is not None:
            try:
                end_ts = dt.datetime.strptime(end_date[:10], "%Y-%m-%d")
            except Exception as exc:
                raise ValueError(f"Invalid end_date value: {end_date!r}") from exc
        else:
            end_ts = dt.datetime.now(dt.timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            end_ts = end_ts.replace(tzinfo=None)  # naive for comparison

        if end_ts < start_ts:
            raise ValueError(
                "end_date must be on or after backtest_start. "
                f"Received backtest_start={start_ts.date()} and end_date={end_ts.date()}."
            )

        window = frame.filter(
            (pl.col(DATE_COL) >= start_ts) & (pl.col(DATE_COL) <= end_ts)
        )
        if window.is_empty():
            raise ColumnMapError(
                "No rows available in the requested backtest window "
                f"[{start_ts.date()}, {end_ts.date()}]."
            )

        invalid_price = pl.col("price_usd").is_null()
        if window["price_usd"].dtype in (pl.Float32, pl.Float64):
            invalid_price = invalid_price | ~pl.col("price_usd").is_finite()
        invalid_rows = window.filter(invalid_price)
        if invalid_rows.height > 0:
            first_missing = str(invalid_rows[DATE_COL][0])[:10]
            raise ColumnMapError(
                f"Missing price_usd values in window. First missing date: {first_missing}."
            )

        return window

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _apply_column_map(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return a copy of *df* with columns renamed per ``column_map``."""
        if not self.column_map:
            return df.clone()

        rename: dict[str, str] = {}
        for lib_col, user_col in self.column_map.items():
            if user_col not in df.columns:
                raise ColumnMapError(
                    f"column_map references '{user_col}' which is not present in the "
                    f"DataFrame. Available columns: {list(df.columns)}"
                )
            if user_col != lib_col:
                rename[user_col] = lib_col

        return df.rename(rename)

    @staticmethod
    def _to_daily_date(df: pl.DataFrame) -> pl.DataFrame:
        """Ensure df has a normalised daily date column."""
        if DATE_COL not in df.columns:
            raise ColumnMapError(
                f"DataFrame must have a '{DATE_COL}' column. "
                "Rename your date column to 'date' or add it via with_columns."
            )
        col = df[DATE_COL]
        if col.dtype == pl.Utf8:
            df = df.with_columns(pl.col(DATE_COL).str.to_datetime())
        if "Datetime" in str(df[DATE_COL].dtype):
            df = df.with_columns(
                pl.col(DATE_COL).dt.replace_time_zone(None).dt.truncate("1d")
            )
        df = df.unique(subset=[DATE_COL], keep="last").sort(DATE_COL)
        return df

    @staticmethod
    def _validate_required_columns(df: pl.DataFrame) -> None:
        missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ColumnMapError(
                f"Required library columns are missing after applying column_map: {missing}. "
                "Use column_map={{\"price_usd\": \"<your price column>\"}} to map them."
            )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ColumnMapDataProvider(rows={self.df.height}, "
            f"column_map={self.column_map!r})"
        )
