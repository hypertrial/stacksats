"""Column-mapping data provider for flexible data ingestion without a BRK parquet file.

Allows users to supply any Pandas DataFrame by declaring a column map that
maps library-canonical column names (e.g. ``price_usd``, ``mvrv``) to the
actual column names in their DataFrame.

Example usage::

    import pandas as pd
    from stacksats.column_map_provider import ColumnMapDataProvider
    from stacksats.runner import StrategyRunner

    df = pd.read_csv("my_data.csv", index_col=0, parse_dates=True)
    runner = StrategyRunner(
        data_provider=ColumnMapDataProvider(
            df=df,
            column_map={"price_usd": "Close", "mvrv": "MVRV_Ratio"},
        )
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

#: The only column truly required by the strategy framework.
_REQUIRED_COLUMNS: tuple[str, ...] = ("price_usd",)


class ColumnMapError(ValueError):
    """Raised when the column map or supplied DataFrame is invalid."""


@dataclass
class ColumnMapDataProvider:
    """BTC data provider backed by any user-supplied Pandas DataFrame.

    Parameters
    ----------
    df:
        A Pandas DataFrame with a ``DatetimeIndex`` at daily frequency
        (or finer — it will be resampled to daily).
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

    df: pd.DataFrame
    column_map: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Public interface (same as BTCDataProvider)
    # ------------------------------------------------------------------ #

    def load(
        self,
        *,
        backtest_start: str = "2018-01-01",
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Return the canonical BTC DataFrame for the requested window.

        Applies the column map, enforces a daily DatetimeIndex, and slices
        to ``[backtest_start, end_date]``.
        """
        frame = self._apply_column_map(self.df)
        frame = self._to_daily_index(frame)
        self._validate_required_columns(frame)

        start_ts = pd.to_datetime(backtest_start).normalize()
        if end_date is not None:
            try:
                end_ts = pd.to_datetime(end_date).normalize()
            except Exception as exc:
                raise ValueError(f"Invalid end_date value: {end_date!r}") from exc
        else:
            end_ts = pd.Timestamp.now().normalize()

        if end_ts < start_ts:
            raise ValueError(
                "end_date must be on or after backtest_start. "
                f"Received backtest_start={start_ts.date()} and end_date={end_ts.date()}."
            )

        window = frame.loc[start_ts:end_ts].copy()
        if window.empty:
            raise ColumnMapError(
                "No rows available in the requested backtest window "
                f"[{start_ts.date()}, {end_ts.date()}]."
            )

        if window["price_usd"].isna().any():
            first_missing = window.index[window["price_usd"].isna()][0].strftime("%Y-%m-%d")
            raise ColumnMapError(
                f"Missing price_usd values in window. First missing date: {first_missing}."
            )

        return window

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _apply_column_map(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of *df* with columns renamed per ``column_map``."""
        if not self.column_map:
            return df.copy()

        # Build the pandas rename dict: user_col → library_col
        rename: dict[str, str] = {}
        for lib_col, user_col in self.column_map.items():
            if user_col not in df.columns:
                raise ColumnMapError(
                    f"column_map references '{user_col}' which is not present in the "
                    f"DataFrame. Available columns: {list(df.columns)}"
                )
            if user_col != lib_col:
                rename[user_col] = lib_col

        return df.rename(columns=rename).copy()

    @staticmethod
    def _to_daily_index(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure df has a normalised daily DatetimeIndex."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ColumnMapError(
                "DataFrame index must be a DatetimeIndex. "
                f"Got {type(df.index).__name__}."
            )
        frame = df.copy()
        frame.index = pd.DatetimeIndex(frame.index).normalize()
        # Deduplicate (keep last), sort ascending
        frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
        return frame

    @staticmethod
    def _validate_required_columns(df: pd.DataFrame) -> None:
        missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ColumnMapError(
                f"Required library columns are missing after applying column_map: {missing}. "
                "Use column_map={{\"price_usd\": \"<your price column>\"}} to map them."
            )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ColumnMapDataProvider(rows={len(self.df)}, "
            f"column_map={self.column_map!r})"
        )
