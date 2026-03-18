"""Tests for ColumnMapDataProvider and StrategyRunner.from_dataframe."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from stacksats.column_map_provider import ColumnMapDataProvider, ColumnMapError
from stacksats.runner import StrategyRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(
    *,
    start: str = "2018-01-01",
    periods: int = 1000,
    price_col: str = "price_usd",
    price: float = 30_000.0,
    extra_cols: dict | None = None,
) -> pl.DataFrame:
    start_d = dt.datetime.strptime(start[:10], "%Y-%m-%d")
    dates = [start_d + dt.timedelta(days=i) for i in range(periods)]
    data: dict[str, list] = {"date": dates, price_col: [price] * periods}
    if extra_cols:
        for k, v in extra_cols.items():
            data[k] = [v] * periods
    return pl.DataFrame(data)


# ---------------------------------------------------------------------------
# ColumnMapDataProvider unit tests
# ---------------------------------------------------------------------------


class TestColumnMapProviderHappyPath:
    def test_load_canonical_columns_no_map(self) -> None:
        """DataFrame with canonical column names loads without a map."""
        df = _make_df()
        provider = ColumnMapDataProvider(df=df)
        result = provider.load(backtest_start="2018-01-01", end_date="2020-01-01")
        assert "price_usd" in result.columns
        assert "date" in result.columns
        assert str(result["date"].min())[:10] == "2018-01-01"
        assert str(result["date"].max())[:10] == "2020-01-01"

    def test_load_with_column_map(self) -> None:
        """DataFrame with non-canonical price column renamed via column_map."""
        df = _make_df(price_col="Close")
        provider = ColumnMapDataProvider(df=df, column_map={"price_usd": "Close"})
        result = provider.load(backtest_start="2018-01-01", end_date="2020-01-01")
        assert "price_usd" in result.columns
        assert "Close" not in result.columns

    def test_load_with_mvrv_column_map(self) -> None:
        """DataFrame with both price and mvrv mapped from user columns."""
        df = _make_df(
            price_col="BTC_Close",
            extra_cols={"MVRV_Ratio": 2.5},
        )
        provider = ColumnMapDataProvider(
            df=df,
            column_map={"price_usd": "BTC_Close", "mvrv": "MVRV_Ratio"},
        )
        result = provider.load(backtest_start="2018-01-01", end_date="2020-01-01")
        assert "price_usd" in result.columns
        assert "mvrv" in result.columns

    def test_load_lazy_matches_eager(self) -> None:
        df = _make_df(start="2015-01-01", periods=3000, extra_cols={"mvrv": 1.5})
        provider = ColumnMapDataProvider(df=df)

        eager = provider.load(backtest_start="2018-06-01", end_date="2018-06-10")
        lazy = provider.load_lazy(backtest_start="2018-06-01", end_date="2018-06-10")

        assert isinstance(lazy, pl.LazyFrame)
        assert_frame_equal(eager, lazy.collect(), check_dtypes=False)

    def test_load_includes_pre_start_history_for_warmup_by_default(self) -> None:
        """load() retains pre-start rows so feature warmup is available."""
        df = _make_df(start="2015-01-01", periods=3000)
        provider = ColumnMapDataProvider(df=df)
        result = provider.load(backtest_start="2018-06-01")
        assert str(result["date"].min())[:10] == "2015-01-01"

    def test_load_windowing_respects_backtest_start_when_warmup_disabled(self) -> None:
        """load() clips to backtest_start when include_warmup=False."""
        df = _make_df(start="2015-01-01", periods=3000)
        provider = ColumnMapDataProvider(df=df)
        result = provider.load(backtest_start="2018-06-01", include_warmup=False)
        assert str(result["date"].min())[:10] == "2018-06-01"

    def test_load_windowing_respects_end_date(self) -> None:
        """load() clips at end_date."""
        df = _make_df(start="2018-01-01", periods=2000)
        provider = ColumnMapDataProvider(df=df)
        result = provider.load(backtest_start="2018-01-01", end_date="2019-01-01")
        assert str(result["date"].max())[:10] == "2019-01-01"

    def test_load_deduplicates_and_sorts_index(self) -> None:
        """Duplicate dates in the input are deduplicated."""
        df = pl.DataFrame({
            "date": [
                dt.datetime(2020, 1, 1),
                dt.datetime(2020, 1, 1),
                dt.datetime(2020, 1, 2),
            ],
            "price_usd": [100.0, 200.0, 300.0],
        })
        provider = ColumnMapDataProvider(df=df)
        result = provider.load(backtest_start="2020-01-01", end_date="2020-01-02")
        assert result.height == 2
        row = result.filter(pl.col("date") == dt.datetime(2020, 1, 1))
        assert float(row["price_usd"][0]) == 200.0  # keep last

    def test_unmapped_columns_are_preserved(self) -> None:
        """Extra user columns that aren't in the map are kept."""
        df = _make_df(extra_cols={"my_signal": 0.5})
        provider = ColumnMapDataProvider(df=df)
        result = provider.load(backtest_start="2018-01-01", end_date="2019-01-01")
        assert "my_signal" in result.columns


class TestColumnMapProviderErrors:
    def test_missing_price_usd_raises(self) -> None:
        """Missing price_usd after mapping raises ColumnMapError."""
        df = _make_df(price_col="Close")
        provider = ColumnMapDataProvider(df=df)
        with pytest.raises(ColumnMapError, match="price_usd"):
            provider.load()

    def test_column_map_references_nonexistent_column(self) -> None:
        """Column map referencing a column not in the df raises ColumnMapError."""
        df = _make_df()
        provider = ColumnMapDataProvider(df=df, column_map={"price_usd": "DoesNotExist"})
        with pytest.raises(ColumnMapError, match="DoesNotExist"):
            provider.load()

    def test_no_date_column_raises(self) -> None:
        """DataFrame without a date column raises ColumnMapError."""
        df = pl.DataFrame({"price_usd": [100.0, 200.0], "other": [0, 1]})
        provider = ColumnMapDataProvider(df=df)
        with pytest.raises(ColumnMapError, match="date"):
            provider.load()

    def test_empty_window_raises(self) -> None:
        """Requesting a window entirely outside the data coverage raises."""
        df = _make_df(start="2020-01-01", periods=100)
        provider = ColumnMapDataProvider(df=df)
        with pytest.raises(ColumnMapError, match="No rows available"):
            provider.load(backtest_start="2015-01-01", end_date="2015-12-31")

    def test_invalid_end_date_raises_value_error(self) -> None:
        """Invalid end_date string raises ValueError."""
        df = _make_df()
        provider = ColumnMapDataProvider(df=df)
        with pytest.raises(ValueError, match="Invalid end_date"):
            provider.load(end_date="not-a-date")

    def test_end_before_start_raises_value_error(self) -> None:
        """end_date before backtest_start raises ValueError."""
        df = _make_df()
        provider = ColumnMapDataProvider(df=df)
        with pytest.raises(ValueError, match="on or after backtest_start"):
            provider.load(backtest_start="2020-01-01", end_date="2019-01-01")

    def test_price_usd_nan_raises(self) -> None:
        """Missing (NaN) price_usd values in window raises ColumnMapError."""
        df = pl.DataFrame({
            "date": [dt.datetime(2020, 1, 1) + dt.timedelta(days=i) for i in range(5)],
            "price_usd": [100.0, float("nan"), 300.0, 400.0, 500.0],
        })
        provider = ColumnMapDataProvider(df=df)
        with pytest.raises(ColumnMapError, match="Missing price_usd"):
            provider.load(backtest_start="2020-01-01", end_date="2020-01-05")

    def test_price_usd_nan_in_warmup_history_raises(self) -> None:
        """Warmup rows are part of the loaded frame and must have valid price_usd."""
        df = pl.DataFrame(
            {
                "date": [
                    dt.datetime(2019, 12, 31),
                    dt.datetime(2020, 1, 1),
                    dt.datetime(2020, 1, 2),
                ],
                "price_usd": [float("nan"), 100.0, 101.0],
            }
        )
        provider = ColumnMapDataProvider(df=df)
        with pytest.raises(ColumnMapError, match="Missing price_usd"):
            provider.load(backtest_start="2020-01-01", end_date="2020-01-02")


# ---------------------------------------------------------------------------
# StrategyRunner.from_dataframe integration tests (require runner to use polars)
# ---------------------------------------------------------------------------


class TestColumnMapProviderCoverageEdges:
    def test_validate_required_columns_raises_when_missing(self) -> None:
        """_validate_required_columns raises when required columns are missing."""
        df = pl.DataFrame({"date": [dt.datetime(2024, 1, 1)], "x": [1.0]})
        with pytest.raises(ColumnMapError, match="Required library columns are missing"):
            ColumnMapDataProvider._validate_required_columns(df)

    def test_validate_required_columns_lazy_raises_when_missing(self) -> None:
        """_validate_required_columns_lazy raises when required columns are missing."""
        df_lazy = pl.DataFrame({"date": [dt.datetime(2024, 1, 1)], "x": [1.0]}).lazy()
        with pytest.raises(ColumnMapError, match="Required library columns are missing"):
            ColumnMapDataProvider._validate_required_columns_lazy(df_lazy)

    def test_to_daily_date_lazy_with_utf8_dtype(self) -> None:
        """_to_daily_date_lazy handles pl.Utf8 date column."""
        df = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "price_usd": [100.0, 101.0],
            "mvrv": [1.0, 1.0],
        })
        result = ColumnMapDataProvider._to_daily_date_lazy(df.lazy())
        assert result.collect().height == 2

    def test_to_daily_date_lazy_with_date_dtype(self) -> None:
        """_to_daily_date_lazy handles pl.Date column."""
        df = pl.DataFrame({
            "date": pl.Series([dt.date(2024, 1, 1), dt.date(2024, 1, 2)]).cast(pl.Date),
            "price_usd": [100.0, 101.0],
            "mvrv": [1.0, 1.0],
        })
        result = ColumnMapDataProvider._to_daily_date_lazy(df.lazy())
        assert result.collect().height == 2


class TestStrategyRunnerFromDataframe:
    def test_from_dataframe_returns_strategy_runner_instance(self) -> None:
        """from_dataframe returns a StrategyRunner configured with ColumnMapDataProvider."""
        df = _make_df()
        runner = StrategyRunner.from_dataframe(df, column_map={"price_usd": "price_usd"})
        assert isinstance(runner, StrategyRunner)
        assert isinstance(runner._data_provider, ColumnMapDataProvider)

    def test_from_dataframe_no_column_map(self) -> None:
        """from_dataframe with no column_map defaults to empty map (canonical pass-through)."""
        df = _make_df()
        runner = StrategyRunner.from_dataframe(df)
        assert runner._data_provider.column_map == {}
