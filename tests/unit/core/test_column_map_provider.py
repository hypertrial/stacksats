"""Tests for ColumnMapDataProvider and StrategyRunner.from_dataframe."""

from __future__ import annotations

import pandas as pd
import pytest

from stacksats.column_map_provider import ColumnMapDataProvider, ColumnMapError
from stacksats.runner import StrategyRunner
from stacksats.strategies.examples import UniformStrategy
from stacksats.strategy_types import BacktestConfig


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
) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="D")
    data = {price_col: price}
    if extra_cols:
        data.update(extra_cols)
    return pd.DataFrame(data, index=idx)


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
        assert result.index.min() == pd.Timestamp("2018-01-01")
        assert result.index.max() == pd.Timestamp("2020-01-01")

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

    def test_load_windowing_respects_backtest_start(self) -> None:
        """load() slices the window from backtest_start onward."""
        df = _make_df(start="2015-01-01", periods=3000)
        provider = ColumnMapDataProvider(df=df)
        result = provider.load(backtest_start="2018-06-01")
        assert result.index.min() == pd.Timestamp("2018-06-01")

    def test_load_windowing_respects_end_date(self) -> None:
        """load() clips at end_date."""
        df = _make_df(start="2018-01-01", periods=2000)
        provider = ColumnMapDataProvider(df=df)
        result = provider.load(backtest_start="2018-01-01", end_date="2019-01-01")
        assert result.index.max() == pd.Timestamp("2019-01-01")

    def test_load_deduplicates_and_sorts_index(self) -> None:
        """Duplicate dates in the input are deduplicated."""
        idx = pd.DatetimeIndex(["2020-01-01", "2020-01-01", "2020-01-02"])
        df = pd.DataFrame({"price_usd": [100.0, 200.0, 300.0]}, index=idx)
        provider = ColumnMapDataProvider(df=df)
        result = provider.load(backtest_start="2020-01-01", end_date="2020-01-02")
        assert len(result) == 2
        assert float(result.loc["2020-01-01", "price_usd"]) == 200.0  # keep last

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
        # no map provided — price_usd not in df
        provider = ColumnMapDataProvider(df=df)
        with pytest.raises(ColumnMapError, match="price_usd"):
            provider.load()

    def test_column_map_references_nonexistent_column(self) -> None:
        """Column map referencing a column not in the df raises ColumnMapError."""
        df = _make_df()
        provider = ColumnMapDataProvider(df=df, column_map={"price_usd": "DoesNotExist"})
        with pytest.raises(ColumnMapError, match="DoesNotExist"):
            provider.load()

    def test_non_datetime_index_raises(self) -> None:
        """DataFrame with a non-DatetimeIndex raises ColumnMapError."""
        df = pd.DataFrame({"price_usd": [100.0, 200.0]}, index=[0, 1])
        provider = ColumnMapDataProvider(df=df)
        with pytest.raises(ColumnMapError, match="DatetimeIndex"):
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
        import numpy as np

        idx = pd.date_range("2020-01-01", periods=5, freq="D")
        df = pd.DataFrame({"price_usd": [100.0, np.nan, 300.0, 400.0, 500.0]}, index=idx)
        provider = ColumnMapDataProvider(df=df)
        with pytest.raises(ColumnMapError, match="Missing price_usd"):
            provider.load(backtest_start="2020-01-01", end_date="2020-01-05")


# ---------------------------------------------------------------------------
# StrategyRunner.from_dataframe integration tests
# ---------------------------------------------------------------------------


class TestStrategyRunnerFromDataframe:
    def test_from_dataframe_canonical_columns(self) -> None:
        """from_dataframe with canonical column names runs a backtest successfully."""
        df = _make_df(start="2018-01-01", periods=1100)
        runner = StrategyRunner.from_dataframe(df)
        result = runner.backtest(
            UniformStrategy(),
            BacktestConfig(start_date="2018-01-01", end_date="2020-01-01"),
        )
        assert result is not None
        assert result.win_rate >= 0.0

    def test_from_dataframe_with_column_map(self) -> None:
        """from_dataframe with column_map works end-to-end."""
        df = _make_df(start="2018-01-01", periods=1100, price_col="Close")
        runner = StrategyRunner.from_dataframe(df, column_map={"price_usd": "Close"})
        result = runner.backtest(
            UniformStrategy(),
            BacktestConfig(start_date="2018-01-01", end_date="2020-01-01"),
        )
        assert result is not None

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
