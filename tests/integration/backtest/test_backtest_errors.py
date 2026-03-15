"""Polars-native backtest error handling coverage."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from stacksats.backtest import compute_weights_with_features
from stacksats.model_development import precompute_features
from tests.integration.backtest.polars_backtest_testkit import (
    dt_at,
    make_btc_df,
    slice_dates,
)

pytestmark = pytest.mark.integration


class TestComputeWeightsSharedErrors:
    """Tests for explicit feature-window weight computation."""

    def test_empty_window_returns_empty_dataframe(self, sample_features_df):
        empty_df = pl.DataFrame(schema={"date": pl.Datetime("us")})
        result = compute_weights_with_features(empty_df, features_df=sample_features_df)
        assert isinstance(result, pl.DataFrame)
        assert result.is_empty()
        assert result.columns == ["date", "weight"]


class TestPrecomputeFeaturesErrors:
    """Tests for Polars-only precompute_features error handling."""

    def test_missing_price_column_raises_error(self):
        df = pl.DataFrame({
            "date": [dt_at("2020-01-01"), dt_at("2020-01-02"), dt_at("2020-01-03")],
            "other_column": [100, 200, 300],
        })
        with pytest.raises(KeyError):
            precompute_features(df)

    def test_wrong_price_column_name_raises_error(self):
        df = pl.DataFrame({
            "date": [dt_at("2020-01-01"), dt_at("2020-01-02"), dt_at("2020-01-03")],
            "price": [100, 200, 300],
        })
        with pytest.raises(KeyError):
            precompute_features(df)

    def test_empty_dataframe_handling(self):
        empty_df = pl.DataFrame(schema={"date": pl.Datetime("us"), "price_usd": pl.Float64})
        result = precompute_features(empty_df)
        assert result.is_empty()


class TestDataQualityErrors:
    """Tests for data-quality edge cases in feature generation."""

    def test_handles_nan_prices_in_features(self, sample_btc_df):
        df_with_nan = sample_btc_df.with_columns(
            pl.when(pl.col("date") == dt_at("2024-06-15"))
            .then(float("nan"))
            .otherwise(pl.col("price_usd"))
            .alias("price_usd")
        )
        result = precompute_features(df_with_nan)
        assert result.height > 0

    def test_handles_zero_prices(self, sample_btc_df):
        df_with_zero = sample_btc_df.with_columns(
            pl.when(pl.col("date") == dt_at("2024-06-15"))
            .then(0.0)
            .otherwise(pl.col("price_usd"))
            .alias("price_usd")
        )
        result = precompute_features(df_with_zero)
        assert result.height > 0

    def test_handles_negative_prices(self, sample_btc_df):
        df_with_neg = sample_btc_df.with_columns(
            pl.when(pl.col("date") == dt_at("2024-06-15"))
            .then(-1000.0)
            .otherwise(pl.col("price_usd"))
            .alias("price_usd")
        )
        result = precompute_features(df_with_neg)
        assert result.height > 0


class TestBoundaryConditions:
    """Tests for short and boundary date windows."""

    def test_single_row_dataframe(self, sample_btc_df):
        single_row = sample_btc_df.head(1)
        result = precompute_features(single_row)
        assert result.height <= 1

    def test_very_short_date_range(self, sample_features_df):
        start_date = dt_at("2024-01-01")
        end_date = dt_at("2024-01-02")
        window_feat = slice_dates(sample_features_df, start_date, end_date)
        if window_feat.height > 0:
            with pytest.raises(ValueError, match="configured fixed span"):
                compute_weights_with_features(window_feat, features_df=sample_features_df)

    def test_date_range_at_data_boundaries(self, sample_features_df):
        first_date = sample_features_df["date"].min()
        end_date = first_date + dt.timedelta(days=364)
        window_feat = slice_dates(sample_features_df, first_date, end_date)
        if window_feat.height > 0:
            result = compute_weights_with_features(window_feat, features_df=sample_features_df)
            assert result.height > 0

    def test_non_daily_frequency_handling(self):
        weekly_dates = [
            dt.datetime(2020, 1, 1) + dt.timedelta(days=7 * offset)
            for offset in range(100)
        ]
        df = make_btc_df(start="2020-01-01", days=100).select(["price_usd", "mvrv"]).with_columns(
            pl.Series("date", weekly_dates)
        )
        result = precompute_features(df.select(["date", "price_usd", "mvrv"]))
        assert result.height > 0
