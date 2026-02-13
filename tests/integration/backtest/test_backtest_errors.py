"""Tests for error handling in backtest.py.

Tests error conditions, exception handling, and edge cases that should
raise appropriate errors or handle gracefully.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import stacksats.backtest as backtest
from stacksats.backtest import compute_weights_modal
from stacksats.model_development import precompute_features

# -----------------------------------------------------------------------------
# compute_weights_modal Error Tests
# -----------------------------------------------------------------------------


class TestComputeWeightsModalErrors:
    """Tests for error conditions in compute_weights_modal."""

    def test_features_not_precomputed_raises_error(self):
        """Test that compute_weights_modal raises error when _FEATURES_DF is None."""
        # Save original state
        original_features = backtest._FEATURES_DF

        try:
            # Set to None to simulate uninitialized state
            backtest._FEATURES_DF = None

            # Create a dummy DataFrame
            dummy_df = pd.DataFrame(
                {"col1": [1, 2, 3]},
                index=pd.date_range("2024-01-01", periods=3, freq="D"),
            )

            with pytest.raises(ValueError, match="Features not precomputed"):
                compute_weights_modal(dummy_df)

        finally:
            # Restore original state
            backtest._FEATURES_DF = original_features

    def test_empty_window_returns_empty_series(self, sample_features_df):
        """Test that empty window input returns empty Series."""
        backtest._FEATURES_DF = sample_features_df

        # Create empty DataFrame
        empty_df = pd.DataFrame(index=pd.DatetimeIndex([]))

        result = compute_weights_modal(empty_df)

        assert len(result) == 0
        assert isinstance(result, pd.Series)


# -----------------------------------------------------------------------------
# precompute_features Error Tests
# -----------------------------------------------------------------------------


class TestPrecomputeFeaturesErrors:
    """Tests for error conditions in precompute_features."""

    def test_missing_price_column_raises_error(self):
        """Test that missing price column raises KeyError."""
        # Create DataFrame without required price column
        df = pd.DataFrame(
            {"other_column": [100, 200, 300]},
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
        )

        with pytest.raises(KeyError):
            precompute_features(df)

    def test_wrong_price_column_name_raises_error(self):
        """Test that wrong column name raises KeyError."""
        df = pd.DataFrame(
            {"price": [100, 200, 300]},  # Wrong name
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
        )

        with pytest.raises(KeyError):
            precompute_features(df)

    def test_empty_dataframe_handling(self):
        """Test behavior with empty DataFrame."""
        empty_df = pd.DataFrame(columns=["PriceUSD_coinmetrics"])
        empty_df.index = pd.DatetimeIndex([])

        # Should handle empty data gracefully (may return empty or raise)
        result = precompute_features(empty_df)
        # Empty input should produce empty output
        assert len(result) == 0


# -----------------------------------------------------------------------------
# Data Quality Error Tests
# -----------------------------------------------------------------------------


class TestDataQualityErrors:
    """Tests for handling data quality issues."""

    def test_handles_nan_prices_in_features(self, sample_btc_df):
        """Test that NaN prices are handled during feature computation."""
        # Create DataFrame with NaN prices
        df_with_nan = sample_btc_df.copy()
        df_with_nan.loc["2024-06-15", "PriceUSD_coinmetrics"] = float("nan")

        # Should not raise, but produce features
        result = precompute_features(df_with_nan)

        # Result should exist and have features
        assert len(result) > 0

    def test_handles_zero_prices(self, sample_btc_df):
        """Test behavior with zero prices (edge case)."""
        df_with_zero = sample_btc_df.copy()
        # Setting a zero price - this is invalid but should be handled
        df_with_zero.loc["2024-06-15", "PriceUSD_coinmetrics"] = 0.0

        # Should complete (may produce warnings or clipped values)
        result = precompute_features(df_with_zero)
        assert len(result) > 0

    def test_handles_negative_prices(self, sample_btc_df):
        """Test behavior with negative prices (invalid data)."""
        df_with_neg = sample_btc_df.copy()
        # Negative price is invalid
        df_with_neg.loc["2024-06-15", "PriceUSD_coinmetrics"] = -1000.0

        # Should complete (log of negative will produce NaN/Inf, which gets handled)
        result = precompute_features(df_with_neg)
        assert len(result) > 0


# -----------------------------------------------------------------------------
# Boundary Condition Tests
# -----------------------------------------------------------------------------


class TestBoundaryConditions:
    """Tests for boundary conditions that may cause errors."""

    def test_single_row_dataframe(self, sample_btc_df):
        """Test handling of single-row DataFrame."""
        single_row = sample_btc_df.iloc[[0]].copy()

        result = precompute_features(single_row)

        # Should handle gracefully
        assert len(result) <= 1

    def test_very_short_date_range(self, sample_features_df):
        """Very short ranges should fail strict span validation."""
        backtest._FEATURES_DF = sample_features_df

        # Two-day window
        start_date = pd.Timestamp("2024-01-01")
        end_date = pd.Timestamp("2024-01-02")

        window_feat = sample_features_df.loc[start_date:end_date]

        if len(window_feat) > 0:
            with pytest.raises(ValueError, match="365, 366, or 367 allocation days"):
                compute_weights_modal(window_feat)

    def test_date_range_at_data_boundaries(self, sample_btc_df, sample_features_df):
        """Boundary windows must still respect 365/366-day contract."""
        backtest._FEATURES_DF = sample_features_df

        # Use dates at the start of features
        first_date = sample_features_df.index.min()
        window_feat = sample_features_df.loc[first_date : first_date + pd.Timedelta(days=364)]

        if len(window_feat) > 0:
            result = compute_weights_modal(window_feat)
            assert len(result) > 0

    def test_non_daily_frequency_handling(self):
        """Test that non-daily frequency data is handled."""
        # Create weekly data
        dates = pd.date_range("2020-01-01", periods=100, freq="W")
        df = pd.DataFrame(
            {"PriceUSD_coinmetrics": [10000 + i * 100 for i in range(100)]},
            index=dates,
        )

        # Should handle non-daily frequency
        result = precompute_features(df)
        assert len(result) > 0
