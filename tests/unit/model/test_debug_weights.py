"""Test weight computation pipeline to ensure dynamic weights are working."""

import numpy as np
import pandas as pd
import pytest

from stacksats.framework_contract import ALLOCATION_SPAN_DAYS
from stacksats.model_development import (
    DYNAMIC_STRENGTH,
    _clean_array,
    compute_dynamic_multiplier,
    compute_window_weights,
    precompute_features,
)


class TestWeightComputationPipeline:
    """Test the complete weight computation pipeline."""

    @pytest.fixture
    def btc_data(self, sample_btc_df):
        """Use offline synthetic BTC fixture data for all tests."""
        return sample_btc_df

    @pytest.fixture
    def features_data(self, btc_data):
        """Precompute features once for all tests."""
        return precompute_features(btc_data)

    def test_mvrv_data_available(self, btc_data):
        """Verify MVRV data is available for weight computation."""
        assert "CapMVRVCur" in btc_data.columns, "CapMVRVCur column missing"
        mvrv_col = btc_data["CapMVRVCur"]
        assert mvrv_col.notna().sum() > 0, "No MVRV data available"

        # Check recent data in fixture range
        recent = btc_data.loc["2025-01-01":]
        assert len(recent) > 0, "No recent data"
        assert recent["CapMVRVCur"].notna().sum() > 0, "No MVRV data for recent dates"

    def test_features_vary(self, features_data):
        """Verify computed features have variation."""
        key_features = [
            "price_vs_ma",
            "mvrv_zscore",
            "mvrv_gradient",
            "mvrv_percentile",
        ]

        for col in key_features:
            assert col in features_data.columns, f"Feature {col} missing"
            feat = features_data[col]
            assert feat.notna().sum() > 0, f"Feature {col} has no valid values"
            assert feat.nunique() > 1, f"Feature {col} has no variation"

    def test_dynamic_multipliers_vary(self, features_data):
        """Verify dynamic multipliers have significant variation."""
        start_date = pd.to_datetime("2024-12-01")
        end_date = pd.to_datetime("2025-12-01")
        range_features = features_data.loc[start_date:end_date]

        assert len(range_features) > 0, "No features in test range"

        # Extract features
        price_vs_ma = _clean_array(range_features["price_vs_ma"].values)
        mvrv_zscore = _clean_array(range_features["mvrv_zscore"].values)
        mvrv_gradient = _clean_array(range_features["mvrv_gradient"].values)

        mvrv_percentile = None
        if "mvrv_percentile" in range_features.columns:
            mvrv_percentile = _clean_array(range_features["mvrv_percentile"].values)
            mvrv_percentile = np.where(mvrv_percentile == 0, 0.5, mvrv_percentile)

        mvrv_acceleration = None
        if "mvrv_acceleration" in range_features.columns:
            mvrv_acceleration = _clean_array(range_features["mvrv_acceleration"].values)

        mvrv_volatility = None
        if "mvrv_volatility" in range_features.columns:
            mvrv_volatility = _clean_array(range_features["mvrv_volatility"].values)
            mvrv_volatility = np.where(mvrv_volatility == 0, 0.5, mvrv_volatility)

        signal_confidence = None
        if "signal_confidence" in range_features.columns:
            signal_confidence = _clean_array(range_features["signal_confidence"].values)
            signal_confidence = np.where(signal_confidence == 0, 0.5, signal_confidence)

        # Compute multipliers
        dyn = compute_dynamic_multiplier(
            price_vs_ma,
            mvrv_zscore,
            mvrv_gradient,
            mvrv_percentile,
            mvrv_acceleration,
            mvrv_volatility,
            signal_confidence,
        )

        # Verify variation
        unique_count = len(np.unique(dyn))
        assert unique_count > 1, (
            f"Multipliers don't vary (only {unique_count} unique values)"
        )

        # Verify reasonable range
        assert dyn.min() > 0, "Multipliers should be positive"
        assert dyn.max() > 1, "Maximum multiplier should be > 1"

        # Verify significant variation (at least 10x difference)
        ratio = dyn.max() / dyn.min()
        assert ratio > 10, f"Insufficient multiplier variation (ratio: {ratio:.1f}x)"

    def test_weights_sum_to_one(self, features_data):
        """Verify computed weights sum to 1.0."""
        start_date = pd.to_datetime("2024-12-01")
        end_date = start_date + pd.Timedelta(days=ALLOCATION_SPAN_DAYS - 1)
        current_date = pd.Timestamp.now().normalize()

        weights = compute_window_weights(
            features_data, start_date, end_date, current_date
        )

        # Check sum
        total = weights.sum()
        assert abs(total - 1.0) < 1e-10, f"Weights don't sum to 1.0 (sum: {total})"

    def test_weights_have_variation(self, features_data):
        """Verify weights have meaningful variation."""
        start_date = pd.to_datetime("2024-12-01")
        end_date = start_date + pd.Timedelta(days=ALLOCATION_SPAN_DAYS - 1)
        current_date = pd.Timestamp.now().normalize()

        weights = compute_window_weights(
            features_data, start_date, end_date, current_date
        )

        # Check variation
        unique_count = weights.nunique()
        assert unique_count > 1, (
            f"Weights don't vary (only {unique_count} unique values)"
        )

        # Check range (should be much wider than uniform)
        min_weight = weights.min()
        max_weight = weights.max()
        ratio = max_weight / min_weight

        # Expect at least 100x variation for good dynamic behavior
        assert ratio > 100, (
            f"Insufficient weight variation (ratio: {ratio:.1f}x, min: {min_weight:.6f}, max: {max_weight:.6f})"
        )

    def test_weights_use_correct_parameters(self, features_data):
        """Verify weights use the expected DYNAMIC_STRENGTH and clipping."""
        # This test ensures the debug scripts match production parameters
        assert DYNAMIC_STRENGTH == 5.0, (
            f"Unexpected DYNAMIC_STRENGTH: {DYNAMIC_STRENGTH}"
        )

        # Test that clipping behavior is as expected
        test_values = np.array([-10, -5, 0, 50, 100, 150])
        clipped = np.clip(test_values, -5, 100)
        expected = np.array([-5, -5, 0, 50, 100, 100])

        np.testing.assert_array_equal(clipped, expected, "Clipping behavior unexpected")
