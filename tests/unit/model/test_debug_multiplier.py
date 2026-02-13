"""Test dynamic multiplier computation to ensure signals are working correctly."""

import numpy as np
import pandas as pd
import pytest

from stacksats.model_development import (
    DYNAMIC_STRENGTH,
    MVRV_VOLATILITY_DAMPENING,
    _clean_array,
    compute_acceleration_modifier,
    compute_adaptive_trend_modifier,
    compute_asymmetric_extreme_boost,
    compute_dynamic_multiplier,
    compute_percentile_signal,
    precompute_features,
)


class TestDynamicMultiplierComputation:
    """Test dynamic multiplier computation step by step."""

    @pytest.fixture
    def btc_data(self, sample_btc_df):
        """Use offline synthetic BTC fixture data for all tests."""
        return sample_btc_df

    @pytest.fixture
    def features_data(self, btc_data):
        """Precompute features once for all tests."""
        return precompute_features(btc_data)

    @pytest.fixture
    def sample_features(self, features_data):
        """Get sample features for testing."""
        start_date = pd.to_datetime("2024-12-01")
        end_date = pd.to_datetime("2025-12-01")
        range_features = features_data.loc[start_date:end_date]

        # Extract features
        price_vs_ma = _clean_array(range_features["price_vs_ma"].values)
        mvrv_zscore = _clean_array(range_features["mvrv_zscore"].values)
        mvrv_gradient = _clean_array(range_features["mvrv_gradient"].values)
        mvrv_percentile = _clean_array(range_features["mvrv_percentile"].values)
        mvrv_percentile = np.where(mvrv_percentile == 0, 0.5, mvrv_percentile)
        mvrv_acceleration = _clean_array(range_features["mvrv_acceleration"].values)
        mvrv_volatility = _clean_array(range_features["mvrv_volatility"].values)
        mvrv_volatility = np.where(mvrv_volatility == 0, 0.5, mvrv_volatility)
        signal_confidence = _clean_array(range_features["signal_confidence"].values)
        signal_confidence = np.where(signal_confidence == 0, 0.5, signal_confidence)

        return {
            "price_vs_ma": price_vs_ma,
            "mvrv_zscore": mvrv_zscore,
            "mvrv_gradient": mvrv_gradient,
            "mvrv_percentile": mvrv_percentile,
            "mvrv_acceleration": mvrv_acceleration,
            "mvrv_volatility": mvrv_volatility,
            "signal_confidence": signal_confidence,
        }

    def test_mvrv_value_signal(self, sample_features):
        """Test MVRV value signal computation."""
        mvrv_zscore = sample_features["mvrv_zscore"]
        value_signal = -mvrv_zscore

        # Should have variation
        assert len(np.unique(value_signal)) > 1, "Value signal has no variation"
        # Should be positive for negative z-scores (undervaluation)
        assert np.any(value_signal > 0), (
            "Should have positive signals for undervaluation"
        )

    def test_extreme_boost(self, sample_features):
        """Test asymmetric extreme boost."""
        mvrv_zscore = sample_features["mvrv_zscore"]
        boost = compute_asymmetric_extreme_boost(mvrv_zscore)

        # Should have variation
        assert len(np.unique(boost)) > 1, "Extreme boost has no variation"
        # Boost should respond in both directions across regimes
        assert np.any(boost > 0), "Expected positive boost values"
        assert np.any(boost < 0), "Expected negative boost values"

    def test_ma_signal_with_trend_modifier(self, sample_features):
        """Test MA signal with adaptive trend modification."""
        price_vs_ma = sample_features["price_vs_ma"]
        mvrv_gradient = sample_features["mvrv_gradient"]
        mvrv_zscore = sample_features["mvrv_zscore"]

        ma_signal = -price_vs_ma
        trend_modifier = compute_adaptive_trend_modifier(mvrv_gradient, mvrv_zscore)
        ma_signal = ma_signal * trend_modifier

        # Should have variation
        assert len(np.unique(ma_signal)) > 1, "MA signal has no variation"
        # Trend modifier should be in reasonable range
        assert 0.3 <= trend_modifier.min() <= trend_modifier.max() <= 1.5, (
            "Trend modifier out of range"
        )

    def test_percentile_signal(self, sample_features):
        """Test percentile signal computation."""
        mvrv_percentile = sample_features["mvrv_percentile"]
        pct_signal = compute_percentile_signal(mvrv_percentile)

        # Should have variation
        assert len(np.unique(pct_signal)) > 1, "Percentile signal has no variation"
        # Should be in [-1, 1] range
        assert pct_signal.min() >= -1 and pct_signal.max() <= 1, (
            "Percentile signal out of range"
        )

    def test_acceleration_modifier(self, sample_features):
        """Test acceleration modifier."""
        mvrv_acceleration = sample_features["mvrv_acceleration"]
        mvrv_gradient = sample_features["mvrv_gradient"]

        accel_modifier = compute_acceleration_modifier(mvrv_acceleration, mvrv_gradient)

        # Should be in [0.5, 1.5] range
        assert 0.5 <= accel_modifier.min() <= accel_modifier.max() <= 1.5, (
            "Acceleration modifier out of range"
        )

    def test_combined_signals(self, sample_features):
        """Test combined signal computation."""
        price_vs_ma = sample_features["price_vs_ma"]
        mvrv_zscore = sample_features["mvrv_zscore"]
        mvrv_gradient = sample_features["mvrv_gradient"]
        mvrv_percentile = sample_features["mvrv_percentile"]
        mvrv_acceleration = sample_features["mvrv_acceleration"]
        mvrv_volatility = sample_features["mvrv_volatility"]
        signal_confidence = sample_features["signal_confidence"]

        # Compute step by step
        value_signal = -mvrv_zscore
        extreme_boost = compute_asymmetric_extreme_boost(mvrv_zscore)
        value_signal = value_signal + extreme_boost

        ma_signal = -price_vs_ma
        trend_modifier = compute_adaptive_trend_modifier(mvrv_gradient, mvrv_zscore)
        ma_signal = ma_signal * trend_modifier

        pct_signal = compute_percentile_signal(mvrv_percentile)
        accel_modifier = compute_acceleration_modifier(mvrv_acceleration, mvrv_gradient)

        combined = value_signal * 0.70 + ma_signal * 0.20 + pct_signal * 0.10
        accel_modifier_subtle = 0.85 + 0.30 * (accel_modifier - 0.5) / 0.5
        accel_modifier_subtle = np.clip(accel_modifier_subtle, 0.85, 1.15)
        combined = combined * accel_modifier_subtle

        confidence_boost = np.where(
            signal_confidence > 0.7, 1.0 + 0.15 * (signal_confidence - 0.7) / 0.3, 1.0
        )
        combined = combined * confidence_boost

        volatility_dampening = np.where(
            mvrv_volatility > 0.8,
            1.0 - MVRV_VOLATILITY_DAMPENING * (mvrv_volatility - 0.8) / 0.2,
            1.0,
        )
        combined = combined * volatility_dampening

        # Should have significant variation
        assert len(np.unique(combined)) > 1, "Combined signal has no variation"
        assert (combined.max() - combined.min()) > 1.2, (
            "Insufficient combined signal spread"
        )

    def test_final_multiplier_computation(self, sample_features):
        """Test final multiplier computation with correct parameters."""
        dyn = compute_dynamic_multiplier(
            sample_features["price_vs_ma"],
            sample_features["mvrv_zscore"],
            sample_features["mvrv_gradient"],
            sample_features["mvrv_percentile"],
            sample_features["mvrv_acceleration"],
            sample_features["mvrv_volatility"],
            sample_features["signal_confidence"],
        )

        # Should have significant variation
        assert len(np.unique(dyn)) > 1, "Multipliers have no variation"
        assert dyn.min() > 0, "Multipliers should be positive"

        # With DYNAMIC_STRENGTH=5.0 and clip [-5,100], expect wide range
        ratio = dyn.max() / dyn.min()
        assert ratio > 10, f"Insufficient multiplier variation (ratio: {ratio:.1f}x)"

    def test_correct_parameters_used(self):
        """Verify correct parameters are used."""
        assert DYNAMIC_STRENGTH == 5.0, f"Wrong DYNAMIC_STRENGTH: {DYNAMIC_STRENGTH}"
        assert MVRV_VOLATILITY_DAMPENING == 0.2, (
            f"Wrong MVRV_VOLATILITY_DAMPENING: {MVRV_VOLATILITY_DAMPENING}"
        )

    def test_clipping_behavior(self):
        """Test that clipping works as expected."""
        # Test the actual clipping used in production
        test_combined = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
        adjustment = test_combined * DYNAMIC_STRENGTH
        clipped = np.clip(adjustment, -5, 100)

        # Should not be clipped for reasonable values
        assert clipped[0] == adjustment[0], "Small positive value should not be clipped"
        assert clipped[-1] == 100.0, "Large value should be clipped to 100"

        # Convert to multipliers
        multipliers = np.exp(clipped)
        assert multipliers.min() > 0, "Multipliers should be positive"
        assert multipliers.max() == np.exp(100), "Large multiplier should be exp(100)"
