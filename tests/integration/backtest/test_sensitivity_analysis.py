"""Parameter sensitivity analysis tests for the Bitcoin DCA strategy.

These tests verify:
1. Small constant perturbations cause proportionally small weight changes
2. The model is not overly sensitive to any single parameter
3. Zone threshold changes are stable
4. Extreme parameter values are handled gracefully
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import stacksats.model_development as md
from stacksats.model_development import (
    DYNAMIC_STRENGTH,
    FEATS,
    compute_asymmetric_extreme_boost,
    compute_dynamic_multiplier,
    compute_weights_fast,
    precompute_features,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sensitivity_btc_df():
    """Create BTC price data for sensitivity tests."""
    dates = pd.date_range(start="2020-01-01", end="2025-12-31", freq="D")
    np.random.seed(42)
    base_price = 10000
    returns = np.random.normal(0.001, 0.03, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))

    # Create MVRV data that varies with price cycles
    mvrv_base = 1.5 + 0.5 * np.sin(np.arange(len(dates)) * 2 * np.pi / 1461)
    mvrv_noise = np.random.normal(0, 0.2, len(dates))
    mvrv = np.clip(mvrv_base + mvrv_noise, 0.5, 4.0)

    df = pd.DataFrame({"PriceUSD_coinmetrics": prices, "CapMVRVCur": mvrv}, index=dates)
    df["PriceUSD"] = df["PriceUSD_coinmetrics"]
    df.index.name = "time"
    return df


@pytest.fixture(scope="module")
def sensitivity_features_df(sensitivity_btc_df):
    """Precompute features for sensitivity tests."""
    return precompute_features(sensitivity_btc_df)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def weight_difference_metrics(weights1: pd.Series, weights2: pd.Series) -> dict:
    """Calculate various difference metrics between two weight vectors."""
    diff = weights1 - weights2
    abs_diff = np.abs(diff)

    return {
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "rmse": float(np.sqrt((diff**2).mean())),
        "max_relative_diff": float((abs_diff / np.maximum(weights1, 1e-10)).max()),
        "correlation": float(weights1.corr(weights2)),
    }


def compute_weights_with_dynamic_strength(
    features_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    dynamic_strength: float,
) -> pd.Series:
    """Compute weights with a custom DYNAMIC_STRENGTH value."""
    original_strength = md.DYNAMIC_STRENGTH
    md.DYNAMIC_STRENGTH = dynamic_strength
    try:
        weights = compute_weights_fast(features_df, start_date, end_date)
    finally:
        md.DYNAMIC_STRENGTH = original_strength
    return weights


# -----------------------------------------------------------------------------
# DYNAMIC_STRENGTH Sensitivity Tests
# -----------------------------------------------------------------------------


class TestDynamicStrengthSensitivity:
    """Test sensitivity to DYNAMIC_STRENGTH perturbations."""

    def test_small_dynamic_strength_perturbation(self, sensitivity_features_df):
        """Verify 5% DYNAMIC_STRENGTH perturbation causes bounded weight changes."""
        start_date = pd.Timestamp("2024-01-01")
        end_date = pd.Timestamp("2024-12-31")

        base_weights = compute_weights_fast(
            sensitivity_features_df, start_date, end_date
        )

        # Perturb DYNAMIC_STRENGTH by 5%
        perturbed_weights = compute_weights_with_dynamic_strength(
            sensitivity_features_df, start_date, end_date, DYNAMIC_STRENGTH * 1.05
        )

        metrics = weight_difference_metrics(base_weights, perturbed_weights)

        # Max weight change from 5% perturbation should be bounded
        assert metrics["max_abs_diff"] < 0.15, (
            f"5% DYNAMIC_STRENGTH perturbation causes {metrics['max_abs_diff']:.4f} "
            "max weight change, expected < 0.15"
        )

    def test_dynamic_strength_range(self, sensitivity_features_df):
        """Test that various DYNAMIC_STRENGTH values produce valid weights."""
        start_date = pd.Timestamp("2024-01-01")
        end_date = pd.Timestamp("2024-12-31")

        for strength in [1.0, 2.0, 3.0, 4.0, 5.0]:
            weights = compute_weights_with_dynamic_strength(
                sensitivity_features_df, start_date, end_date, strength
            )

            assert np.isclose(weights.sum(), 1.0, rtol=1e-6), (
                f"Weights don't sum to 1.0 for DYNAMIC_STRENGTH={strength}"
            )
            assert weights.notna().all(), f"NaN weights for DYNAMIC_STRENGTH={strength}"
            assert (weights >= -1e-10).all(), (
                f"Negative weights for DYNAMIC_STRENGTH={strength}"
            )


# -----------------------------------------------------------------------------
# Zone Threshold Sensitivity Tests
# -----------------------------------------------------------------------------


class TestZoneThresholdSensitivity:
    """Test sensitivity to MVRV zone threshold changes."""

    def test_zone_threshold_stability(self):
        """Verify zone classification is stable to small threshold changes."""
        mvrv_values = np.linspace(-3, 4, 100)

        # Base zones
        base_zones = md.classify_mvrv_zone(mvrv_values)

        # Perturb thresholds by 10%
        original_thresholds = {
            "deep_value": md.MVRV_ZONE_DEEP_VALUE,
            "value": md.MVRV_ZONE_VALUE,
            "caution": md.MVRV_ZONE_CAUTION,
            "danger": md.MVRV_ZONE_DANGER,
        }

        try:
            md.MVRV_ZONE_DEEP_VALUE *= 0.9
            md.MVRV_ZONE_VALUE *= 0.9
            md.MVRV_ZONE_CAUTION *= 1.1
            md.MVRV_ZONE_DANGER *= 1.1

            perturbed_zones = md.classify_mvrv_zone(mvrv_values)
        finally:
            md.MVRV_ZONE_DEEP_VALUE = original_thresholds["deep_value"]
            md.MVRV_ZONE_VALUE = original_thresholds["value"]
            md.MVRV_ZONE_CAUTION = original_thresholds["caution"]
            md.MVRV_ZONE_DANGER = original_thresholds["danger"]

        # Most zones should remain the same
        zone_change_rate = (base_zones != perturbed_zones).mean()
        assert zone_change_rate < 0.3, (
            f"Zone classification too sensitive: {zone_change_rate:.1%} changed"
        )

    def test_extreme_boost_continuity(self):
        """Verify extreme boost function is continuous across zone boundaries."""
        mvrv_values = np.linspace(-4, 4, 1000)
        boosts = compute_asymmetric_extreme_boost(mvrv_values)

        # Check for reasonable continuity (no extreme jumps)
        boost_diffs = np.abs(np.diff(boosts))
        max_diff = boost_diffs.max()

        # Allow for some discontinuity at zone boundaries (up to 0.6)
        assert max_diff < 0.6, (
            f"Extreme boost has discontinuity: max_diff={max_diff:.4f}"
        )


# -----------------------------------------------------------------------------
# Multiplier Component Sensitivity Tests
# -----------------------------------------------------------------------------


class TestMultiplierComponentSensitivity:
    """Test sensitivity of individual multiplier components."""

    def test_dynamic_multiplier_components(self):
        """Test that each signal component has bounded contribution."""
        n = 366
        price_vs_ma = np.random.uniform(-1, 1, n)
        mvrv_zscore = np.random.uniform(-4, 4, n)
        mvrv_gradient = np.random.uniform(-1, 1, n)
        mvrv_percentile = np.random.uniform(0, 1, n)
        mvrv_acceleration = np.random.uniform(-1, 1, n)
        mvrv_volatility = np.random.uniform(0, 1, n)
        signal_confidence = np.random.uniform(0, 1, n)

        # Full multiplier
        full = compute_dynamic_multiplier(
            price_vs_ma,
            mvrv_zscore,
            mvrv_gradient,
            mvrv_percentile,
            mvrv_acceleration,
            mvrv_volatility,
            signal_confidence,
        )

        # Multiplier with only MVRV (zero other signals)
        mvrv_only = compute_dynamic_multiplier(
            np.zeros(n),
            mvrv_zscore,
            np.zeros(n),
            None,
            None,
            None,
            None,
        )

        # Multiplier with only MA (zero other signals)
        ma_only = compute_dynamic_multiplier(
            price_vs_ma,
            np.zeros(n),
            np.zeros(n),
            None,
            None,
            None,
            None,
        )

        # All multipliers should be positive and finite
        for mult, name in [
            (full, "full"),
            (mvrv_only, "mvrv_only"),
            (ma_only, "ma_only"),
        ]:
            assert (mult > 0).all(), f"{name} has non-positive values"
            assert np.all(np.isfinite(mult)), f"{name} has non-finite values"

    def test_extreme_input_handling(self):
        """Test that extreme input values produce finite multipliers."""
        # Test with extreme values (100 elements each)
        extreme_price_vs_ma = np.array([-1, 1, 0] * 33 + [-1])
        extreme_mvrv = np.array([-4, 4, 0] * 33 + [-4])
        extreme_gradient = np.array([-1, 1, 0] * 33 + [-1])

        multiplier = compute_dynamic_multiplier(
            extreme_price_vs_ma, extreme_mvrv, extreme_gradient
        )

        assert (multiplier > 0).all(), "Extreme inputs produce non-positive multipliers"
        assert np.all(np.isfinite(multiplier)), (
            "Extreme inputs produce non-finite multipliers"
        )
        assert multiplier.min() > 0.001, "Multipliers too close to zero"


# -----------------------------------------------------------------------------
# Window Size Sensitivity Tests
# -----------------------------------------------------------------------------


class TestWindowSensitivity:
    """Test sensitivity to different window sizes."""

    def test_ma_window_sensitivity(self, sensitivity_btc_df):
        """Test sensitivity to MA_WINDOW changes."""
        original_window = md.MA_WINDOW

        results = []
        for window in [100, 150, 200, 250, 300]:
            md.MA_WINDOW = window
            try:
                features = precompute_features(sensitivity_btc_df)
                weights = compute_weights_fast(
                    features,
                    pd.Timestamp("2024-01-01"),
                    pd.Timestamp("2024-12-31"),
                )
                results.append(
                    {
                        "window": window,
                        "mean_weight": weights.mean(),
                        "std_weight": weights.std(),
                    }
                )
            finally:
                md.MA_WINDOW = original_window

        # Results should have variation but not extreme
        stds = [r["std_weight"] for r in results]
        assert max(stds) / min(stds) < 5, (
            "MA window has extreme effect on weight variance"
        )


# -----------------------------------------------------------------------------
# Correlation Tests
# -----------------------------------------------------------------------------


class TestFeatureCorrelations:
    """Test correlations between features don't cause instability."""

    def test_feature_independence(self, sensitivity_features_df):
        """Test that weight changes from different features aren't perfectly correlated."""
        start_date = pd.Timestamp("2024-01-01")
        end_date = pd.Timestamp("2024-12-31")

        df = sensitivity_features_df.loc[start_date:end_date]

        # Get feature columns
        feature_cols = [c for c in FEATS if c in df.columns]

        # Compute correlation matrix
        if len(feature_cols) >= 2:
            corr_matrix = df[feature_cols].corr()

            # Check that no features are perfectly correlated
            # (excluding self-correlation on diagonal)
            off_diagonal = corr_matrix.values[
                np.triu_indices_from(corr_matrix.values, k=1)
            ]
            max_corr = np.abs(off_diagonal).max()

            assert max_corr < 0.99, (
                f"Features are too correlated: max_corr={max_corr:.4f}"
            )


# -----------------------------------------------------------------------------
# Sensitivity Report
# -----------------------------------------------------------------------------


class TestSensitivityReport:
    """Generate sensitivity metrics for documentation."""

    def test_compute_strength_sensitivity_report(self, sensitivity_features_df):
        """Compute and verify sensitivity to DYNAMIC_STRENGTH."""
        start_date = pd.Timestamp("2024-01-01")
        end_date = pd.Timestamp("2024-12-31")

        base_weights = compute_weights_fast(
            sensitivity_features_df, start_date, end_date
        )

        sensitivities = []
        for strength_mult in [0.8, 0.9, 1.0, 1.1, 1.2]:
            perturbed_strength = DYNAMIC_STRENGTH * strength_mult
            perturbed_weights = compute_weights_with_dynamic_strength(
                sensitivity_features_df, start_date, end_date, perturbed_strength
            )

            metrics = weight_difference_metrics(base_weights, perturbed_weights)
            sensitivities.append(
                {
                    "strength_mult": strength_mult,
                    "strength_value": perturbed_strength,
                    **metrics,
                }
            )

        # Verify report structure
        assert len(sensitivities) == 5

        # The 1.0 multiplier should have zero diff (same as base)
        base_case = [s for s in sensitivities if s["strength_mult"] == 1.0][0]
        assert base_case["max_abs_diff"] < 1e-10, "Base case should have zero diff"

        # Overall sensitivity should be reasonable
        max_sens = max(s["max_abs_diff"] for s in sensitivities)
        assert max_sens < 0.3, f"Max sensitivity too high: {max_sens:.4f}"
