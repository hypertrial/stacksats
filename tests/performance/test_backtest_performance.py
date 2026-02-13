"""Performance benchmark tests for backtest.py.

These tests are marked with @pytest.mark.performance and can be
skipped in regular test runs using: pytest -m "not performance"

Performance thresholds are set conservatively to avoid flaky tests
while still catching significant performance regressions.
"""

import sys
import time
from pathlib import Path

import pandas as pd
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import stacksats.backtest as backtest
from stacksats.backtest import compute_weights_modal
from stacksats.model_development import compute_weights_fast, precompute_features
from stacksats.prelude import compute_cycle_spd

# -----------------------------------------------------------------------------
# Feature Precomputation Performance Tests
# -----------------------------------------------------------------------------


@pytest.mark.performance
class TestFeaturePrecomputationPerformance:
    """Performance tests for feature precomputation."""

    def test_feature_precompute_time(self, sample_btc_df):
        """Benchmark: Feature precomputation should complete in < 5 seconds."""
        start = time.time()
        precompute_features(sample_btc_df)
        elapsed = time.time() - start

        # Should complete quickly (< 5 seconds for ~2000 days of data)
        assert elapsed < 5.0, (
            f"Feature precomputation took {elapsed:.2f}s, threshold: 5.0s"
        )

    def test_feature_precompute_is_reasonable(self, sample_btc_df):
        """Test that precompute time scales reasonably with data size."""
        # Time with half the data
        half_df = sample_btc_df.iloc[: len(sample_btc_df) // 2]

        start_half = time.time()
        precompute_features(half_df)
        time_half = time.time() - start_half

        # Time with full data
        start_full = time.time()
        precompute_features(sample_btc_df)
        time_full = time.time() - start_full

        # Full shouldn't take more than 4x the half time (allowing for overhead)
        assert time_full < time_half * 4, (
            f"Performance doesn't scale reasonably: half={time_half:.2f}s, "
            f"full={time_full:.2f}s"
        )


# -----------------------------------------------------------------------------
# Weight Computation Performance Tests
# -----------------------------------------------------------------------------


@pytest.mark.performance
class TestWeightComputationPerformance:
    """Performance tests for weight computation."""

    def test_single_window_weight_computation(self, sample_features_df):
        """Benchmark: Single window weight computation should be < 100ms."""
        start_date = pd.Timestamp("2024-01-01")
        end_date = pd.Timestamp("2024-12-31")

        if start_date not in sample_features_df.index:
            pytest.skip("Start date not in features index")
        if end_date not in sample_features_df.index:
            pytest.skip("End date not in features index")

        # Warm-up run
        compute_weights_fast(sample_features_df, start_date, end_date)

        # Timed run
        start = time.time()
        compute_weights_fast(sample_features_df, start_date, end_date)
        elapsed = time.time() - start

        assert elapsed < 0.1, (
            f"Single window computation took {elapsed * 1000:.1f}ms, threshold: 100ms"
        )

    def test_multiple_windows_weight_computation(self, sample_features_df):
        """Benchmark: 100 window computations should be < 5 seconds."""
        base_date = pd.Timestamp("2024-01-01")

        if base_date not in sample_features_df.index:
            pytest.skip("Base date not in features index")

        start = time.time()
        for i in range(100):
            start_date = base_date + pd.Timedelta(days=i)
            end_date = start_date + pd.Timedelta(days=30)

            if end_date > sample_features_df.index.max():
                break

            compute_weights_fast(sample_features_df, start_date, end_date)

        elapsed = time.time() - start

        assert elapsed < 5.0, (
            f"100 window computations took {elapsed:.2f}s, threshold: 5.0s"
        )

    def test_compute_weights_modal_performance(self, sample_features_df):
        """Benchmark: compute_weights_modal should be < 100ms per window."""
        backtest._FEATURES_DF = sample_features_df

        start_date = pd.Timestamp("2024-01-01")
        end_date = pd.Timestamp("2024-12-31")

        if start_date not in sample_features_df.index:
            pytest.skip("Start date not in features index")
        if end_date not in sample_features_df.index:
            pytest.skip("End date not in features index")

        window_feat = sample_features_df.loc[start_date:end_date]

        # Warm-up
        compute_weights_modal(window_feat)

        # Timed run
        start = time.time()
        compute_weights_modal(window_feat)
        elapsed = time.time() - start

        assert elapsed < 0.1, (
            f"compute_weights_modal took {elapsed * 1000:.1f}ms, threshold: 100ms"
        )


# -----------------------------------------------------------------------------
# Full Backtest Performance Tests
# -----------------------------------------------------------------------------


@pytest.mark.performance
class TestFullBacktestPerformance:
    """Performance tests for full backtest pipeline."""

    def test_backtest_completes_in_reasonable_time(
        self, sample_btc_df, sample_features_df
    ):
        """Benchmark: Full backtest should complete in < 60 seconds.

        Note: This test uses sample data which may have fewer windows
        than production. The threshold is set conservatively.
        """
        backtest._FEATURES_DF = sample_features_df

        start = time.time()
        spd_table = compute_cycle_spd(
            sample_btc_df, compute_weights_modal, features_df=sample_features_df
        )
        elapsed = time.time() - start

        num_windows = len(spd_table)

        # Calculate per-window time
        per_window_time = elapsed / max(num_windows, 1)

        # Should complete in reasonable time
        assert elapsed < 60.0, (
            f"Backtest with {num_windows} windows took {elapsed:.2f}s, threshold: 60s"
        )

        # Per-window time should be < 100ms on average
        assert per_window_time < 0.1, (
            f"Per-window time is {per_window_time * 1000:.1f}ms, threshold: 100ms"
        )


# -----------------------------------------------------------------------------
# Memory Efficiency Tests
# -----------------------------------------------------------------------------


@pytest.mark.performance
class TestMemoryEfficiency:
    """Tests for memory efficiency of computations."""

    def test_weight_computation_returns_appropriate_size(self, sample_features_df):
        """Test that weight computation returns appropriately sized output."""
        start_date = pd.Timestamp("2024-01-01")
        end_date = pd.Timestamp("2024-12-31")

        if start_date not in sample_features_df.index:
            pytest.skip("Start date not in features index")
        if end_date not in sample_features_df.index:
            pytest.skip("End date not in features index")

        weights = compute_weights_fast(sample_features_df, start_date, end_date)

        # Output size should match input range
        expected_len = len(sample_features_df.loc[start_date:end_date])
        assert len(weights) == expected_len

        # Memory footprint should be reasonable (< 1MB for weights)
        mem_bytes = weights.nbytes
        assert mem_bytes < 1_000_000, f"Weights use {mem_bytes} bytes, expected < 1MB"

    def test_features_df_size_is_reasonable(self, sample_btc_df):
        """Test that features DataFrame has reasonable memory footprint."""
        features_df = precompute_features(sample_btc_df)

        # Calculate memory usage
        mem_mb = features_df.memory_usage(deep=True).sum() / (1024 * 1024)

        # Should be < 100MB for typical data
        assert mem_mb < 100, f"Features use {mem_mb:.1f}MB, expected < 100MB"


# -----------------------------------------------------------------------------
# Scalability Tests
# -----------------------------------------------------------------------------


@pytest.mark.performance
class TestScalability:
    """Tests for scalability of computations."""

    def test_weight_computation_scales_linearly(self, sample_features_df):
        """Test that weight computation time scales linearly with window size."""
        base_date = pd.Timestamp("2024-01-01")

        if base_date not in sample_features_df.index:
            pytest.skip("Base date not in features index")

        times = []
        sizes = [30, 60, 90, 120]

        for days in sizes:
            end_date = base_date + pd.Timedelta(days=days)
            if end_date > sample_features_df.index.max():
                break

            start = time.time()
            compute_weights_fast(sample_features_df, base_date, end_date)
            elapsed = time.time() - start
            times.append((days, elapsed))

        if len(times) < 2:
            pytest.skip("Not enough data points for scalability test")

        # Check that time doesn't grow super-linearly
        # Time for 4x window should be < 10x the time for 1x window
        first_time = times[0][1]
        last_time = times[-1][1]
        size_ratio = times[-1][0] / times[0][0]

        # Allow for overhead, but should roughly scale
        assert last_time < first_time * size_ratio * 5, (
            f"Computation doesn't scale linearly: {times[0]} -> {times[-1]}"
        )

    def test_many_small_windows_performance(self, sample_features_df):
        """Test performance with many small (7-day) windows."""
        base_date = pd.Timestamp("2024-01-01")

        if base_date not in sample_features_df.index:
            pytest.skip("Base date not in features index")

        num_windows = 50
        window_size = 7

        start = time.time()
        for i in range(num_windows):
            start_date = base_date + pd.Timedelta(days=i)
            end_date = start_date + pd.Timedelta(days=window_size)

            if end_date > sample_features_df.index.max():
                break

            compute_weights_fast(sample_features_df, start_date, end_date)

        elapsed = time.time() - start

        # Many small windows should still be fast
        assert elapsed < 5.0, (
            f"{num_windows} small windows took {elapsed:.2f}s, threshold: 5.0s"
        )
