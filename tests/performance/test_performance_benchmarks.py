"""Performance benchmark tests for the Bitcoin DCA strategy system.

These tests track computation speed and detect performance regressions.
Use pytest-benchmark to run: pytest tests/test_performance_benchmarks.py -v

Tests are split into two categories:
1. Benchmark tests (require pytest-benchmark) - skipped if not installed
2. Threshold tests (use time.perf_counter) - always run
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from stacksats.export_weights import generate_date_ranges, process_start_date_batch
from stacksats.model_development import (
    allocate_sequential_stable,
    compute_dynamic_multiplier,
    compute_weights_fast,
    precompute_features,
    softmax,
)

# Check if pytest-benchmark is available
try:
    import pytest_benchmark  # noqa: F401

    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False

# Skip marker for benchmark-dependent tests only
requires_benchmark = pytest.mark.skipif(
    not BENCHMARK_AVAILABLE,
    reason="pytest-benchmark not installed. Install with: pip install pytest-benchmark",
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def benchmark_btc_df():
    """Create BTC price data for benchmarking."""
    dates = pd.date_range(start="2020-01-01", end="2025-12-31", freq="D")
    np.random.seed(42)
    base_price = 10000
    returns = np.random.normal(0.001, 0.03, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({"PriceUSD_coinmetrics": prices}, index=dates)
    df["PriceUSD"] = df["PriceUSD_coinmetrics"]
    df.index.name = "time"
    return df


@pytest.fixture(scope="module")
def benchmark_features_df(benchmark_btc_df):
    """Precompute features for benchmarking."""
    return precompute_features(benchmark_btc_df)


# -----------------------------------------------------------------------------
# Core Function Benchmarks
# -----------------------------------------------------------------------------


@requires_benchmark
@pytest.mark.performance
class TestPrecomputeFeaturesBenchmark:
    """Benchmark tests for precompute_features function."""

    def test_precompute_features_speed(self, benchmark_btc_df, benchmark):
        """Benchmark feature precomputation for 6 years of data."""
        if not BENCHMARK_AVAILABLE:
            pytest.skip("pytest-benchmark not installed")

        result = benchmark(precompute_features, benchmark_btc_df)

        # Verify output is valid
        assert len(result) > 0
        assert "mvrv_zscore" in result.columns
        assert "mvrv_gradient" in result.columns


@requires_benchmark
@pytest.mark.performance
class TestComputeWeightsBenchmark:
    """Benchmark tests for weight computation functions."""

    def test_compute_weights_fast_1_year(self, benchmark_features_df, benchmark):
        """Benchmark weight computation for 1-year window."""
        if not BENCHMARK_AVAILABLE:
            pytest.skip("pytest-benchmark not installed")

        start_date = pd.Timestamp("2024-01-01")
        end_date = pd.Timestamp("2024-12-31")

        result = benchmark(
            compute_weights_fast, benchmark_features_df, start_date, end_date
        )

        # Verify output
        assert len(result) == 366  # 2024 is leap year
        assert np.isclose(result.sum(), 1.0, rtol=1e-6)

    def test_compute_weights_fast_6_months(self, benchmark_features_df, benchmark):
        """Benchmark weight computation for 6-month window."""
        if not BENCHMARK_AVAILABLE:
            pytest.skip("pytest-benchmark not installed")

        start_date = pd.Timestamp("2024-01-01")
        end_date = pd.Timestamp("2024-06-30")

        result = benchmark(
            compute_weights_fast, benchmark_features_df, start_date, end_date
        )

        # Verify output
        assert len(result) > 0
        assert np.isclose(result.sum(), 1.0, rtol=1e-6)

    def test_compute_weights_fast_1_month(self, benchmark_features_df, benchmark):
        """Benchmark weight computation for 1-month window."""
        if not BENCHMARK_AVAILABLE:
            pytest.skip("pytest-benchmark not installed")

        start_date = pd.Timestamp("2024-06-01")
        end_date = pd.Timestamp("2024-06-30")

        result = benchmark(
            compute_weights_fast, benchmark_features_df, start_date, end_date
        )

        # Verify output
        assert len(result) == 30
        assert np.isclose(result.sum(), 1.0, rtol=1e-6)


@requires_benchmark
@pytest.mark.performance
class TestMathFunctionsBenchmark:
    """Benchmark tests for mathematical utility functions."""

    def test_softmax_speed(self, benchmark):
        """Benchmark softmax function."""
        if not BENCHMARK_AVAILABLE:
            pytest.skip("pytest-benchmark not installed")

        x = np.random.randn(100)
        result = benchmark(softmax, x)

        assert np.isclose(result.sum(), 1.0)

    def test_allocate_sequential_stable_speed(self, benchmark):
        """Benchmark allocate_sequential_stable function."""
        if not BENCHMARK_AVAILABLE:
            pytest.skip("pytest-benchmark not installed")

        raw = np.random.exponential(1, 366)
        result = benchmark(allocate_sequential_stable, raw, n_past=366)

        assert np.isclose(result.sum(), 1.0)

    def test_dynamic_multiplier_speed(self, benchmark):
        """Benchmark compute_dynamic_multiplier function."""
        if not BENCHMARK_AVAILABLE:
            pytest.skip("pytest-benchmark not installed")

        price_vs_ma = np.random.uniform(-1, 1, 366)
        mvrv_zscore = np.random.uniform(-4, 4, 366)
        mvrv_gradient = np.random.uniform(-1, 1, 366)
        result = benchmark(
            compute_dynamic_multiplier, price_vs_ma, mvrv_zscore, mvrv_gradient
        )

        assert len(result) == 366
        assert (result > 0).all()


@requires_benchmark
@pytest.mark.performance
class TestBatchProcessingBenchmark:
    """Benchmark tests for batch processing functions."""

    def test_process_single_date_range(
        self, benchmark_features_df, benchmark_btc_df, benchmark
    ):
        """Benchmark processing a single date range."""
        if not BENCHMARK_AVAILABLE:
            pytest.skip("pytest-benchmark not installed")

        start_date = pd.Timestamp("2025-01-01")
        end_date = pd.Timestamp("2025-12-31")
        current_date = pd.Timestamp("2025-12-31")

        result = benchmark(
            process_start_date_batch,
            start_date,
            [end_date],
            benchmark_features_df,
            benchmark_btc_df,
            current_date,
            "PriceUSD_coinmetrics",
        )

        assert len(result) > 0
        assert "weight" in result.columns

    def test_process_multiple_date_ranges(
        self, benchmark_features_df, benchmark_btc_df, benchmark
    ):
        """Benchmark processing multiple valid ranges from same start date."""
        if not BENCHMARK_AVAILABLE:
            pytest.skip("pytest-benchmark not installed")

        start_date = pd.Timestamp("2024-01-01")
        end_dates = [pd.Timestamp("2024-12-30"), pd.Timestamp("2024-12-31")]
        current_date = pd.Timestamp("2025-12-31")

        result = benchmark(
            process_start_date_batch,
            start_date,
            end_dates,
            benchmark_features_df,
            benchmark_btc_df,
            current_date,
            "PriceUSD_coinmetrics",
        )

        assert len(result) > 0

    def test_generate_date_ranges_speed(self, benchmark):
        """Benchmark date range generation."""
        if not BENCHMARK_AVAILABLE:
            pytest.skip("pytest-benchmark not installed")

        result = benchmark(generate_date_ranges, "2025-12-01", "2027-12-31", 120)

        assert len(result) > 0


# -----------------------------------------------------------------------------
# Performance Threshold Tests (without benchmark fixture)
# These tests run even without pytest-benchmark installed
# -----------------------------------------------------------------------------


class TestPerformanceThresholds:
    """Tests that verify performance meets minimum thresholds.

    These tests work without pytest-benchmark by using time.perf_counter().
    """

    @pytest.fixture
    def perf_btc_df(self):
        """Create BTC price data for performance tests."""
        dates = pd.date_range(start="2020-01-01", end="2025-12-31", freq="D")
        np.random.seed(42)
        base_price = 10000
        returns = np.random.normal(0.001, 0.03, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame({"PriceUSD_coinmetrics": prices}, index=dates)
        df["PriceUSD"] = df["PriceUSD_coinmetrics"]
        df.index.name = "time"
        return df

    @pytest.fixture
    def perf_features_df(self, perf_btc_df):
        """Precompute features for performance tests."""
        return precompute_features(perf_btc_df)

    def test_precompute_features_under_1_second(self, perf_btc_df):
        """Verify feature precomputation completes in under 1 second."""
        start = time.perf_counter()
        _ = precompute_features(perf_btc_df)
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, f"precompute_features took {elapsed:.2f}s, expected < 1s"

    def test_compute_weights_under_500ms(self, perf_features_df):
        """Verify weight computation completes in under 500ms.

        Note: First call may be slower due to JIT compilation and cache warming.
        """
        start_date = pd.Timestamp("2024-01-01")
        end_date = pd.Timestamp("2024-12-31")

        # Warm up - first call is often slower
        _ = compute_weights_fast(perf_features_df, start_date, end_date)

        start = time.perf_counter()
        _ = compute_weights_fast(perf_features_df, start_date, end_date)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5, (
            f"compute_weights_fast took {elapsed:.3f}s, expected < 0.5s"
        )

    def test_1000_weight_computations_under_10_seconds(self, perf_features_df):
        """Verify 1000 weight computations complete in under 10 seconds."""
        start_date = pd.Timestamp("2024-06-01")
        end_date = pd.Timestamp("2024-06-30")

        start = time.perf_counter()
        for _ in range(1000):
            _ = compute_weights_fast(perf_features_df, start_date, end_date)
        elapsed = time.perf_counter() - start

        assert elapsed < 10.0, (
            f"1000 weight computations took {elapsed:.2f}s, expected < 10s"
        )

    def test_batch_processing_under_1_second(self, perf_features_df, perf_btc_df):
        """Verify batch processing valid ranges completes in under 1 second."""
        start_date = pd.Timestamp("2024-01-01")
        end_dates = [pd.Timestamp("2024-12-30"), pd.Timestamp("2024-12-31")]
        current_date = pd.Timestamp("2025-12-31")

        start = time.perf_counter()
        _ = process_start_date_batch(
            start_date,
            end_dates,
            perf_features_df,
            perf_btc_df,
            current_date,
            "PriceUSD_coinmetrics",
        )
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, (
            f"Batch processing 10 ranges took {elapsed:.2f}s, expected < 1s"
        )


# -----------------------------------------------------------------------------
# Memory Usage Tests
# These tests run even without pytest-benchmark installed
# -----------------------------------------------------------------------------


class TestMemoryUsage:
    """Tests that verify reasonable memory usage."""

    @pytest.fixture
    def mem_btc_df(self):
        """Create BTC price data for memory tests."""
        dates = pd.date_range(start="2020-01-01", end="2025-12-31", freq="D")
        np.random.seed(42)
        base_price = 10000
        returns = np.random.normal(0.001, 0.03, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame({"PriceUSD_coinmetrics": prices}, index=dates)
        df["PriceUSD"] = df["PriceUSD_coinmetrics"]
        df.index.name = "time"
        return df

    @pytest.fixture
    def mem_features_df(self, mem_btc_df):
        """Precompute features for memory tests."""
        return precompute_features(mem_btc_df)

    def test_features_dataframe_size(self, mem_features_df):
        """Verify features DataFrame has reasonable size."""
        # Get memory usage in MB
        memory_mb = mem_features_df.memory_usage(deep=True).sum() / (1024 * 1024)

        # Should be less than 50MB for 6 years of data
        assert memory_mb < 50, (
            f"Features DataFrame uses {memory_mb:.1f}MB, expected < 50MB"
        )

    def test_weights_series_size(self, mem_features_df):
        """Verify weights Series has reasonable size."""
        start_date = pd.Timestamp("2024-01-01")
        end_date = pd.Timestamp("2024-12-31")

        weights = compute_weights_fast(mem_features_df, start_date, end_date)

        # Get memory usage in KB
        memory_kb = weights.memory_usage(deep=True) / 1024

        # Should be less than 10KB for 1 year
        assert memory_kb < 10, f"Weights Series uses {memory_kb:.1f}KB, expected < 10KB"


# -----------------------------------------------------------------------------
# Scaling Tests
# These tests run even without pytest-benchmark installed
# -----------------------------------------------------------------------------


class TestScaling:
    """Tests that verify computation scales reasonably with input size."""

    @pytest.fixture
    def scale_btc_df(self):
        """Create BTC price data for scaling tests."""
        dates = pd.date_range(start="2020-01-01", end="2025-12-31", freq="D")
        np.random.seed(42)
        base_price = 10000
        returns = np.random.normal(0.001, 0.03, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame({"PriceUSD_coinmetrics": prices}, index=dates)
        df["PriceUSD"] = df["PriceUSD_coinmetrics"]
        df.index.name = "time"
        return df

    @pytest.fixture
    def scale_features_df(self, scale_btc_df):
        """Precompute features for scaling tests."""
        return precompute_features(scale_btc_df)

    def test_weight_computation_scales_linearly(self, scale_features_df):
        """Verify weight computation time scales roughly linearly with window size."""
        # Test different window sizes
        sizes = [30, 90, 180, 365]
        times = []

        base_date = pd.Timestamp("2024-01-01")

        for days in sizes:
            end_date = base_date + pd.Timedelta(days=days - 1)

            start = time.perf_counter()
            for _ in range(100):  # Average over 100 runs
                _ = compute_weights_fast(scale_features_df, base_date, end_date)
            elapsed = time.perf_counter() - start
            times.append(elapsed / 100)

        # Check that time doesn't grow faster than O(n^2)
        # (actual should be closer to O(n))
        ratio_30_to_365 = times[-1] / times[0]
        size_ratio = sizes[-1] / sizes[0]

        # Allow up to quadratic scaling (O(n^2))
        max_allowed_ratio = size_ratio**2

        assert ratio_30_to_365 < max_allowed_ratio, (
            f"Time scaling {ratio_30_to_365:.1f}x exceeds quadratic {max_allowed_ratio:.1f}x"
        )

    def test_feature_computation_scales_linearly(self, scale_btc_df):
        """Verify feature computation time scales roughly linearly with data size."""
        # Test with different data sizes
        sizes = [365, 730, 1461]  # 1 year, 2 years, 4 years
        times = []

        for days in sizes:
            subset = scale_btc_df.iloc[:days]

            start = time.perf_counter()
            _ = precompute_features(subset)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        # Check that time doesn't grow faster than O(n^2)
        ratio_1_to_4 = times[-1] / times[0]
        size_ratio = sizes[-1] / sizes[0]

        # Allow up to quadratic scaling
        max_allowed_ratio = size_ratio**2

        assert ratio_1_to_4 < max_allowed_ratio, (
            f"Time scaling {ratio_1_to_4:.1f}x exceeds quadratic {max_allowed_ratio:.1f}x"
        )
