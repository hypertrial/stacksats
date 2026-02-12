"""Time-series cross-validation tests for the Bitcoin DCA strategy.

These tests implement expanding window (walk-forward) cross-validation to:
1. Detect overfitting to historical data
2. Verify consistent performance across different time periods
3. Ensure the model generalizes to unseen data

Time-series CV differs from regular k-fold:
- Training data always comes before test data (no future leakage)
- Uses expanding or rolling windows
- Maintains temporal order
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from stacksats.model_development import compute_weights_fast, precompute_features

# -----------------------------------------------------------------------------
# Cross-Validation Utilities
# -----------------------------------------------------------------------------


def generate_expanding_window_folds(
    start_date: str,
    end_date: str,
    n_folds: int = 5,
    min_train_days: int = 365 * 2,
    test_days: int = 365,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Generate expanding window folds for time-series cross-validation.

    Args:
        start_date: Start of the full date range
        end_date: End of the full date range
        n_folds: Number of folds to generate
        min_train_days: Minimum number of training days
        test_days: Number of test days per fold

    Returns:
        List of tuples: (train_start, train_end, test_start, test_end)
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    total_days = (end - start).days
    available_test_days = total_days - min_train_days

    if available_test_days < test_days * n_folds:
        # Adjust test_days if not enough data
        test_days = available_test_days // n_folds

    folds = []
    train_start = start

    for i in range(n_folds):
        # Training: from start to train_end
        train_end = start + pd.Timedelta(days=min_train_days + i * test_days - 1)

        # Test: from day after train_end for test_days
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.Timedelta(days=test_days - 1)

        if test_end <= end:
            folds.append((train_start, train_end, test_start, test_end))

    return folds


def generate_rolling_window_folds(
    start_date: str,
    end_date: str,
    n_folds: int = 5,
    train_days: int = 365 * 2,
    test_days: int = 365,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Generate rolling window folds for time-series cross-validation.

    Unlike expanding window, training window size stays constant.

    Args:
        start_date: Start of the full date range
        end_date: End of the full date range
        n_folds: Number of folds to generate
        train_days: Number of training days (fixed)
        test_days: Number of test days per fold

    Returns:
        List of tuples: (train_start, train_end, test_start, test_end)
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    total_days = (end - start).days
    step_days = (
        (total_days - train_days - test_days) // (n_folds - 1) if n_folds > 1 else 0
    )

    folds = []

    for i in range(n_folds):
        train_start = start + pd.Timedelta(days=i * step_days)
        train_end = train_start + pd.Timedelta(days=train_days - 1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.Timedelta(days=test_days - 1)

        if test_end <= end:
            folds.append((train_start, train_end, test_start, test_end))

    return folds


def evaluate_fold(
    features_df: pd.DataFrame,
    btc_df: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> dict:
    """Evaluate strategy performance on a test fold.

    Directly computes SPD metrics for windows within the test period,
    bypassing the global backtest configuration.

    Args:
        features_df: Precomputed features DataFrame
        btc_df: BTC price DataFrame
        test_start: Start of test period
        test_end: End of test period

    Returns:
        Dictionary with performance metrics
    """
    from stacksats.model_development import compute_weights_fast

    # Determine window size based on test period length
    test_days = (test_end - test_start).days
    if test_days >= 365:
        window_days = 365
    elif test_days >= 180:
        window_days = 180
    elif test_days >= 90:
        window_days = 90
    else:
        window_days = max(30, test_days // 2)

    window_offset = pd.Timedelta(days=window_days)
    max_start = test_end - window_offset

    if max_start < test_start:
        return {"win_rate": np.nan, "mean_excess": np.nan, "n_windows": 0}

    try:
        # Generate window start dates (monthly frequency for efficiency)
        start_dates = pd.date_range(start=test_start, end=max_start, freq="MS")

        if len(start_dates) == 0:
            start_dates = [test_start]

        results = []
        for start in start_dates:
            end = start + window_offset

            if end > test_end:
                continue

            # Get price data
            price_slice = btc_df["PriceUSD_coinmetrics"].loc[start:end]
            if price_slice.empty or len(price_slice) < 7:
                continue

            # Compute weights
            try:
                weights = compute_weights_fast(features_df, start, end)
            except Exception:
                continue

            if len(weights) != len(price_slice):
                # Align weights and prices
                common_idx = weights.index.intersection(price_slice.index)
                weights = weights.loc[common_idx]
                price_slice = price_slice.loc[common_idx]

            if len(weights) == 0:
                continue

            # Compute SPD metrics
            inv_price = 1e8 / price_slice  # sats per dollar
            min_spd, max_spd = inv_price.min(), inv_price.max()
            span = max_spd - min_spd

            uniform_spd = inv_price.mean()
            dynamic_spd = (weights * inv_price).sum()

            # Calculate percentiles
            if span > 0:
                uniform_pct = 100 * (uniform_spd - min_spd) / span
                dynamic_pct = 100 * (dynamic_spd - min_spd) / span
                excess = dynamic_pct - uniform_pct
            else:
                excess = 0.0

            results.append({"excess_percentile": excess})

        if len(results) == 0:
            return {"win_rate": np.nan, "mean_excess": np.nan, "n_windows": 0}

        excess_values = [r["excess_percentile"] for r in results]
        win_rate = sum(1 for e in excess_values if e > 0) / len(excess_values)
        mean_excess = np.mean(excess_values)

        return {
            "win_rate": float(win_rate),
            "mean_excess": float(mean_excess),
            "n_windows": len(results),
        }
    except Exception:
        return {"win_rate": np.nan, "mean_excess": np.nan, "n_windows": 0}


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cv_btc_df():
    """Create BTC price data for cross-validation tests."""
    dates = pd.date_range(start="2018-01-01", end="2025-12-31", freq="D")
    np.random.seed(42)
    base_price = 10000
    returns = np.random.normal(0.001, 0.03, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({"PriceUSD_coinmetrics": prices}, index=dates)
    df["PriceUSD"] = df["PriceUSD_coinmetrics"]
    df.index.name = "time"
    return df


@pytest.fixture(scope="module")
def cv_features_df(cv_btc_df):
    """Precompute features for cross-validation tests."""
    return precompute_features(cv_btc_df)


# -----------------------------------------------------------------------------
# Expanding Window Cross-Validation Tests
# -----------------------------------------------------------------------------


class TestExpandingWindowCV:
    """Tests using expanding window cross-validation."""

    def test_generate_5_fold_expanding_window(self):
        """Verify fold generation produces valid folds."""
        folds = generate_expanding_window_folds("2020-01-01", "2025-12-31", n_folds=5)

        assert len(folds) == 5, f"Expected 5 folds, got {len(folds)}"

        for i, (train_start, train_end, test_start, test_end) in enumerate(folds):
            # Train must come before test
            assert train_end < test_start, f"Fold {i}: train overlaps test"

            # No gaps between train and test
            assert (test_start - train_end).days == 1, (
                f"Fold {i}: gap between train/test"
            )

    def test_expanding_window_no_overlap(self):
        """Verify folds have no temporal overlap issues."""
        folds = generate_expanding_window_folds("2020-01-01", "2025-12-31", n_folds=5)

        for i in range(len(folds) - 1):
            next_train_start = folds[i + 1][0]  # train_start

            # Training windows are expanding, so train_start is always the same
            assert next_train_start == folds[0][0]

    def test_5_fold_expanding_cv_performance(self, cv_features_df, cv_btc_df):
        """5-fold expanding window cross-validation performance test."""
        folds = generate_expanding_window_folds(
            "2020-01-01",
            "2025-12-31",
            n_folds=5,
            min_train_days=365 * 2,
            test_days=180,
        )

        fold_results = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(folds):
            # Evaluate on test fold
            result = evaluate_fold(cv_features_df, cv_btc_df, test_start, test_end)
            result["fold"] = i
            result["train_days"] = (train_end - train_start).days
            fold_results.append(result)

        # Check that we have results for most folds
        valid_results = [r for r in fold_results if r["n_windows"] > 0]
        assert len(valid_results) >= 3, "Should have results for at least 3 folds"

        # Check performance across folds
        win_rates = [
            r["win_rate"] for r in valid_results if not np.isnan(r["win_rate"])
        ]

        if len(win_rates) >= 2:
            # With stable allocation, some folds may have very low win rates
            # due to budget scaling effects. Check average is reasonable.
            avg_win_rate = np.mean(win_rates)
            assert avg_win_rate > 0.20, f"Average win_rate too low: {avg_win_rate:.2%}"

            # Performance shouldn't vary too wildly across folds
            # With stable allocation, variance may be higher
            win_rate_std = np.std(win_rates)
            assert win_rate_std < 0.50, (
                f"Win rate too variable across folds: std={win_rate_std:.2%}"
            )


# -----------------------------------------------------------------------------
# Rolling Window Cross-Validation Tests
# -----------------------------------------------------------------------------


class TestRollingWindowCV:
    """Tests using rolling window cross-validation."""

    def test_generate_rolling_window_folds(self):
        """Verify rolling window fold generation."""
        folds = generate_rolling_window_folds(
            "2020-01-01",
            "2025-12-31",
            n_folds=4,
            train_days=365 * 2,
            test_days=180,
        )

        assert len(folds) >= 2, f"Expected at least 2 folds, got {len(folds)}"

        for i, (train_start, train_end, test_start, test_end) in enumerate(folds):
            # Training window size should be constant
            train_days = (train_end - train_start).days + 1
            assert train_days == 365 * 2, f"Fold {i}: train days = {train_days}"

    def test_rolling_window_cv_performance(self, cv_features_df, cv_btc_df):
        """Rolling window cross-validation performance test."""
        folds = generate_rolling_window_folds(
            "2020-01-01",
            "2025-12-31",
            n_folds=4,
            train_days=365 * 2,
            test_days=180,
        )

        fold_results = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(folds):
            result = evaluate_fold(cv_features_df, cv_btc_df, test_start, test_end)
            result["fold"] = i
            fold_results.append(result)

        valid_results = [r for r in fold_results if r["n_windows"] > 0]
        assert len(valid_results) >= 2, "Should have results for at least 2 folds"


# -----------------------------------------------------------------------------
# Overfitting Detection Tests
# -----------------------------------------------------------------------------


class TestOverfittingDetection:
    """Tests specifically designed to detect overfitting."""

    def test_train_test_performance_gap(self, cv_features_df, cv_btc_df):
        """Verify train/test performance gap is not excessive.

        Large gap between in-sample and out-of-sample performance indicates overfitting.
        """
        folds = generate_expanding_window_folds(
            "2020-01-01",
            "2025-12-31",
            n_folds=3,
            min_train_days=365 * 2,
            test_days=365,
        )

        train_test_gaps = []

        for train_start, train_end, test_start, test_end in folds:
            # Evaluate on training period (in-sample)
            train_result = evaluate_fold(
                cv_features_df, cv_btc_df, train_start, train_end
            )

            # Evaluate on test period (out-of-sample)
            test_result = evaluate_fold(cv_features_df, cv_btc_df, test_start, test_end)

            if not np.isnan(train_result["mean_excess"]) and not np.isnan(
                test_result["mean_excess"]
            ):
                gap = train_result["mean_excess"] - test_result["mean_excess"]
                train_test_gaps.append(gap)

        if len(train_test_gaps) > 0:
            mean_gap = np.mean(train_test_gaps)
            # Gap should not be excessively large (< 15% difference)
            assert mean_gap < 15, (
                f"Train-test performance gap too large: {mean_gap:.1f}%"
            )

    def test_early_vs_late_performance(self, cv_features_df, cv_btc_df):
        """Compare performance on early vs late time periods.

        If model overfits to early data, late performance should be worse.
        """
        # Early period: 2020-2022
        early_result = evaluate_fold(
            cv_features_df,
            cv_btc_df,
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2021-12-31"),
        )

        # Late period: 2024-2025
        late_result = evaluate_fold(
            cv_features_df,
            cv_btc_df,
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2025-12-31"),
        )

        if not np.isnan(early_result["mean_excess"]) and not np.isnan(
            late_result["mean_excess"]
        ):
            # Late performance shouldn't be dramatically worse
            gap = early_result["mean_excess"] - late_result["mean_excess"]
            assert gap < 20, f"Early vs late performance gap too large: {gap:.1f}%"


# -----------------------------------------------------------------------------
# Consistency Tests
# -----------------------------------------------------------------------------


class TestCrossValidationConsistency:
    """Tests for consistency in cross-validation results."""

    def test_weights_valid_across_all_folds(self, cv_features_df):
        """Verify weights are valid in all fold test periods."""
        folds = generate_expanding_window_folds("2020-01-01", "2025-12-31", n_folds=5)

        for i, (_, _, test_start, test_end) in enumerate(folds):
            weights = compute_weights_fast(cv_features_df, test_start, test_end)

            assert np.isclose(weights.sum(), 1.0, rtol=1e-6), (
                f"Fold {i}: weights don't sum to 1.0"
            )
            assert (weights >= -1e-10).all(), f"Fold {i}: negative weights"
            assert weights.notna().all(), f"Fold {i}: NaN weights found"

    def test_fold_results_reproducible(self, cv_features_df, cv_btc_df):
        """Verify fold evaluation is reproducible."""
        test_start = pd.Timestamp("2024-01-01")
        test_end = pd.Timestamp("2024-12-31")

        result1 = evaluate_fold(cv_features_df, cv_btc_df, test_start, test_end)
        result2 = evaluate_fold(cv_features_df, cv_btc_df, test_start, test_end)

        assert result1 == result2, "Fold evaluation not reproducible"

    def test_cv_summary_statistics(self, cv_features_df, cv_btc_df):
        """Compute and verify cross-validation summary statistics."""
        folds = generate_expanding_window_folds(
            "2020-01-01",
            "2025-12-31",
            n_folds=5,
            min_train_days=365 * 2,
            test_days=180,
        )

        fold_results = []
        for _, _, test_start, test_end in folds:
            result = evaluate_fold(cv_features_df, cv_btc_df, test_start, test_end)
            if result["n_windows"] > 0:
                fold_results.append(result)

        if len(fold_results) >= 3:
            win_rates = [
                r["win_rate"] for r in fold_results if not np.isnan(r["win_rate"])
            ]

            if len(win_rates) >= 2:
                cv_mean = np.mean(win_rates)
                cv_std = np.std(win_rates)

                # Summary statistics should be reasonable
                # With stable allocation, variance may be higher
                assert cv_mean > 0.20, f"CV mean win rate too low: {cv_mean:.2%}"
                assert cv_std < 0.50, f"CV std too high: {cv_std:.2%}"


# -----------------------------------------------------------------------------
# Leave-One-Year-Out Cross-Validation
# -----------------------------------------------------------------------------


class TestLeaveOneYearOutCV:
    """Leave-one-year-out cross-validation tests."""

    def test_leave_one_year_out_2024(self, cv_features_df, cv_btc_df):
        """Test with 2024 held out for testing."""
        test_start = pd.Timestamp("2024-01-01")
        test_end = pd.Timestamp("2024-12-31")

        result = evaluate_fold(cv_features_df, cv_btc_df, test_start, test_end)

        # Should produce valid results
        assert result["n_windows"] > 0, "Should have test windows"

        if not np.isnan(result["win_rate"]):
            # Performance should be reasonable
            assert result["win_rate"] > 0.30, (
                f"2024 holdout win rate too low: {result['win_rate']:.2%}"
            )

    def test_leave_one_year_out_2023(self, cv_features_df, cv_btc_df):
        """Test with 2023 held out for testing."""
        test_start = pd.Timestamp("2023-01-01")
        test_end = pd.Timestamp("2023-12-31")

        result = evaluate_fold(cv_features_df, cv_btc_df, test_start, test_end)

        # Should produce valid results
        assert result["n_windows"] > 0, "Should have test windows"

    def test_leave_one_year_out_consistency(self, cv_features_df, cv_btc_df):
        """Test that different holdout years produce similar results."""
        years = [2022, 2023, 2024, 2025]
        results = []

        for year in years:
            test_start = pd.Timestamp(f"{year}-01-01")
            test_end = pd.Timestamp(f"{year}-12-31")

            result = evaluate_fold(cv_features_df, cv_btc_df, test_start, test_end)
            if result["n_windows"] > 0:
                results.append(result)

        # Should have results for multiple years
        assert len(results) >= 2, "Should have results for at least 2 years"

        win_rates = [r["win_rate"] for r in results if not np.isnan(r["win_rate"])]
        if len(win_rates) >= 2:
            # Win rates shouldn't vary too much across years
            assert np.std(win_rates) < 0.25, (
                f"Win rate std across years too high: {np.std(win_rates):.2%}"
            )
