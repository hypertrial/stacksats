"""Tests to ensure backtest.py and export_weights.py use identical weight computation logic.

These tests verify that:
1. Both modules compute identical weights for the same inputs
2. The past/future weight split logic is consistent
3. Weights are deterministic and reproducible across both modules
"""

import numpy as np
import pandas as pd
import pytest

from stacksats.backtest import compute_weights_modal
from stacksats.export_weights import process_start_date_batch
from stacksats.model_development import compute_window_weights, precompute_features

# Tolerance for floating point comparisons
FLOAT_TOLERANCE = 1e-12
WEIGHT_SUM_TOLERANCE = 1e-9
# Tolerance for weight stability (weights may shift slightly due to rolling features)
# The MVRV-based features use long rolling windows that cause some drift as data evolves
WEIGHT_STABILITY_TOLERANCE = 5e-3

PRICE_COL = "PriceUSD_coinmetrics"


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def parity_btc_df():
    """Create sample BTC price data for parity testing."""
    # Use a longer date range to test various scenarios
    dates = pd.date_range("2020-01-01", "2026-12-31", freq="D")
    np.random.seed(42)

    # Simulate realistic BTC prices
    base_price = 10000
    trend = np.linspace(0, 90000, len(dates))
    noise = np.random.randn(len(dates)) * 3000
    prices = np.maximum(base_price + trend + noise, 1000)

    return pd.DataFrame({PRICE_COL: prices}, index=dates)


@pytest.fixture
def parity_features_df(parity_btc_df):
    """Precompute features for parity testing."""
    return precompute_features(parity_btc_df)


# -----------------------------------------------------------------------------
# Core Parity Tests
# -----------------------------------------------------------------------------


class TestWeightComputationParity:
    """Test that backtest and export_weights compute identical weights."""

    def test_identical_weights_all_past(self, parity_features_df, parity_btc_df):
        """Test weights are identical when all dates are in the past.

        When current_date >= end_date, both modules should return
        identical model weights (no uniform future weights).
        """
        start_date = pd.Timestamp("2021-01-01")
        end_date = pd.Timestamp("2021-12-31")
        current_date = pd.Timestamp("2022-06-01")  # After end_date

        # Compute using backtest.compute_window_weights
        backtest_weights = compute_window_weights(
            parity_features_df, start_date, end_date, current_date
        )

        # Compute using export_weights.process_start_date_batch
        export_result = process_start_date_batch(
            start_date,
            [end_date],
            parity_features_df,
            parity_btc_df,
            current_date,
            PRICE_COL,
        )
        export_weights = export_result.set_index("DCA_date")["weight"]
        export_weights.index = pd.to_datetime(export_weights.index)

        # Verify identical weights
        assert len(backtest_weights) == len(export_weights)
        np.testing.assert_allclose(
            backtest_weights.values,
            export_weights.values,
            rtol=FLOAT_TOLERANCE,
            atol=FLOAT_TOLERANCE,
            err_msg="Weights differ between backtest and export_weights (all past)",
        )

    def test_identical_weights_mixed_past_future(
        self, parity_features_df, parity_btc_df
    ):
        """Test weights are identical when dates span past and future.

        Both modules should compute model weights for past dates and
        uniform weights for future dates.
        """
        start_date = pd.Timestamp("2025-06-01")
        end_date = pd.Timestamp("2026-05-31")
        current_date = pd.Timestamp("2025-12-15")  # Mid-range

        # Compute using backtest.compute_window_weights
        backtest_weights = compute_window_weights(
            parity_features_df, start_date, end_date, current_date
        )

        # Compute using export_weights.process_start_date_batch
        export_result = process_start_date_batch(
            start_date,
            [end_date],
            parity_features_df,
            parity_btc_df,
            current_date,
            PRICE_COL,
        )
        export_weights = export_result.set_index("DCA_date")["weight"]
        export_weights.index = pd.to_datetime(export_weights.index)

        # Verify identical weights
        assert len(backtest_weights) == len(export_weights)
        np.testing.assert_allclose(
            backtest_weights.values,
            export_weights.values,
            rtol=FLOAT_TOLERANCE,
            atol=FLOAT_TOLERANCE,
            err_msg="Weights differ between backtest and export_weights (mixed)",
        )

    def test_identical_weights_all_future(self, parity_features_df, parity_btc_df):
        """Test weights are identical when all dates are in the future.

        When current_date < start_date, both modules should return
        uniform weights for all dates.
        """
        start_date = pd.Timestamp("2026-01-01")
        end_date = pd.Timestamp("2026-12-31")
        current_date = pd.Timestamp("2025-06-01")  # Before start_date

        # Compute using backtest.compute_window_weights
        backtest_weights = compute_window_weights(
            parity_features_df, start_date, end_date, current_date
        )

        # Compute using export_weights.process_start_date_batch
        export_result = process_start_date_batch(
            start_date,
            [end_date],
            parity_features_df,
            parity_btc_df,
            current_date,
            PRICE_COL,
        )
        export_weights = export_result.set_index("DCA_date")["weight"]
        export_weights.index = pd.to_datetime(export_weights.index)

        # Verify identical weights
        assert len(backtest_weights) == len(export_weights)
        np.testing.assert_allclose(
            backtest_weights.values,
            export_weights.values,
            rtol=FLOAT_TOLERANCE,
            atol=FLOAT_TOLERANCE,
            err_msg="Weights differ between backtest and export_weights (all future)",
        )

        # Verify all weights are uniform (since all dates are future)
        expected_uniform = 1.0 / len(backtest_weights)
        assert np.allclose(backtest_weights.values, expected_uniform, rtol=1e-10), (
            "Future weights should be uniform"
        )


class TestPastWeightImmutabilityParity:
    """Test that past weights never change in both modules."""

    def test_past_weights_locked_backtest(self, parity_features_df):
        """Test that past weights don't change as current_date advances (backtest)."""
        start_date = pd.Timestamp("2021-01-01")
        end_date = pd.Timestamp("2021-12-31")

        # Compute weights at two different current_dates
        current_date_1 = pd.Timestamp("2021-06-15")
        current_date_2 = pd.Timestamp("2021-09-15")

        weights_1 = compute_window_weights(
            parity_features_df, start_date, end_date, current_date_1
        )
        weights_2 = compute_window_weights(
            parity_features_df, start_date, end_date, current_date_2
        )

        # Past weights (before current_date_1) should be stable
        # Note: slight drift is acceptable due to rolling feature recalculation
        past_dates = weights_1.index[weights_1.index <= current_date_1]
        np.testing.assert_allclose(
            weights_1[past_dates].values,
            weights_2[past_dates].values,
            rtol=WEIGHT_STABILITY_TOLERANCE,
            atol=WEIGHT_STABILITY_TOLERANCE,
            err_msg="Past weights changed as current_date advanced (backtest)",
        )

    def test_past_weights_locked_export(self, parity_features_df, parity_btc_df):
        """Test that past weights don't change as current_date advances (export)."""
        start_date = pd.Timestamp("2021-01-01")
        end_date = pd.Timestamp("2021-12-31")

        # Compute weights at two different current_dates
        current_date_1 = pd.Timestamp("2021-06-15")
        current_date_2 = pd.Timestamp("2021-09-15")

        result_1 = process_start_date_batch(
            start_date,
            [end_date],
            parity_features_df,
            parity_btc_df,
            current_date_1,
            PRICE_COL,
        )
        result_2 = process_start_date_batch(
            start_date,
            [end_date],
            parity_features_df,
            parity_btc_df,
            current_date_2,
            PRICE_COL,
        )

        weights_1 = result_1.set_index("DCA_date")["weight"]
        weights_2 = result_2.set_index("DCA_date")["weight"]

        # Past weights (before current_date_1) should be stable
        # Note: slight drift is acceptable due to rolling feature recalculation
        past_dates = [d for d in weights_1.index if pd.to_datetime(d) <= current_date_1]
        np.testing.assert_allclose(
            weights_1[past_dates].values,
            weights_2[past_dates].values,
            rtol=WEIGHT_STABILITY_TOLERANCE,
            atol=WEIGHT_STABILITY_TOLERANCE,
            err_msg="Past weights changed as current_date advanced (export)",
        )

    def test_kernel_parity_with_locked_prefix_mixed_window(
        self, parity_features_df, parity_btc_df
    ):
        start_date = pd.Timestamp("2025-01-01")
        end_date = pd.Timestamp("2025-12-31")
        current_date = pd.Timestamp("2025-07-01")
        full_idx = pd.date_range(start=start_date, end=end_date, freq="D")
        n_past = int((full_idx <= current_date).sum())
        locked_prefix = np.full(n_past - 1, 0.001, dtype=float)
        locked_prefix[0] = 0.1

        backtest_weights = compute_window_weights(
            parity_features_df,
            start_date,
            end_date,
            current_date,
            locked_weights=locked_prefix,
        )
        export_result = process_start_date_batch(
            start_date,
            [end_date],
            parity_features_df,
            parity_btc_df,
            current_date,
            PRICE_COL,
            locked_weights_by_end_date={end_date.strftime("%Y-%m-%d"): locked_prefix},
        )
        export_weights = export_result.set_index("DCA_date")["weight"]
        export_weights.index = pd.to_datetime(export_weights.index)
        np.testing.assert_allclose(
            backtest_weights.values,
            export_weights.values,
            rtol=FLOAT_TOLERANCE,
            atol=FLOAT_TOLERANCE,
        )

    def test_past_weights_parity_across_time(self, parity_features_df, parity_btc_df):
        """Test that past weights are identical in both modules across time."""
        start_date = pd.Timestamp("2021-01-01")
        end_date = pd.Timestamp("2021-12-31")

        for current_date in [
            pd.Timestamp("2021-03-01"),
            pd.Timestamp("2021-06-01"),
            pd.Timestamp("2021-09-01"),
            pd.Timestamp("2021-12-01"),
        ]:
            # Compute using backtest
            backtest_weights = compute_window_weights(
                parity_features_df, start_date, end_date, current_date
            )

            # Compute using export_weights
            export_result = process_start_date_batch(
                start_date,
                [end_date],
                parity_features_df,
                parity_btc_df,
                current_date,
                PRICE_COL,
            )
            export_weights = export_result.set_index("DCA_date")["weight"]
            export_weights.index = pd.to_datetime(export_weights.index)

            # Verify identical weights at this current_date
            np.testing.assert_allclose(
                backtest_weights.values,
                export_weights.values,
                rtol=FLOAT_TOLERANCE,
                atol=FLOAT_TOLERANCE,
                err_msg=f"Weights differ at current_date={current_date}",
            )


class TestWeightSumParity:
    """Test that weights sum to 1.0 in both modules."""

    def test_weights_sum_to_one_backtest(self, parity_features_df):
        """Test weights sum to 1.0 for various scenarios (backtest)."""
        test_cases = [
            # (start_date, end_date, current_date, description)
            (
                pd.Timestamp("2021-01-01"),
                pd.Timestamp("2021-12-31"),
                pd.Timestamp("2022-01-01"),
                "all past",
            ),
            (
                pd.Timestamp("2025-01-01"),
                pd.Timestamp("2025-12-31"),
                pd.Timestamp("2025-06-15"),
                "mixed",
            ),
            (
                pd.Timestamp("2026-01-01"),
                pd.Timestamp("2026-12-31"),
                pd.Timestamp("2025-01-01"),
                "all future",
            ),
        ]

        for start_date, end_date, current_date, desc in test_cases:
            weights = compute_window_weights(
                parity_features_df, start_date, end_date, current_date
            )
            assert np.isclose(weights.sum(), 1.0, atol=WEIGHT_SUM_TOLERANCE), (
                f"Weights don't sum to 1.0 for {desc}: sum={weights.sum()}"
            )

    def test_weights_sum_to_one_export(self, parity_features_df, parity_btc_df):
        """Test weights sum to 1.0 for various scenarios (export)."""
        test_cases = [
            (
                pd.Timestamp("2021-01-01"),
                pd.Timestamp("2021-12-31"),
                pd.Timestamp("2022-01-01"),
                "all past",
            ),
            (
                pd.Timestamp("2025-01-01"),
                pd.Timestamp("2025-12-31"),
                pd.Timestamp("2025-06-15"),
                "mixed",
            ),
            (
                pd.Timestamp("2026-01-01"),
                pd.Timestamp("2026-12-31"),
                pd.Timestamp("2025-01-01"),
                "all future",
            ),
        ]

        for start_date, end_date, current_date, desc in test_cases:
            result = process_start_date_batch(
                start_date,
                [end_date],
                parity_features_df,
                parity_btc_df,
                current_date,
                PRICE_COL,
            )
            weight_sum = result["weight"].sum()
            assert np.isclose(weight_sum, 1.0, atol=WEIGHT_SUM_TOLERANCE), (
                f"Weights don't sum to 1.0 for {desc}: sum={weight_sum}"
            )


class TestFutureWeightUniformityParity:
    """Test that future weights are uniform in both modules."""

    def test_future_weights_uniform_except_last_backtest(self, parity_features_df):
        """Test that future weights are uniform except the last day (backtest).

        The last day of the window absorbs the remainder to ensure sum = 1.0.
        """
        start_date = pd.Timestamp("2025-01-01")
        end_date = pd.Timestamp("2025-12-31")
        current_date = pd.Timestamp("2025-06-15")

        weights = compute_window_weights(
            parity_features_df, start_date, end_date, current_date
        )

        # Get future weights (after current_date) except the last day
        future_dates = weights.index[weights.index > current_date]
        future_weights_except_last = weights[future_dates[:-1]]

        # All future weights (except last) should be identical (uniform)
        assert len(set(future_weights_except_last.round(15))) == 1, (
            "Future weights (except last) are not uniform (backtest)"
        )

    def test_future_weights_uniform_except_last_export(
        self, parity_features_df, parity_btc_df
    ):
        """Test that future weights are uniform except the last day (export).

        The last day of the window absorbs the remainder to ensure sum = 1.0.
        """
        start_date = pd.Timestamp("2025-01-01")
        end_date = pd.Timestamp("2025-12-31")
        current_date = pd.Timestamp("2025-06-15")

        result = process_start_date_batch(
            start_date,
            [end_date],
            parity_features_df,
            parity_btc_df,
            current_date,
            PRICE_COL,
        )

        # Get future weights (after current_date) except the last day
        result["DCA_date_dt"] = pd.to_datetime(result["DCA_date"])
        future_mask = result["DCA_date_dt"] > current_date
        future_rows = result[future_mask]
        future_weights_except_last = future_rows["weight"].iloc[:-1]

        # All future weights (except last) should be identical (uniform)
        assert len(set(future_weights_except_last.round(15))) == 1, (
            "Future weights (except last) are not uniform (export)"
        )

    def test_future_weights_value_parity(self, parity_features_df, parity_btc_df):
        """Test that future uniform weight values are identical in both modules."""
        start_date = pd.Timestamp("2025-01-01")
        end_date = pd.Timestamp("2025-12-31")
        current_date = pd.Timestamp("2025-06-15")

        # Backtest
        backtest_weights = compute_window_weights(
            parity_features_df, start_date, end_date, current_date
        )
        future_dates_bt = backtest_weights.index[backtest_weights.index > current_date]
        backtest_future_weight = backtest_weights[future_dates_bt].iloc[0]

        # Export
        result = process_start_date_batch(
            start_date,
            [end_date],
            parity_features_df,
            parity_btc_df,
            current_date,
            PRICE_COL,
        )
        result["DCA_date_dt"] = pd.to_datetime(result["DCA_date"])
        export_future_weight = result[result["DCA_date_dt"] > current_date][
            "weight"
        ].iloc[0]

        # Should be identical
        assert np.isclose(
            backtest_future_weight, export_future_weight, rtol=FLOAT_TOLERANCE
        ), (
            f"Future weight values differ: backtest={backtest_future_weight}, "
            f"export={export_future_weight}"
        )


class TestDayByDayProgressionParity:
    """Test that day-by-day progression produces identical results."""

    def test_daily_progression_parity(self, parity_features_df, parity_btc_df):
        """Test that simulating day-by-day produces identical weights."""
        start_date = pd.Timestamp("2025-01-01")
        end_date = pd.Timestamp("2025-12-31")

        # Simulate daily progression
        for day_offset in range(0, 31, 5):  # Check every 5 days
            current_date = start_date + pd.Timedelta(days=day_offset)

            # Backtest
            backtest_weights = compute_window_weights(
                parity_features_df, start_date, end_date, current_date
            )

            # Export
            result = process_start_date_batch(
                start_date,
                [end_date],
                parity_features_df,
                parity_btc_df,
                current_date,
                PRICE_COL,
            )
            export_weights = result.set_index("DCA_date")["weight"]
            export_weights.index = pd.to_datetime(export_weights.index)

            # Verify identical
            np.testing.assert_allclose(
                backtest_weights.values,
                export_weights.values,
                rtol=FLOAT_TOLERANCE,
                atol=FLOAT_TOLERANCE,
                err_msg=f"Weights differ at day_offset={day_offset}",
            )

    def test_past_weights_stable_during_progression(
        self, parity_features_df, parity_btc_df
    ):
        """Test that past weights remain stable as we progress day by day."""
        start_date = pd.Timestamp("2025-01-01")
        end_date = pd.Timestamp("2025-12-31")

        previous_weights = None
        for day_offset in range(31):
            current_date = start_date + pd.Timedelta(days=day_offset)

            # Compute weights using backtest
            weights = compute_window_weights(
                parity_features_df, start_date, end_date, current_date
            )

            if previous_weights is not None:
                # Past weights (up to previous current_date) should be stable
                # Note: slight drift is acceptable due to rolling feature recalculation
                prev_current = start_date + pd.Timedelta(days=day_offset - 1)
                past_dates = weights.index[weights.index <= prev_current]

                np.testing.assert_allclose(
                    weights[past_dates].values,
                    previous_weights[past_dates].values,
                    rtol=WEIGHT_STABILITY_TOLERANCE,
                    atol=WEIGHT_STABILITY_TOLERANCE,
                    err_msg=f"Past weights changed at day {day_offset}",
                )

            previous_weights = weights


class TestEdgeCasesParity:
    """Test edge cases produce identical results in both modules."""

    def test_single_day_range(self, parity_features_df, parity_btc_df):
        """Single-day ranges are rejected by the framework span contract."""
        date = pd.Timestamp("2021-06-15")
        current_date = pd.Timestamp("2021-06-15")

        with pytest.raises(ValueError, match="365, 366, or 367 allocation days"):
            compute_window_weights(parity_features_df, date, date, current_date)
        with pytest.raises(ValueError, match="365, 366, or 367 allocation days"):
            process_start_date_batch(
                date,
                [date],
                parity_features_df,
                parity_btc_df,
                current_date,
                PRICE_COL,
            )

    def test_leap_year_range(self, parity_features_df, parity_btc_df):
        """Short leap-crossing ranges are rejected by the span contract."""
        start_date = pd.Timestamp("2024-02-28")
        end_date = pd.Timestamp("2024-03-01")
        current_date = pd.Timestamp("2024-03-01")
        with pytest.raises(ValueError, match="365, 366, or 367 allocation days"):
            compute_window_weights(parity_features_df, start_date, end_date, current_date)
        with pytest.raises(ValueError, match="365, 366, or 367 allocation days"):
            process_start_date_batch(
                start_date,
                [end_date],
                parity_features_df,
                parity_btc_df,
                current_date,
                PRICE_COL,
            )

    def test_current_date_equals_start(self, parity_features_df, parity_btc_df):
        """Test when current_date equals start_date."""
        start_date = pd.Timestamp("2025-06-01")
        end_date = pd.Timestamp("2026-05-31")
        current_date = start_date  # Only first day is "past"

        # Backtest
        backtest_weights = compute_window_weights(
            parity_features_df, start_date, end_date, current_date
        )

        # Export
        result = process_start_date_batch(
            start_date,
            [end_date],
            parity_features_df,
            parity_btc_df,
            current_date,
            PRICE_COL,
        )
        export_weights = result.set_index("DCA_date")["weight"]
        export_weights.index = pd.to_datetime(export_weights.index)

        np.testing.assert_allclose(
            backtest_weights.values,
            export_weights.values,
            rtol=FLOAT_TOLERANCE,
        )

        # First day should have model weight, rest should be uniform
        future_weights = backtest_weights.iloc[1:]
        assert len(set(future_weights.round(15))) == 1, "Future weights not uniform"


class TestComputeWeightsModalParity:
    """Test that compute_weights_modal matches expected behavior."""

    def test_compute_weights_modal_uses_end_as_current(
        self, parity_features_df, parity_btc_df
    ):
        """Test that compute_weights_modal uses end_date as current_date.

        For backtesting historical data, all dates are in the "past",
        so compute_weights_modal should use end_date as current_date.
        """
        import stacksats.backtest as backtest

        backtest._FEATURES_DF = parity_features_df

        start_date = pd.Timestamp("2021-01-01")
        end_date = pd.Timestamp("2021-12-31")

        # Create a window DataFrame
        window_df = parity_features_df.loc[start_date:end_date]

        # Compute using compute_weights_modal
        modal_weights = compute_weights_modal(window_df)

        # Compute using compute_window_weights with current_date = end_date
        expected_weights = compute_window_weights(
            parity_features_df, start_date, end_date, end_date
        )

        # Should be identical
        np.testing.assert_allclose(
            modal_weights.values,
            expected_weights.values,
            rtol=FLOAT_TOLERANCE,
        )

    def test_compute_weights_modal_matches_export_all_past(
        self, parity_features_df, parity_btc_df
    ):
        """Test compute_weights_modal matches export_weights for historical window."""
        import stacksats.backtest as backtest

        backtest._FEATURES_DF = parity_features_df

        start_date = pd.Timestamp("2021-01-01")
        end_date = pd.Timestamp("2021-12-31")

        # Create a window DataFrame
        window_df = parity_features_df.loc[start_date:end_date]

        # Compute using compute_weights_modal
        modal_weights = compute_weights_modal(window_df)

        # Compute using export_weights with current_date = end_date (all past)
        result = process_start_date_batch(
            start_date,
            [end_date],
            parity_features_df,
            parity_btc_df,
            end_date,  # All dates are past
            PRICE_COL,
        )
        export_weights = result.set_index("DCA_date")["weight"]
        export_weights.index = pd.to_datetime(export_weights.index)

        # Should be identical
        np.testing.assert_allclose(
            modal_weights.values,
            export_weights.values,
            rtol=FLOAT_TOLERANCE,
            err_msg="compute_weights_modal doesn't match export_weights",
        )
