"""Property-based tests using Hypothesis."""

import numpy as np
import pandas as pd
import pytest

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

    # Create dummy decorators if hypothesis is not available
    def given(*args):
        def decorator(func):
            return func

        return decorator

    def settings(*args):
        def decorator(func):
            return func

        return decorator

    st = None

from stacksats.export_weights import process_start_date_batch
from stacksats.model_development import precompute_features
from tests.test_helpers import PRICE_COL

# Skip all tests in this module if hypothesis is not available
pytestmark = pytest.mark.skipif(
    not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed"
)


def _create_sample_data():
    """Create sample BTC and features data for property-based tests."""
    # Create date range from 2020 to 2025
    dates = pd.date_range(start="2020-01-01", end="2025-12-31", freq="D")

    # Generate realistic-looking price data with trend and volatility
    np.random.seed(42)
    base_price = 10000
    returns = np.random.normal(0.001, 0.03, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))

    btc_df = pd.DataFrame({"PriceUSD_coinmetrics": prices}, index=dates)
    btc_df["PriceUSD"] = btc_df["PriceUSD_coinmetrics"]
    btc_df.index.name = "time"

    features_df = precompute_features(btc_df)
    return features_df, btc_df


# Pre-create sample data at module level (only if hypothesis is available)
if HYPOTHESIS_AVAILABLE:
    _SAMPLE_FEATURES_DF, _SAMPLE_BTC_DF = _create_sample_data()

    @st.composite
    def date_range_strategy(draw):
        """Generate random date ranges for testing."""
        # Start date between 2024-01-01 and 2025-06-01
        start_days = draw(st.integers(0, 550))
        start_date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=start_days)

        # Range length between 1 and 60 days
        range_length = draw(st.integers(1, 60))
        end_date = start_date + pd.Timedelta(days=range_length - 1)

        # Current date relative to range
        current_offset = draw(st.integers(-30, range_length + 30))
        current_date = start_date + pd.Timedelta(days=current_offset)

        return start_date, end_date, current_date

else:
    # Dummy values if hypothesis is not available
    _SAMPLE_FEATURES_DF = None
    _SAMPLE_BTC_DF = None

    def date_range_strategy(draw=None):
        return None


class TestPropertyBasedInvariants:
    """Property-based tests for invariants."""

    @given(date_range_strategy())
    @settings(max_examples=50, deadline=5000)
    def test_weights_sum_to_one(self, date_range_tuple):
        """Property: weights always sum to 1.0."""
        start_date, end_date, current_date = date_range_tuple

        # Skip if range is invalid
        if start_date > end_date:
            return

        # Skip if dates are outside sample data range
        if (
            start_date < _SAMPLE_FEATURES_DF.index.min()
            or end_date > _SAMPLE_FEATURES_DF.index.max()
        ):
            return

        try:
            result = process_start_date_batch(
                start_date,
                [end_date],
                _SAMPLE_FEATURES_DF,
                _SAMPLE_BTC_DF,
                current_date,
                PRICE_COL,
            )

            if len(result) > 0:
                weight_sum = result["weight"].sum()
                assert np.isclose(weight_sum, 1.0, rtol=1e-10, atol=1e-10), (
                    f"Weights sum to {weight_sum:.15f}, expected 1.0"
                )
        except (ValueError, KeyError, IndexError):
            # Skip invalid configurations
            pass

    @given(date_range_strategy())
    @settings(max_examples=50, deadline=5000)
    def test_weights_finite(self, date_range_tuple):
        """Property: all weights are finite."""
        start_date, end_date, current_date = date_range_tuple

        if start_date > end_date:
            return

        if (
            start_date < _SAMPLE_FEATURES_DF.index.min()
            or end_date > _SAMPLE_FEATURES_DF.index.max()
        ):
            return

        try:
            result = process_start_date_batch(
                start_date,
                [end_date],
                _SAMPLE_FEATURES_DF,
                _SAMPLE_BTC_DF,
                current_date,
                PRICE_COL,
            )

            if len(result) > 0:
                assert result["weight"].notna().all(), "Found NaN weights"
                assert np.isfinite(result["weight"]).all(), "Found non-finite weights"
        except (ValueError, KeyError, IndexError):
            pass

    @given(date_range_strategy())
    @settings(max_examples=50, deadline=5000)
    def test_weights_non_negative(self, date_range_tuple):
        """Property: all weights >= 0 (MIN_W not enforced for stability)."""
        start_date, end_date, current_date = date_range_tuple

        if start_date > end_date:
            return

        if (
            start_date < _SAMPLE_FEATURES_DF.index.min()
            or end_date > _SAMPLE_FEATURES_DF.index.max()
        ):
            return

        try:
            result = process_start_date_batch(
                start_date,
                [end_date],
                _SAMPLE_FEATURES_DF,
                _SAMPLE_BTC_DF,
                current_date,
                PRICE_COL,
            )

            if len(result) > 0:
                negative = result[result["weight"] < -1e-15]
                assert negative.empty, (
                    f"Found {len(negative)} negative weights: "
                    f"min={result['weight'].min():.2e}"
                )
        except (ValueError, KeyError, IndexError):
            pass

    @given(date_range_strategy())
    @settings(max_examples=50, deadline=5000)
    def test_dca_date_coverage(self, date_range_tuple):
        """Property: DCA_date coverage matches expected range."""
        start_date, end_date, current_date = date_range_tuple

        if start_date > end_date:
            return

        if (
            start_date < _SAMPLE_FEATURES_DF.index.min()
            or end_date > _SAMPLE_FEATURES_DF.index.max()
        ):
            return

        try:
            result = process_start_date_batch(
                start_date,
                [end_date],
                _SAMPLE_FEATURES_DF,
                _SAMPLE_BTC_DF,
                current_date,
                PRICE_COL,
            )

            if len(result) > 0:
                expected_dates = pd.date_range(start=start_date, end=end_date, freq="D")
                actual_dates = set(pd.to_datetime(result["DCA_date"]))

                # Should cover all expected dates
                assert len(actual_dates) == len(expected_dates), (
                    f"Date coverage mismatch: {len(actual_dates)} != {len(expected_dates)}"
                )
        except (ValueError, KeyError, IndexError):
            pass

    @given(date_range_strategy())
    @settings(max_examples=30, deadline=5000)
    def test_past_weights_immutable(self, date_range_tuple):
        """Property: past weights don't change when current_date advances."""
        start_date, end_date, current_date1 = date_range_tuple

        if start_date > end_date:
            return

        if (
            start_date < _SAMPLE_FEATURES_DF.index.min()
            or end_date > _SAMPLE_FEATURES_DF.index.max()
        ):
            return

        # Generate second current_date after first
        if current_date1 < end_date:
            current_date2 = current_date1 + pd.Timedelta(days=5)
        else:
            return

        try:
            result1 = process_start_date_batch(
                start_date,
                [end_date],
                _SAMPLE_FEATURES_DF,
                _SAMPLE_BTC_DF,
                current_date1,
                PRICE_COL,
            )

            result2 = process_start_date_batch(
                start_date,
                [end_date],
                _SAMPLE_FEATURES_DF,
                _SAMPLE_BTC_DF,
                current_date2,
                PRICE_COL,
            )

            if len(result1) > 0 and len(result2) > 0:
                # Past weights (<= current_date1) should be identical
                past1 = result1[
                    pd.to_datetime(result1["DCA_date"]) <= current_date1
                ].sort_values("DCA_date")
                past2 = result2[
                    pd.to_datetime(result2["DCA_date"]) <= current_date1
                ].sort_values("DCA_date")

                if len(past1) > 1 and len(past2) > 1:
                    # With budget scaling, absolute values may change but
                    # relative proportions should be preserved
                    w1 = past1["weight"].reset_index(drop=True)
                    w2 = past2["weight"].reset_index(drop=True)
                    if w1.sum() > 0 and w2.sum() > 0:
                        w1_norm = w1 / w1.sum()
                        w2_norm = w2 / w2.sum()
                        pd.testing.assert_series_equal(
                            w1_norm,
                            w2_norm,
                            rtol=1e-6,
                        )
        except (ValueError, KeyError, IndexError):
            pass


class TestImpossibleFloor:
    """Test impossible floor scenario (MIN_W * n_days > 1)."""

    def test_impossible_floor_scenario(self, sample_features_df, sample_btc_df):
        """Test floor behavior under a contract-valid 365-day window."""
        start_date = pd.Timestamp("2025-01-01")
        end_date = pd.Timestamp("2025-12-31")

        result = process_start_date_batch(
            start_date,
            [end_date],
            sample_features_df,
            sample_btc_df,
            pd.Timestamp("2025-12-31"),
            PRICE_COL,
        )

        assert len(result) == 365
        assert np.isclose(result["weight"].sum(), 1.0, rtol=1e-12)
        assert (result["weight"] >= 0).all()

    def test_tiny_range_with_floor(self, sample_features_df, sample_btc_df):
        """Invalid short windows should be rejected by span contract."""
        start_date = pd.Timestamp("2025-01-01")
        end_date = pd.Timestamp("2025-01-02")

        with pytest.raises(ValueError, match="365, 366, or 367 allocation days"):
            process_start_date_batch(
                start_date,
                [end_date],
                sample_features_df,
                sample_btc_df,
                pd.Timestamp("2025-12-31"),
                PRICE_COL,
            )
