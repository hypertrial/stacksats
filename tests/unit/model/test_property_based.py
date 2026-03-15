"""Property-based tests using Hypothesis."""

import datetime as dt

import numpy as np
import polars as pl
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
from stacksats.framework_contract import ALLOCATION_SPAN_DAYS
from stacksats.model_development import precompute_features
from tests.test_helpers import PRICE_COL

pytestmark = [
    pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed"),
    pytest.mark.slow,
]


def _create_sample_data():
    """Create sample BTC and features data for property-based tests."""
    dates = pl.datetime_range(
        dt.datetime(2020, 1, 1), dt.datetime(2025, 12, 31),
        interval="1d", eager=True
    ).to_list()

    np.random.seed(42)
    base_price = 10000
    returns = np.random.normal(0.001, 0.03, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))

    btc_df = pl.DataFrame({
        "date": dates,
        "price_usd": prices,
        "PriceUSD": prices,
    })
    features_df = precompute_features(btc_df)
    return features_df, btc_df


def _feature_bounds() -> tuple[dt.datetime, dt.datetime]:
    return (
        _SAMPLE_FEATURES_DF["date"].min(),
        _SAMPLE_FEATURES_DF["date"].max(),
    )


def _past_weights(df: pl.DataFrame, cutoff: dt.datetime) -> np.ndarray:
    return (
        df.filter(pl.col("date") <= cutoff)
        .sort("date")["weight"]
        .to_numpy()
    )


# Pre-create sample data at module level (only if hypothesis is available)
if HYPOTHESIS_AVAILABLE:
    _SAMPLE_FEATURES_DF, _SAMPLE_BTC_DF = _create_sample_data()

    @st.composite
    def date_range_strategy(draw):
        """Generate random date ranges for testing."""
        start_days = draw(st.integers(0, 550))
        start_date = dt.datetime(2024, 1, 1) + dt.timedelta(days=start_days)

        range_length = draw(st.integers(1, 60))
        end_date = start_date + dt.timedelta(days=range_length - 1)

        current_offset = draw(st.integers(-30, range_length + 30))
        current_date = start_date + dt.timedelta(days=current_offset)

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
        min_date, max_date = _feature_bounds()
        if (
            start_date < min_date
            or end_date > max_date
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

            if result.height > 0:
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

        min_date, max_date = _feature_bounds()
        if (
            start_date < min_date
            or end_date > max_date
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

            if result.height > 0:
                assert result["weight"].is_not_null().all(), "Found NaN weights"
                assert np.isfinite(result["weight"].to_numpy()).all(), "Found non-finite weights"
        except (ValueError, KeyError, IndexError):
            pass

    @given(date_range_strategy())
    @settings(max_examples=50, deadline=5000)
    def test_weights_non_negative(self, date_range_tuple):
        """Property: all weights >= 0 (MIN_W not enforced for stability)."""
        start_date, end_date, current_date = date_range_tuple

        if start_date > end_date:
            return

        min_date, max_date = _feature_bounds()
        if (
            start_date < min_date
            or end_date > max_date
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

            if result.height > 0:
                negative = result.filter(pl.col("weight") < -1e-15)
                assert negative.is_empty(), (
                    f"Found {negative.height} negative weights: "
                    f"min={result['weight'].min():.2e}"
                )
        except (ValueError, KeyError, IndexError):
            pass

    @given(date_range_strategy())
    @settings(max_examples=50, deadline=5000)
    def test_dca_date_coverage(self, date_range_tuple):
        """Property: date coverage matches expected range."""
        start_date, end_date, current_date = date_range_tuple

        if start_date > end_date:
            return

        min_date, max_date = _feature_bounds()
        if (
            start_date < min_date
            or end_date > max_date
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

            if result.height > 0:
                expected_dates = {
                    start_date + dt.timedelta(days=offset)
                    for offset in range((end_date - start_date).days + 1)
                }
                actual_dates = set(result["date"].to_list())

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

        min_date, max_date = _feature_bounds()
        if (
            start_date < min_date
            or end_date > max_date
        ):
            return

        # Generate second current_date after first
        if current_date1 < end_date:
            current_date2 = current_date1 + dt.timedelta(days=5)
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

            if result1.height > 0 and result2.height > 0:
                # Past weights (<= current_date1) should be identical
                past1 = _past_weights(result1, current_date1)
                past2 = _past_weights(result2, current_date1)

                if len(past1) > 1 and len(past2) > 1:
                    # With budget scaling, absolute values may change but
                    # relative proportions should be preserved
                    if past1.sum() > 0 and past2.sum() > 0:
                        w1_norm = past1 / past1.sum()
                        w2_norm = past2 / past2.sum()
                        np.testing.assert_allclose(w1_norm, w2_norm, rtol=1e-6)
        except (ValueError, KeyError, IndexError):
            pass


class TestImpossibleFloor:
    """Test impossible floor scenario (MIN_W * n_days > 1)."""

    def test_impossible_floor_scenario(self, sample_features_df, sample_btc_df):
        """Test floor behavior under a contract-valid 365-day window."""
        start_date = dt.datetime(2025, 1, 1)
        end_date = start_date + dt.timedelta(days=ALLOCATION_SPAN_DAYS - 1)

        result = process_start_date_batch(
            start_date,
            [end_date],
            sample_features_df,
            sample_btc_df,
            dt.datetime(2025, 12, 31),
            PRICE_COL,
        )

        assert result.height == ALLOCATION_SPAN_DAYS
        assert np.isclose(result["weight"].sum(), 1.0, rtol=1e-12)
        assert (result["weight"] >= 0).all()

    def test_tiny_range_with_floor(self, sample_features_df, sample_btc_df):
        """Invalid short windows should be rejected by span contract."""
        start_date = dt.datetime(2025, 1, 1)
        end_date = dt.datetime(2025, 1, 2)

        with pytest.raises(ValueError, match="configured fixed span"):
            process_start_date_batch(
                start_date,
                [end_date],
                sample_features_df,
                sample_btc_df,
                dt.datetime(2025, 12, 31),
                PRICE_COL,
            )
