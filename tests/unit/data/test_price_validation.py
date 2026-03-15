"""Tests for BTC price validation."""

import polars as pl

from tests.test_helpers import SAMPLE_END


class TestPriceValidation:
    """Test BTC price constraints."""

    def test_prices_positive(self, sample_weights_df):
        """Verify all non-null prices are positive."""
        prices = sample_weights_df["price_usd"].drop_nulls()
        assert bool((prices > 0).all()), f"Found {(prices <= 0).sum()} non-positive prices"

    def test_prices_reasonable_range(self, sample_weights_df):
        """Verify prices are within reasonable bounds ($100 - $10M)."""
        prices = sample_weights_df["price_usd"].drop_nulls()
        if len(prices) > 0:
            assert prices.min() >= 100, f"Min price {prices.min()} unreasonably low"
            assert prices.max() <= 10_000_000, (
                f"Max price {prices.max()} unreasonably high"
            )

    def test_price_consistency_across_ranges(self, sample_weights_df):
        """Verify same date has same price_usd across all ranges."""
        for dca_date, group in sample_weights_df.partition_by("date", as_dict=True).items():
            prices = group["price_usd"].drop_nulls()
            if len(prices) > 1:
                unique = prices.unique()
                assert len(unique) == 1, (
                    f"date {dca_date}: inconsistent prices {unique}"
                )

    def test_future_dates_null_prices(self, sample_weights_df):
        """Verify dates beyond sample data have NULL prices."""
        beyond = sample_weights_df.filter(pl.col("date") > SAMPLE_END)
        if not beyond.is_empty():
            non_null_count = beyond["price_usd"].is_not_null().sum()
            assert beyond["price_usd"].is_null().all(), (
                f"Found {non_null_count} non-null prices beyond sample data"
            )
