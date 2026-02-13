"""Tests for BTC price validation."""

import pandas as pd

from tests.test_helpers import SAMPLE_END


class TestPriceValidation:
    """Test BTC price constraints."""

    def test_prices_positive(self, sample_weights_df):
        """Verify all non-null prices are positive."""
        prices = sample_weights_df["btc_usd"].dropna()
        assert (prices > 0).all(), f"Found {(prices <= 0).sum()} non-positive prices"

    def test_prices_reasonable_range(self, sample_weights_df):
        """Verify prices are within reasonable bounds ($100 - $10M)."""
        prices = sample_weights_df["btc_usd"].dropna()
        if len(prices) > 0:
            assert prices.min() >= 100, f"Min price {prices.min()} unreasonably low"
            assert prices.max() <= 10_000_000, (
                f"Max price {prices.max()} unreasonably high"
            )

    def test_price_consistency_across_ranges(self, sample_weights_df):
        """Verify same DCA_date has same btc_usd across all ranges."""
        for dca_date, group in sample_weights_df.groupby("DCA_date"):
            prices = group["btc_usd"].dropna()
            if len(prices) > 1:
                unique = prices.unique()
                assert len(unique) == 1, (
                    f"DCA_date {dca_date}: inconsistent prices {unique}"
                )

    def test_future_dates_null_prices(self, sample_weights_df):
        """Verify dates beyond sample data have NULL prices."""
        max_date = pd.Timestamp(SAMPLE_END)
        beyond = sample_weights_df[
            pd.to_datetime(sample_weights_df["DCA_date"]) > max_date
        ]
        if not beyond.empty:
            assert beyond["btc_usd"].isna().all(), (
                f"Found {beyond['btc_usd'].notna().sum()} non-null prices beyond sample data"
            )
