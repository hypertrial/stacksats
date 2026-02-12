"""Comprehensive tests for historical BTC price fetching.

Tests cover:
- Historical API fetchers (CoinGecko, Binance)
- Robust historical wrapper with fallbacks
- Date-specific fetching logic
"""

import pytest
import pandas as pd
import responses

from stacksats.btc_price_fetcher import (
    fetch_historical_price_coingecko,
    fetch_historical_price_binance,
    fetch_btc_price_historical,
)

# Use fixed dates for testing
TEST_DATE = pd.Timestamp("2024-01-01")


class TestHistoricalFetchers:
    """Tests for individual historical API fetchers."""

    @responses.activate
    def test_coingecko_historical_success(self):
        """Test successful historical price fetch from CoinGecko."""
        # CoinGecko uses DD-MM-YYYY
        responses.add(
            responses.GET,
            "https://api.coingecko.com/api/v3/coins/bitcoin/history",
            json={
                "market_data": {
                    "current_price": {"usd": 42500.0}
                }
            },
            status=200,
        )
        price = fetch_historical_price_coingecko(TEST_DATE)
        assert price == 42500.0
        assert responses.calls[0].request.params["date"] == "01-01-2024"

    @responses.activate
    def test_coingecko_historical_invalid_data(self):
        """Test CoinGecko historical with malformed response."""
        responses.add(
            responses.GET,
            "https://api.coingecko.com/api/v3/coins/bitcoin/history",
            json={"id": "bitcoin"},  # Missing market_data
            status=200,
        )
        with pytest.raises(ValueError, match="Invalid CoinGecko historical response"):
            fetch_historical_price_coingecko(TEST_DATE)

    @responses.activate
    def test_binance_historical_success(self):
        """Test successful historical price fetch from Binance."""
        # Binance klines returns a list of lists
        responses.add(
            responses.GET,
            "https://api.binance.com/api/v3/klines",
            json=[
                [
                    1704067200000, "42000.0", "43000.0", "41500.0", "42600.0",
                    "100.0", 1704153599999, "4260000.0", 50000, "50.0", "2130000.0", "0"
                ]
            ],
            status=200,
        )
        price = fetch_historical_price_binance(TEST_DATE)
        assert price == 42600.0  # Index 4 is Close
        
        # Verify timestamp
        expected_ts = int(TEST_DATE.timestamp() * 1000)
        assert responses.calls[0].request.params["startTime"] == str(expected_ts)

    @responses.activate
    def test_binance_historical_empty(self):
        """Test Binance historical with no data."""
        responses.add(
            responses.GET,
            "https://api.binance.com/api/v3/klines",
            json=[],
            status=200,
        )
        with pytest.raises(ValueError, match="No Binance historical data"):
            fetch_historical_price_binance(TEST_DATE)


class TestRobustHistoricalFetcher:
    """Tests for the robust historical fetcher wrapper."""

    @responses.activate
    def test_fallback_logic(self):
        """Test that Binance is used if CoinGecko fails."""
        # CoinGecko fails
        responses.add(
            responses.GET,
            "https://api.coingecko.com/api/v3/coins/bitcoin/history",
            status=500,
        )
        # Binance succeeds
        responses.add(
            responses.GET,
            "https://api.binance.com/api/v3/klines",
            json=[[0, 0, 0, 0, "43000.0", 0, 0, 0, 0, 0, 0, 0]],
            status=200,
        )
        
        price = fetch_btc_price_historical(TEST_DATE)
        assert price == 43000.0

    @responses.activate
    def test_all_historical_fail_returns_none(self):
        """Test that None is returned if all historical sources fail."""
        responses.add(
            responses.GET,
            "https://api.coingecko.com/api/v3/coins/bitcoin/history",
            status=500,
        )
        responses.add(
            responses.GET,
            "https://api.binance.com/api/v3/klines",
            status=404,
        )
        
        price = fetch_btc_price_historical(TEST_DATE)
        assert price is None
