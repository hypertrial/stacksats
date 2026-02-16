"""Comprehensive tests for historical BTC price fetching.

Tests cover:
- Historical data fetchers (CoinMetrics, CoinGecko, Binance)
- Robust historical wrapper with fallbacks
- Date-specific fetching logic
"""

import pytest
import pandas as pd
import responses

from stacksats.btc_price_fetcher import (
    fetch_historical_price_coinmetrics,
    fetch_historical_price_coingecko,
    fetch_historical_price_binance,
    fetch_btc_price_historical,
)

# Use fixed dates for testing
TEST_DATE = pd.Timestamp("2024-01-01")


@pytest.fixture(autouse=True)
def clear_coinmetrics_cache():
    """Clear CoinMetrics cache before each test."""
    from stacksats.btc_price_fetcher import _load_coinmetrics_data
    _load_coinmetrics_data.cache_clear()
    yield
    _load_coinmetrics_data.cache_clear()


class TestHistoricalFetchers:
    """Tests for individual historical API fetchers."""

    @responses.activate
    def test_coinmetrics_historical_success(self):
        """Test successful historical price fetch from CoinMetrics."""
        # Mock CoinMetrics CSV response (minimal valid CSV with required columns)
        csv_content = "time,PriceUSD,CapMVRVCur\n2024-01-01,42500.0,2.5\n"
        responses.add(
            responses.GET,
            "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv",
            body=csv_content,
            status=200,
            content_type="text/csv",
        )
        price = fetch_historical_price_coinmetrics(TEST_DATE)
        assert price == 42500.0

    @responses.activate
    def test_coinmetrics_historical_date_not_found(self):
        """Test CoinMetrics historical with date not in data."""
        csv_content = "time,PriceUSD,CapMVRVCur\n2024-01-02,43000.0,2.6\n"
        responses.add(
            responses.GET,
            "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv",
            body=csv_content,
            status=200,
            content_type="text/csv",
        )
        with pytest.raises(ValueError, match="not found in CoinMetrics data"):
            fetch_historical_price_coinmetrics(TEST_DATE)

    @responses.activate
    def test_coinmetrics_historical_missing_price(self):
        """Test CoinMetrics historical with missing price."""
        csv_content = "time,PriceUSD,CapMVRVCur\n2024-01-01,,\n"
        responses.add(
            responses.GET,
            "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv",
            body=csv_content,
            status=200,
            content_type="text/csv",
        )
        with pytest.raises(ValueError, match="PriceUSD is missing"):
            fetch_historical_price_coinmetrics(TEST_DATE)

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
    def test_coinmetrics_primary_source(self):
        """Test that CoinMetrics is used first."""
        # CoinMetrics succeeds
        csv_content = "time,PriceUSD,CapMVRVCur\n2024-01-01,42500.0,2.5\n"
        responses.add(
            responses.GET,
            "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv",
            body=csv_content,
            status=200,
            content_type="text/csv",
        )

        price = fetch_btc_price_historical(TEST_DATE)
        assert price == 42500.0
        # Verify CoinGecko/Binance were not called
        assert len([c for c in responses.calls if "coingecko" in c.request.url.lower()]) == 0
        assert len([c for c in responses.calls if "binance" in c.request.url.lower()]) == 0

    @responses.activate
    def test_fallback_logic(self):
        """Test that CoinGecko is used if CoinMetrics fails, then Binance."""
        # CoinMetrics fails (date not found)
        csv_content = "time,PriceUSD,CapMVRVCur\n2024-01-02,43000.0,2.6\n"
        responses.add(
            responses.GET,
            "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv",
            body=csv_content,
            status=200,
            content_type="text/csv",
        )
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
        # CoinMetrics fails (date not found)
        csv_content = "time,PriceUSD,CapMVRVCur\n2024-01-02,43000.0,2.6\n"
        responses.add(
            responses.GET,
            "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv",
            body=csv_content,
            status=200,
            content_type="text/csv",
        )
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


def test_historical_fetch_warns_on_invalid_price_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stacksats import btc_price_fetcher as fetcher

    monkeypatch.setattr(fetcher, "fetch_historical_price_coinmetrics", lambda date: 10.0)
    monkeypatch.setattr(fetcher, "fetch_historical_price_coingecko", lambda date: 42000.0)
    monkeypatch.setattr(
        fetcher,
        "fetch_historical_price_binance",
        lambda date: (_ for _ in ()).throw(ValueError("should not be called")),
    )
    monkeypatch.setattr(fetcher, "validate_price", lambda price, previous_price=None: price > 1000.0)

    price = fetcher.fetch_btc_price_historical(TEST_DATE)

    assert price == 42000.0
