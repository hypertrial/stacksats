"""Comprehensive tests for btc_price_fetcher module.

Tests cover:
- Individual API fetchers (CoinGecko, Coinbase, Bitstamp, Kraken, Binance)
- Price validation logic
- Retry behavior with mocked failures
- Fallback mechanism across all sources
- Edge cases and error handling
"""

import os
import pytest
import responses
import requests

from stacksats.btc_price_fetcher import (
    fetch_price_coingecko,
    fetch_price_coinbase,
    fetch_price_bitstamp,
    fetch_price_kraken,
    fetch_price_binance,
    validate_price,
    fetch_btc_price_robust,
    MIN_BTC_PRICE,
    MAX_BTC_PRICE,
)


# =============================================================================
# Test Individual API Fetchers with Mocked Responses
# =============================================================================


class TestCoinGeckoFetcher:
    """Tests for CoinGecko API fetcher."""

    @responses.activate
    def test_successful_fetch(self):
        """Test successful price fetch from CoinGecko."""
        responses.add(
            responses.GET,
            "https://api.coingecko.com/api/v3/simple/price",
            json={"bitcoin": {"usd": 97500.50}},
            status=200,
        )
        price = fetch_price_coingecko()
        assert price == 97500.50

    @responses.activate
    def test_missing_bitcoin_key_raises_error(self):
        """Test that missing 'bitcoin' key raises ValueError."""
        responses.add(
            responses.GET,
            "https://api.coingecko.com/api/v3/simple/price",
            json={"ethereum": {"usd": 3000}},
            status=200,
        )
        with pytest.raises(ValueError, match="Invalid CoinGecko response format"):
            fetch_price_coingecko()

    @responses.activate
    def test_missing_usd_key_raises_error(self):
        """Test that missing 'usd' key raises ValueError."""
        responses.add(
            responses.GET,
            "https://api.coingecko.com/api/v3/simple/price",
            json={"bitcoin": {"eur": 90000}},
            status=200,
        )
        with pytest.raises(ValueError, match="Invalid CoinGecko response format"):
            fetch_price_coingecko()

    @responses.activate
    def test_http_error_raises_exception(self):
        """Test that HTTP errors raise RequestException."""
        responses.add(
            responses.GET,
            "https://api.coingecko.com/api/v3/simple/price",
            json={"error": "rate limit exceeded"},
            status=429,
        )
        with pytest.raises(requests.exceptions.HTTPError):
            fetch_price_coingecko()


class TestCoinbaseFetcher:
    """Tests for Coinbase API fetcher."""

    @responses.activate
    def test_successful_fetch(self):
        """Test successful price fetch from Coinbase."""
        responses.add(
            responses.GET,
            "https://api.coinbase.com/v2/prices/BTC-USD/spot",
            json={"data": {"base": "BTC", "currency": "USD", "amount": "98000.25"}},
            status=200,
        )
        price = fetch_price_coinbase()
        assert price == 98000.25

    @responses.activate
    def test_missing_data_key_raises_error(self):
        """Test that missing 'data' key raises ValueError."""
        responses.add(
            responses.GET,
            "https://api.coinbase.com/v2/prices/BTC-USD/spot",
            json={"result": {"amount": "98000"}},
            status=200,
        )
        with pytest.raises(ValueError, match="Invalid Coinbase response format"):
            fetch_price_coinbase()

    @responses.activate
    def test_missing_amount_key_raises_error(self):
        """Test that missing 'amount' key raises ValueError."""
        responses.add(
            responses.GET,
            "https://api.coinbase.com/v2/prices/BTC-USD/spot",
            json={"data": {"base": "BTC", "currency": "USD"}},
            status=200,
        )
        with pytest.raises(ValueError, match="Invalid Coinbase response format"):
            fetch_price_coinbase()

    @responses.activate
    def test_http_error_raises_exception(self):
        """Test that HTTP errors raise RequestException."""
        responses.add(
            responses.GET,
            "https://api.coinbase.com/v2/prices/BTC-USD/spot",
            json={"error": "service unavailable"},
            status=503,
        )
        with pytest.raises(requests.exceptions.HTTPError):
            fetch_price_coinbase()


class TestBitstampFetcher:
    """Tests for Bitstamp API fetcher."""

    @responses.activate
    def test_successful_fetch(self):
        """Test successful price fetch from Bitstamp."""
        responses.add(
            responses.GET,
            "https://www.bitstamp.net/api/v2/ticker/btcusd/",
            json={"last": "97800.00", "high": "99000", "low": "96000"},
            status=200,
        )
        price = fetch_price_bitstamp()
        assert price == 97800.00

    @responses.activate
    def test_missing_last_key_raises_error(self):
        """Test that missing 'last' key raises ValueError."""
        responses.add(
            responses.GET,
            "https://www.bitstamp.net/api/v2/ticker/btcusd/",
            json={"high": "99000", "low": "96000"},
            status=200,
        )
        with pytest.raises(ValueError, match="Invalid Bitstamp response format"):
            fetch_price_bitstamp()

    @responses.activate
    def test_http_error_raises_exception(self):
        """Test that HTTP errors raise RequestException."""
        responses.add(
            responses.GET,
            "https://www.bitstamp.net/api/v2/ticker/btcusd/",
            status=500,
        )
        with pytest.raises(requests.exceptions.HTTPError):
            fetch_price_bitstamp()


class TestKrakenFetcher:
    """Tests for Kraken API fetcher."""

    @responses.activate
    def test_successful_fetch_xxbtzusd(self):
        """Test successful price fetch from Kraken with XXBTZUSD pair."""
        responses.add(
            responses.GET,
            "https://api.kraken.com/0/public/Ticker",
            json={
                "error": [],
                "result": {
                    "XXBTZUSD": {
                        "a": ["97500.00000", "1", "1.000"],
                        "b": ["97499.00000", "1", "1.000"],
                        "c": ["97500.50000", "0.00100000"],  # Last trade price
                        "v": ["1000.00000000", "5000.00000000"],
                    }
                },
            },
            status=200,
        )
        price = fetch_price_kraken()
        assert price == 97500.50

    @responses.activate
    def test_successful_fetch_xbtusd(self):
        """Test successful price fetch from Kraken with XBTUSD pair fallback."""
        responses.add(
            responses.GET,
            "https://api.kraken.com/0/public/Ticker",
            json={
                "error": [],
                "result": {
                    "XBTUSD": {
                        "c": ["98000.00000", "0.00100000"],
                    }
                },
            },
            status=200,
        )
        price = fetch_price_kraken()
        assert price == 98000.00

    @responses.activate
    def test_api_error_raises_exception(self):
        """Test that Kraken API error field raises ValueError."""
        responses.add(
            responses.GET,
            "https://api.kraken.com/0/public/Ticker",
            json={"error": ["EGeneral:Invalid arguments"], "result": {}},
            status=200,
        )
        with pytest.raises(ValueError, match="Kraken API error"):
            fetch_price_kraken()

    @responses.activate
    def test_missing_pair_data_raises_error(self):
        """Test that missing pair data raises ValueError."""
        responses.add(
            responses.GET,
            "https://api.kraken.com/0/public/Ticker",
            json={"error": [], "result": {}},
            status=200,
        )
        with pytest.raises(ValueError, match="Invalid Kraken response format"):
            fetch_price_kraken()

    @responses.activate
    def test_missing_c_key_raises_error(self):
        """Test that missing 'c' key (last trade) raises ValueError."""
        responses.add(
            responses.GET,
            "https://api.kraken.com/0/public/Ticker",
            json={
                "error": [],
                "result": {"XXBTZUSD": {"a": ["97500.00000", "1", "1.000"]}},
            },
            status=200,
        )
        with pytest.raises(ValueError, match="Invalid Kraken response format"):
            fetch_price_kraken()

    @responses.activate
    def test_http_error_raises_exception(self):
        """Test that HTTP errors raise RequestException."""
        responses.add(
            responses.GET,
            "https://api.kraken.com/0/public/Ticker",
            status=502,
        )
        with pytest.raises(requests.exceptions.HTTPError):
            fetch_price_kraken()


class TestBinanceFetcher:
    """Tests for Binance API fetcher."""

    @responses.activate
    def test_successful_fetch(self):
        """Test successful price fetch from Binance."""
        responses.add(
            responses.GET,
            "https://api.binance.com/api/v3/ticker/price",
            json={"symbol": "BTCUSDT", "price": "97750.50000000"},
            status=200,
        )
        price = fetch_price_binance()
        assert price == 97750.50

    @responses.activate
    def test_missing_price_key_raises_error(self):
        """Test that missing 'price' key raises ValueError."""
        responses.add(
            responses.GET,
            "https://api.binance.com/api/v3/ticker/price",
            json={"symbol": "BTCUSDT"},
            status=200,
        )
        with pytest.raises(ValueError, match="Invalid Binance response format"):
            fetch_price_binance()

    @responses.activate
    def test_http_error_raises_exception(self):
        """Test that HTTP errors raise RequestException."""
        responses.add(
            responses.GET,
            "https://api.binance.com/api/v3/ticker/price",
            json={"code": -1121, "msg": "Invalid symbol."},
            status=400,
        )
        with pytest.raises(requests.exceptions.HTTPError):
            fetch_price_binance()


# =============================================================================
# Test Price Validation
# =============================================================================


class TestPriceValidation:
    """Tests for price validation logic."""

    def test_valid_price_in_range(self):
        """Test that prices within valid range pass validation."""
        assert validate_price(50000.0) is True
        assert validate_price(100000.0) is True
        assert validate_price(MIN_BTC_PRICE) is True
        assert validate_price(MAX_BTC_PRICE) is True

    def test_price_below_minimum_fails(self):
        """Test that prices below minimum fail validation."""
        assert validate_price(MIN_BTC_PRICE - 1) is False
        assert validate_price(500.0) is False
        assert validate_price(0.0) is False

    def test_price_above_maximum_fails(self):
        """Test that prices above maximum fail validation."""
        assert validate_price(MAX_BTC_PRICE + 1) is False
        assert validate_price(2000000.0) is False

    def test_negative_price_fails(self):
        """Test that negative prices fail validation."""
        assert validate_price(-1.0) is False
        assert validate_price(-50000.0) is False

    def test_non_numeric_price_fails(self):
        """Test that non-numeric prices fail validation."""
        assert validate_price("50000") is False
        assert validate_price(None) is False

    def test_price_change_within_threshold_passes(self):
        """Test that reasonable price changes pass validation."""
        previous = 100000.0
        # 10% increase should pass
        assert validate_price(110000.0, previous_price=previous) is True
        # 10% decrease should pass
        assert validate_price(90000.0, previous_price=previous) is True

    def test_price_change_exceeds_threshold_still_passes_with_warning(self):
        """Test that large price changes still pass (with warning) since it could be legitimate."""
        previous = 100000.0
        # 25% change exceeds threshold but should still pass (just warns)
        assert validate_price(125000.0, previous_price=previous) is True
        assert validate_price(75000.0, previous_price=previous) is True

    def test_validation_with_zero_previous_price(self):
        """Test validation when previous price is zero."""
        # Should still validate current price without comparison
        assert validate_price(50000.0, previous_price=0.0) is True

    def test_validation_with_none_previous_price(self):
        """Test validation when previous price is None."""
        assert validate_price(50000.0, previous_price=None) is True


# =============================================================================
# Test Robust Fetcher with Fallbacks
# =============================================================================


class TestRobustFetcher:
    """Tests for robust fetcher with fallback mechanism."""

    @responses.activate
    def test_returns_first_valid_price(self):
        """Test that first successful source's price is returned."""
        responses.add(
            responses.GET,
            "https://api.coingecko.com/api/v3/simple/price",
            json={"bitcoin": {"usd": 97500}},
            status=200,
        )
        price = fetch_btc_price_robust()
        assert price == 97500

    @responses.activate
    def test_fallback_to_second_source(self):
        """Test that second source is tried when first fails."""
        # CoinGecko fails
        responses.add(
            responses.GET,
            "https://api.coingecko.com/api/v3/simple/price",
            status=500,
        )
        # Coinbase succeeds
        responses.add(
            responses.GET,
            "https://api.coinbase.com/v2/prices/BTC-USD/spot",
            json={"data": {"amount": "98000"}},
            status=200,
        )
        price = fetch_btc_price_robust()
        assert price == 98000

    @responses.activate
    def test_fallback_to_third_source(self):
        """Test that third source is tried when first two fail."""
        # CoinGecko fails
        responses.add(
            responses.GET,
            "https://api.coingecko.com/api/v3/simple/price",
            status=500,
        )
        # Coinbase fails
        responses.add(
            responses.GET,
            "https://api.coinbase.com/v2/prices/BTC-USD/spot",
            status=503,
        )
        # Bitstamp succeeds
        responses.add(
            responses.GET,
            "https://www.bitstamp.net/api/v2/ticker/btcusd/",
            json={"last": "97800"},
            status=200,
        )
        price = fetch_btc_price_robust()
        assert price == 97800

    @responses.activate
    def test_fallback_to_kraken(self):
        """Test fallback to Kraken (4th source)."""
        # First 3 sources fail
        responses.add(
            responses.GET,
            "https://api.coingecko.com/api/v3/simple/price",
            status=500,
        )
        responses.add(
            responses.GET,
            "https://api.coinbase.com/v2/prices/BTC-USD/spot",
            status=503,
        )
        responses.add(
            responses.GET,
            "https://www.bitstamp.net/api/v2/ticker/btcusd/",
            status=502,
        )
        # Kraken succeeds
        responses.add(
            responses.GET,
            "https://api.kraken.com/0/public/Ticker",
            json={"error": [], "result": {"XXBTZUSD": {"c": ["97600", "0.001"]}}},
            status=200,
        )
        price = fetch_btc_price_robust()
        assert price == 97600

    @responses.activate
    def test_fallback_to_binance(self):
        """Test fallback to Binance (5th source)."""
        # First 4 sources fail
        responses.add(
            responses.GET,
            "https://api.coingecko.com/api/v3/simple/price",
            status=500,
        )
        responses.add(
            responses.GET,
            "https://api.coinbase.com/v2/prices/BTC-USD/spot",
            status=503,
        )
        responses.add(
            responses.GET,
            "https://www.bitstamp.net/api/v2/ticker/btcusd/",
            status=502,
        )
        responses.add(
            responses.GET,
            "https://api.kraken.com/0/public/Ticker",
            status=500,
        )
        # Binance succeeds
        responses.add(
            responses.GET,
            "https://api.binance.com/api/v3/ticker/price",
            json={"symbol": "BTCUSDT", "price": "97700"},
            status=200,
        )
        price = fetch_btc_price_robust()
        assert price == 97700

    @responses.activate
    def test_all_sources_fail_returns_none(self):
        """Test that None is returned when all sources fail."""
        # All sources fail
        responses.add(
            responses.GET,
            "https://api.coingecko.com/api/v3/simple/price",
            status=500,
        )
        responses.add(
            responses.GET,
            "https://api.coinbase.com/v2/prices/BTC-USD/spot",
            status=503,
        )
        responses.add(
            responses.GET,
            "https://www.bitstamp.net/api/v2/ticker/btcusd/",
            status=502,
        )
        responses.add(
            responses.GET,
            "https://api.kraken.com/0/public/Ticker",
            status=500,
        )
        responses.add(
            responses.GET,
            "https://api.binance.com/api/v3/ticker/price",
            status=400,
        )
        price = fetch_btc_price_robust()
        assert price is None

    @responses.activate
    def test_invalid_price_triggers_fallback(self):
        """Test that invalid price from first source triggers fallback."""
        # CoinGecko returns invalid (too low) price
        responses.add(
            responses.GET,
            "https://api.coingecko.com/api/v3/simple/price",
            json={"bitcoin": {"usd": 100}},  # Below MIN_BTC_PRICE
            status=200,
        )
        # Coinbase returns valid price
        responses.add(
            responses.GET,
            "https://api.coinbase.com/v2/prices/BTC-USD/spot",
            json={"data": {"amount": "97000"}},
            status=200,
        )
        price = fetch_btc_price_robust()
        assert price == 97000

    @responses.activate
    def test_custom_sources(self):
        """Test using custom sources list."""

        def mock_source():
            return 99999.0

        sources = [(mock_source, "MockSource")]
        price = fetch_btc_price_robust(sources=sources)
        assert price == 99999.0

    @responses.activate
    def test_previous_price_passed_to_validation(self):
        """Test that previous_price is used for validation."""
        responses.add(
            responses.GET,
            "https://api.coingecko.com/api/v3/simple/price",
            json={"bitcoin": {"usd": 97500}},
            status=200,
        )
        # Previous price shouldn't affect validation (large change just warns)
        price = fetch_btc_price_robust(previous_price=50000.0)
        assert price == 97500


# =============================================================================
# Live API Tests (Network-dependent, skipped in CI)
# =============================================================================


@pytest.mark.skipif(
    not os.getenv("RUN_LIVE"), reason="Live API tests are skipped by default; set RUN_LIVE=1 to run"
)
class TestLiveAPIs:
    """Live tests against real APIs (use sparingly to avoid rate limits)."""

    def test_live_coingecko(self):
        """Test live CoinGecko API."""
        price = fetch_price_coingecko()
        assert isinstance(price, float)
        assert MIN_BTC_PRICE <= price <= MAX_BTC_PRICE

    def test_live_coinbase(self):
        """Test live Coinbase API."""
        price = fetch_price_coinbase()
        assert isinstance(price, float)
        assert MIN_BTC_PRICE <= price <= MAX_BTC_PRICE

    def test_live_bitstamp(self):
        """Test live Bitstamp API."""
        price = fetch_price_bitstamp()
        assert isinstance(price, float)
        assert MIN_BTC_PRICE <= price <= MAX_BTC_PRICE

    def test_live_kraken(self):
        """Test live Kraken API."""
        price = fetch_price_kraken()
        assert isinstance(price, float)
        assert MIN_BTC_PRICE <= price <= MAX_BTC_PRICE

    def test_live_binance(self):
        """Test live Binance API."""
        price = fetch_price_binance()
        assert isinstance(price, float)
        assert MIN_BTC_PRICE <= price <= MAX_BTC_PRICE

    def test_live_robust_fetcher(self):
        """Test live robust fetcher returns a valid price."""
        price = fetch_btc_price_robust()
        assert price is not None
        assert isinstance(price, float)
        assert MIN_BTC_PRICE <= price <= MAX_BTC_PRICE


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and unusual scenarios."""

    @responses.activate
    def test_malformed_json_response(self):
        """Test handling of malformed JSON."""
        responses.add(
            responses.GET,
            "https://api.coingecko.com/api/v3/simple/price",
            body="not valid json",
            status=200,
        )
        with pytest.raises(Exception):  # requests will raise JSONDecodeError
            fetch_price_coingecko()

    @responses.activate
    def test_empty_response(self):
        """Test handling of empty response."""
        responses.add(
            responses.GET,
            "https://api.coingecko.com/api/v3/simple/price",
            json={},
            status=200,
        )
        with pytest.raises(ValueError):
            fetch_price_coingecko()

    @responses.activate
    def test_price_as_string_converted_to_float(self):
        """Test that string prices are converted to float."""
        responses.add(
            responses.GET,
            "https://api.coinbase.com/v2/prices/BTC-USD/spot",
            json={"data": {"amount": "97500.12345"}},
            status=200,
        )
        price = fetch_price_coinbase()
        assert isinstance(price, float)
        assert price == 97500.12345

    @responses.activate
    def test_very_precise_price(self):
        """Test handling of high-precision prices."""
        responses.add(
            responses.GET,
            "https://api.binance.com/api/v3/ticker/price",
            json={"symbol": "BTCUSDT", "price": "97500.12345678"},
            status=200,
        )
        price = fetch_price_binance()
        assert price == 97500.12345678

    def test_validate_price_at_boundaries(self):
        """Test validation at exact boundaries."""
        assert validate_price(MIN_BTC_PRICE) is True
        assert validate_price(MAX_BTC_PRICE) is True
        assert validate_price(MIN_BTC_PRICE - 0.01) is False
        assert validate_price(MAX_BTC_PRICE + 0.01) is False
