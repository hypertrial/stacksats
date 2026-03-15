"""Historical BTC price fetching tests (API sources only)."""

from __future__ import annotations

from datetime import datetime

import pytest
import responses

from stacksats.btc_price_fetcher import (
    fetch_btc_price_historical,
    fetch_historical_price_binance,
    fetch_historical_price_coingecko,
)

TEST_DATE = datetime(2024, 1, 1)


@responses.activate
def test_coingecko_historical_success() -> None:
    responses.add(
        responses.GET,
        "https://api.coingecko.com/api/v3/coins/bitcoin/history",
        json={"market_data": {"current_price": {"usd": 42500.0}}},
        status=200,
    )

    price = fetch_historical_price_coingecko(TEST_DATE)

    assert price == 42500.0
    assert responses.calls[0].request.params["date"] == "01-01-2024"


@responses.activate
def test_coingecko_historical_invalid_payload() -> None:
    responses.add(
        responses.GET,
        "https://api.coingecko.com/api/v3/coins/bitcoin/history",
        json={"id": "bitcoin"},
        status=200,
    )

    with pytest.raises(ValueError, match="Invalid CoinGecko historical response"):
        fetch_historical_price_coingecko(TEST_DATE)


@responses.activate
def test_binance_historical_success() -> None:
    responses.add(
        responses.GET,
        "https://api.binance.com/api/v3/klines",
        json=[
            [
                1704067200000,
                "42000.0",
                "43000.0",
                "41500.0",
                "42600.0",
                "100.0",
                1704153599999,
                "4260000.0",
                50000,
                "50.0",
                "2130000.0",
                "0",
            ]
        ],
        status=200,
    )

    price = fetch_historical_price_binance(TEST_DATE)

    assert price == 42600.0
    expected_ts = int(TEST_DATE.timestamp() * 1000)
    assert responses.calls[0].request.params["startTime"] == str(expected_ts)


@responses.activate
def test_binance_historical_empty_payload() -> None:
    responses.add(
        responses.GET,
        "https://api.binance.com/api/v3/klines",
        json=[],
        status=200,
    )

    with pytest.raises(ValueError, match="No Binance historical data"):
        fetch_historical_price_binance(TEST_DATE)


def test_historical_fetcher_falls_back_to_binance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stacksats import btc_price_fetcher as fetcher

    monkeypatch.setattr(
        fetcher,
        "fetch_historical_price_coingecko",
        lambda _date: (_ for _ in ()).throw(ValueError("rate limit")),
    )
    monkeypatch.setattr(fetcher, "fetch_historical_price_binance", lambda _date: 43000.0)

    assert fetcher.fetch_btc_price_historical(TEST_DATE) == 43000.0


def test_historical_fetcher_returns_none_when_all_sources_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stacksats import btc_price_fetcher as fetcher

    monkeypatch.setattr(
        fetcher,
        "fetch_historical_price_coingecko",
        lambda _date: (_ for _ in ()).throw(ValueError("fail-cg")),
    )
    monkeypatch.setattr(
        fetcher,
        "fetch_historical_price_binance",
        lambda _date: (_ for _ in ()).throw(ValueError("fail-binance")),
    )

    assert fetcher.fetch_btc_price_historical(TEST_DATE) is None


def test_historical_fetcher_invalid_price_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stacksats import btc_price_fetcher as fetcher

    monkeypatch.setattr(fetcher, "fetch_historical_price_coingecko", lambda _date: 10.0)
    monkeypatch.setattr(fetcher, "fetch_historical_price_binance", lambda _date: 42000.0)
    monkeypatch.setattr(
        fetcher,
        "validate_price",
        lambda price, previous_price=None: price > 1000.0,
    )

    assert fetch_btc_price_historical(TEST_DATE) == 42000.0
