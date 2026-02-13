from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from stacksats.data_btc import BTCDataProvider, _is_cache_usable


PRICE_COL = "PriceUSD_coinmetrics"


def _csv_bytes(rows: list[dict]) -> bytes:
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def _mock_response(content: bytes) -> MagicMock:
    resp = MagicMock()
    resp.content = content
    resp.raise_for_status.return_value = None
    return resp


def test_is_cache_usable_true_for_recent_complete_window() -> None:
    today = pd.Timestamp("2024-01-10")
    csv_bytes = _csv_bytes(
        [
            {"time": "2024-01-08", "PriceUSD": 43000.0},
            {"time": "2024-01-09", "PriceUSD": 43100.0},
            {"time": "2024-01-10", "PriceUSD": 43200.0},
        ]
    )

    assert _is_cache_usable(csv_bytes, pd.Timestamp("2024-01-01"), today) is True


def test_is_cache_usable_false_for_stale_latest_date() -> None:
    today = pd.Timestamp("2024-01-10")
    csv_bytes = _csv_bytes(
        [
            {"time": "2024-01-01", "PriceUSD": 40000.0},
            {"time": "2024-01-02", "PriceUSD": 40100.0},
        ]
    )

    assert _is_cache_usable(csv_bytes, pd.Timestamp("2024-01-01"), today) is False


def test_is_cache_usable_false_for_malformed_bytes() -> None:
    assert _is_cache_usable(b"not-csv", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-10")) is False


def test_load_uses_fresh_usable_cache_without_network(tmp_path: Path, mocker) -> None:
    now = pd.Timestamp("2024-01-05")
    cache_file = tmp_path / "coinmetrics_btc.csv"
    cache_file.write_bytes(
        _csv_bytes(
            [
                {"time": "2024-01-01", "PriceUSD": 40000.0, "CapMVRVCur": 2.0},
                {"time": "2024-01-02", "PriceUSD": 40100.0, "CapMVRVCur": 2.1},
                {"time": "2024-01-03", "PriceUSD": 40200.0, "CapMVRVCur": 2.2},
                {"time": "2024-01-04", "PriceUSD": 40300.0, "CapMVRVCur": 2.3},
                {"time": "2024-01-05", "PriceUSD": 40400.0, "CapMVRVCur": 2.4},
            ]
        )
    )
    os.utime(cache_file, (now.timestamp() - 60, now.timestamp() - 60))

    mocked_get = mocker.patch("stacksats.data_btc.requests.get")
    mocked_get.side_effect = AssertionError("Network should not be used for fresh cache")

    provider = BTCDataProvider(cache_dir=str(tmp_path), max_age_hours=24, clock=lambda: now)
    df = provider.load(backtest_start="2024-01-01")

    assert mocked_get.call_count == 0
    assert float(df.loc[pd.Timestamp("2024-01-05"), PRICE_COL]) == 40400.0


def test_load_refreshes_when_cache_is_unusable(tmp_path: Path, mocker) -> None:
    now = pd.Timestamp("2024-01-10")
    cache_file = tmp_path / "coinmetrics_btc.csv"
    stale_bytes = _csv_bytes(
        [{"time": "2024-01-01", "PriceUSD": 40000.0, "CapMVRVCur": 2.0}]
    )
    fresh_bytes = _csv_bytes(
        [
            {"time": "2024-01-09", "PriceUSD": 43000.0, "CapMVRVCur": 2.0},
            {"time": "2024-01-10", "PriceUSD": 43100.0, "CapMVRVCur": 2.1},
        ]
    )
    cache_file.write_bytes(stale_bytes)
    os.utime(cache_file, (now.timestamp() - 60, now.timestamp() - 60))

    mocked_get = mocker.patch(
        "stacksats.data_btc.requests.get",
        return_value=_mock_response(fresh_bytes),
    )

    provider = BTCDataProvider(cache_dir=str(tmp_path), max_age_hours=24, clock=lambda: now)
    df = provider.load(backtest_start="2024-01-09")

    assert mocked_get.call_count == 1
    assert cache_file.read_bytes() == fresh_bytes
    assert float(df.loc[pd.Timestamp("2024-01-10"), PRICE_COL]) == 43100.0


def test_load_refreshes_stale_cache_by_age(tmp_path: Path, mocker) -> None:
    now = pd.Timestamp("2024-01-10")
    cache_file = tmp_path / "coinmetrics_btc.csv"
    cache_file.write_bytes(
        _csv_bytes(
            [
                {"time": "2024-01-09", "PriceUSD": 42000.0, "CapMVRVCur": 2.0},
                {"time": "2024-01-10", "PriceUSD": 42100.0, "CapMVRVCur": 2.1},
            ]
        )
    )
    stale_mtime = now.timestamp() - (72 * 3600)
    os.utime(cache_file, (stale_mtime, stale_mtime))

    fresh_bytes = _csv_bytes(
        [
            {"time": "2024-01-09", "PriceUSD": 52000.0, "CapMVRVCur": 2.0},
            {"time": "2024-01-10", "PriceUSD": 52100.0, "CapMVRVCur": 2.1},
        ]
    )
    mocked_get = mocker.patch(
        "stacksats.data_btc.requests.get",
        return_value=_mock_response(fresh_bytes),
    )

    provider = BTCDataProvider(cache_dir=str(tmp_path), max_age_hours=24, clock=lambda: now)
    df = provider.load(backtest_start="2024-01-09")

    assert mocked_get.call_count == 1
    assert float(df.loc[pd.Timestamp("2024-01-10"), PRICE_COL]) == 52100.0


def test_load_without_cache_fetches_network(mocker) -> None:
    now = pd.Timestamp("2024-01-02")
    remote_bytes = _csv_bytes(
        [
            {"time": "2024-01-01", "PriceUSD": 40000.0, "CapMVRVCur": 2.0},
            {"time": "2024-01-02", "PriceUSD": 40100.0, "CapMVRVCur": 2.1},
        ]
    )
    mocked_get = mocker.patch(
        "stacksats.data_btc.requests.get",
        return_value=_mock_response(remote_bytes),
    )

    provider = BTCDataProvider(cache_dir=None, clock=lambda: now)
    df = provider.load(backtest_start="2024-01-01")

    assert mocked_get.call_count == 1
    assert float(df.loc[pd.Timestamp("2024-01-02"), PRICE_COL]) == 40100.0


def test_load_fills_historical_and_today_gaps(mocker) -> None:
    now = pd.Timestamp("2024-01-04")
    remote_bytes = _csv_bytes(
        [
            {"time": "2024-01-01", "PriceUSD": 40000.0, "CapMVRVCur": 2.0},
            {"time": "2024-01-03", "PriceUSD": 42000.0, "CapMVRVCur": 2.2},
        ]
    )
    mocker.patch("stacksats.data_btc.requests.get", return_value=_mock_response(remote_bytes))
    historical = mocker.patch("stacksats.data_btc.fetch_btc_price_historical", return_value=41000.0)
    robust = mocker.patch("stacksats.data_btc.fetch_btc_price_robust", return_value=43000.0)

    provider = BTCDataProvider(cache_dir=None, clock=lambda: now)
    df = provider.load(backtest_start="2024-01-01")

    assert float(df.loc[pd.Timestamp("2024-01-02"), PRICE_COL]) == 41000.0
    assert float(df.loc[pd.Timestamp("2024-01-04"), PRICE_COL]) == 43000.0
    assert historical.call_count == 1
    assert robust.call_count == 1


def test_load_falls_back_to_previous_price_when_fetch_returns_none(mocker) -> None:
    now = pd.Timestamp("2024-01-03")
    remote_bytes = _csv_bytes(
        [
            {"time": "2024-01-01", "PriceUSD": 40000.0, "CapMVRVCur": 2.0},
            {"time": "2024-01-03", "PriceUSD": 42000.0, "CapMVRVCur": 2.2},
        ]
    )
    mocker.patch("stacksats.data_btc.requests.get", return_value=_mock_response(remote_bytes))
    mocker.patch("stacksats.data_btc.fetch_btc_price_historical", return_value=None)
    mocker.patch("stacksats.data_btc.fetch_btc_price_robust", return_value=42000.0)

    provider = BTCDataProvider(cache_dir=None, clock=lambda: now)
    df = provider.load(backtest_start="2024-01-01")

    assert float(df.loc[pd.Timestamp("2024-01-02"), PRICE_COL]) == 40000.0


def test_load_uses_yesterday_mvrv_when_today_missing(mocker) -> None:
    now = pd.Timestamp("2024-01-03")
    remote_bytes = _csv_bytes(
        [
            {"time": "2024-01-01", "PriceUSD": 40000.0, "CapMVRVCur": 2.0},
            {"time": "2024-01-02", "PriceUSD": 41000.0, "CapMVRVCur": 2.1},
            {"time": "2024-01-03", "PriceUSD": 42000.0, "CapMVRVCur": None},
        ]
    )
    mocker.patch("stacksats.data_btc.requests.get", return_value=_mock_response(remote_bytes))

    provider = BTCDataProvider(cache_dir=None, clock=lambda: now)
    df = provider.load(backtest_start="2024-01-01")

    assert float(df.loc[pd.Timestamp("2024-01-03"), "CapMVRVCur"]) == 2.1


def test_load_raises_when_required_prices_remain_missing(mocker) -> None:
    now = pd.Timestamp("2024-01-02")
    remote_bytes = _csv_bytes(
        [
            {"time": "2024-01-01", "PriceUSD": None, "CapMVRVCur": 2.0},
            {"time": "2024-01-02", "PriceUSD": 41000.0, "CapMVRVCur": 2.1},
        ]
    )
    mocker.patch("stacksats.data_btc.requests.get", return_value=_mock_response(remote_bytes))
    mocker.patch("stacksats.data_btc.fetch_btc_price_historical", return_value=None)
    mocker.patch("stacksats.data_btc.fetch_btc_price_robust", return_value=None)

    provider = BTCDataProvider(cache_dir=None, clock=lambda: now)
    with pytest.raises(ValueError, match="prices missing in required date range"):
        provider.load(backtest_start="2024-01-01")
