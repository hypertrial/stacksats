from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from stacksats.data_btc import BTCDataProvider, DataLoadError, _is_cache_usable


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


def test_is_cache_usable_true_for_past_window_without_recent_requirement() -> None:
    today = pd.Timestamp("2024-01-10")
    csv_bytes = _csv_bytes(
        [
            {"time": "2023-12-30", "PriceUSD": 42000.0},
            {"time": "2023-12-31", "PriceUSD": 42100.0},
        ]
    )

    assert (
        _is_cache_usable(
            csv_bytes,
            pd.Timestamp("2023-12-01"),
            today,
            target_end=pd.Timestamp("2023-12-31"),
            require_recent=False,
        )
        is True
    )


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


def test_load_past_only_uses_cache_without_network(tmp_path: Path, mocker) -> None:
    now = pd.Timestamp("2026-02-17")
    cache_file = tmp_path / "coinmetrics_btc.csv"
    cache_file.write_bytes(
        _csv_bytes(
            [
                {"time": "2025-12-30", "PriceUSD": 94000.0, "CapMVRVCur": 2.0},
                {"time": "2025-12-31", "PriceUSD": 95000.0, "CapMVRVCur": 2.1},
            ]
        )
    )
    stale_mtime = now.timestamp() - (90 * 24 * 3600)
    os.utime(cache_file, (stale_mtime, stale_mtime))

    mocked_get = mocker.patch("stacksats.data_btc.requests.get")
    mocked_get.side_effect = AssertionError("Network should not be used for past-only windows")

    provider = BTCDataProvider(cache_dir=str(tmp_path), max_age_hours=24, clock=lambda: now)
    df = provider.load(backtest_start="2025-12-30", end_date="2025-12-31")

    assert mocked_get.call_count == 0
    assert float(df.loc[pd.Timestamp("2025-12-31"), PRICE_COL]) == 95000.0


def test_load_past_only_raises_when_coinmetrics_has_missing_dates(
    tmp_path: Path, mocker
) -> None:
    now = pd.Timestamp("2024-01-10")
    cache_file = tmp_path / "coinmetrics_btc.csv"
    cache_file.write_bytes(
        _csv_bytes(
            [
                {"time": "2024-01-01", "PriceUSD": 40000.0, "CapMVRVCur": 2.0},
                {"time": "2024-01-03", "PriceUSD": 42000.0, "CapMVRVCur": 2.2},
            ]
        )
    )
    os.utime(cache_file, (now.timestamp() - 60, now.timestamp() - 60))

    mocked_get = mocker.patch("stacksats.data_btc.requests.get")
    mocked_get.side_effect = AssertionError("Network should not be used for past-only windows")

    provider = BTCDataProvider(cache_dir=str(tmp_path), max_age_hours=24, clock=lambda: now)
    with pytest.raises(DataLoadError, match="missing dates"):
        provider.load(backtest_start="2024-01-01", end_date="2024-01-03")


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
    provider = BTCDataProvider(cache_dir=None, clock=lambda: now)
    with pytest.raises(DataLoadError, match="missing dates"):
        provider.load(backtest_start="2024-01-01")


def test_load_falls_back_to_previous_price_when_fetch_returns_none(mocker) -> None:
    now = pd.Timestamp("2024-01-03")
    remote_bytes = _csv_bytes(
        [
            {"time": "2024-01-01", "PriceUSD": 40000.0, "CapMVRVCur": 2.0},
            {"time": "2024-01-03", "PriceUSD": 42000.0, "CapMVRVCur": 2.2},
        ]
    )
    mocker.patch("stacksats.data_btc.requests.get", return_value=_mock_response(remote_bytes))
    provider = BTCDataProvider(cache_dir=None, clock=lambda: now)
    with pytest.raises(DataLoadError, match="missing dates"):
        provider.load(backtest_start="2024-01-01")


def test_load_preserves_missing_today_mvrv(mocker) -> None:
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

    assert pd.isna(df.loc[pd.Timestamp("2024-01-03"), "CapMVRVCur"])


def test_load_raises_when_required_prices_remain_missing(mocker) -> None:
    now = pd.Timestamp("2024-01-02")
    remote_bytes = _csv_bytes(
        [
            {"time": "2024-01-01", "PriceUSD": None, "CapMVRVCur": 2.0},
            {"time": "2024-01-02", "PriceUSD": 41000.0, "CapMVRVCur": 2.1},
        ]
    )
    mocker.patch("stacksats.data_btc.requests.get", return_value=_mock_response(remote_bytes))
    provider = BTCDataProvider(cache_dir=None, clock=lambda: now)
    with pytest.raises(DataLoadError, match="missing PriceUSD values"):
        provider.load(backtest_start="2024-01-01")
