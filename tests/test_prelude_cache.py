"""Tests for load_data cache behavior."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
from freezegun import freeze_time

from stacksats.prelude import load_data


def _csv_bytes(price_day_1: float, price_day_2: float) -> bytes:
    df = pd.DataFrame(
        {
            "time": ["2024-01-01", "2024-01-02"],
            "PriceUSD": [price_day_1, price_day_2],
            "CapMVRVCur": [2.0, 2.1],
        }
    )
    return df.to_csv(index=False).encode("utf-8")


@freeze_time("2024-01-02")
def test_load_data_uses_fresh_cache_without_network(mocker, tmp_path: Path):
    """When cache is fresh, load_data should read from disk and skip requests."""
    cache_file = tmp_path / "coinmetrics_btc.csv"
    cache_file.write_bytes(_csv_bytes(40000.0, 41000.0))
    mocker.patch("stacksats.prelude.BACKTEST_START", "2024-01-01")
    mocker.patch(
        "stacksats.prelude.pd.Timestamp.now",
        return_value=pd.Timestamp("2024-01-02"),
    )

    # Keep cache mtime fresh and fail fast if network is touched.
    os.utime(cache_file, (time.time(), time.time()))
    mocked_get = mocker.patch("stacksats.prelude.requests.get")
    mocked_get.side_effect = AssertionError("Network should not be used for fresh cache")

    df = load_data(cache_dir=str(tmp_path), max_age_hours=24)
    assert mocked_get.call_count == 0
    assert df.loc[pd.Timestamp("2024-01-02"), "PriceUSD_coinmetrics"] == 41000.0


@freeze_time("2024-01-02")
def test_load_data_refreshes_stale_cache_and_rewrites_file(mocker, tmp_path: Path):
    """When cache is stale, load_data should fetch remotely and update cache file."""
    cache_file = tmp_path / "coinmetrics_btc.csv"
    mocker.patch("stacksats.prelude.BACKTEST_START", "2024-01-01")
    mocker.patch(
        "stacksats.prelude.pd.Timestamp.now",
        return_value=pd.Timestamp("2024-01-02"),
    )
    stale_bytes = _csv_bytes(10000.0, 11000.0)
    fresh_bytes = _csv_bytes(50000.0, 51000.0)
    cache_file.write_bytes(stale_bytes)

    # Force stale mtime: 72 hours old.
    stale_mtime = time.time() - (72 * 3600)
    os.utime(cache_file, (stale_mtime, stale_mtime))

    mock_response = mocker.MagicMock()
    mock_response.content = fresh_bytes
    mock_response.raise_for_status.return_value = None
    mocked_get = mocker.patch("stacksats.prelude.requests.get", return_value=mock_response)

    df = load_data(cache_dir=str(tmp_path), max_age_hours=24)
    assert mocked_get.call_count == 1
    assert cache_file.read_bytes() == fresh_bytes
    assert df.loc[pd.Timestamp("2024-01-02"), "PriceUSD_coinmetrics"] == 51000.0
