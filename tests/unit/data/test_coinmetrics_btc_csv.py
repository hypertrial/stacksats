"""Tests for CoinMetrics BTC CSV fetcher in btc_api/coinmetrics_btc_csv.py."""


import runpy
from pathlib import Path
import pandas as pd
import pytest
import requests
import responses

import stacksats.btc_api.coinmetrics_btc_csv as coinmetrics_btc_csv_module
from stacksats.btc_api.coinmetrics_btc_csv import (
    fetch_coinmetrics_btc_csv,
    COINMETRICS_BTC_CSV_URL,
)


class TestFetchCoinMetricsBTCCSV:
    """Tests for fetch_coinmetrics_btc_csv function."""

    @responses.activate
    def test_fetch_success(self):
        """Test successful CSV fetch and parsing."""
        csv_content = (
            "time,PriceUSD,CapMVRVCur\n"
            "2024-01-01,50000.0,2.0\n"
            "2024-01-02,51000.0,2.1\n"
        )
        responses.add(
            responses.GET,
            COINMETRICS_BTC_CSV_URL,
            body=csv_content,
            status=200,
            content_type="text/csv",
        )

        df = fetch_coinmetrics_btc_csv()

        assert len(df) == 2
        assert isinstance(df.index, pd.DatetimeIndex)
        assert "PriceUSD" in df.columns
        assert "CapMVRVCur" in df.columns
        assert df.iloc[0]["PriceUSD"] == 50000.0
        assert df.index[0] == pd.Timestamp("2024-01-01")

    @responses.activate
    def test_fetch_http_error(self):
        """Test that HTTP errors are raised."""
        responses.add(
            responses.GET,
            COINMETRICS_BTC_CSV_URL,
            status=404,
        )

        with pytest.raises(requests.RequestException):
            fetch_coinmetrics_btc_csv()

    @responses.activate
    def test_invalid_csv_data(self):
        """Test handling of malformed CSV data."""
        responses.add(
            responses.GET,
            COINMETRICS_BTC_CSV_URL,
            body="this is not a csv",
            status=200,
        )

        # pandas will still try to parse it, but let's see what happens
        # It might just create a single-column DF with one row
        # So we should test for missing required columns
        with pytest.raises(ValueError, match="CSV missing required 'time' column"):
            fetch_coinmetrics_btc_csv()

    @responses.activate
    def test_missing_required_columns(self):
        """Test validation of required columns."""
        csv_content = "time,NotPrice\n2024-01-01,50000.0\n"
        responses.add(
            responses.GET,
            COINMETRICS_BTC_CSV_URL,
            body=csv_content,
            status=200,
        )

        with pytest.raises(ValueError, match="CSV missing required 'PriceUSD' column"):
            fetch_coinmetrics_btc_csv()

    @responses.activate
    def test_save_to_path(self, tmp_path):
        """Test saving the fetched CSV to a file."""
        csv_content = "time,PriceUSD\n2024-01-01,50000.0\n"
        responses.add(
            responses.GET,
            COINMETRICS_BTC_CSV_URL,
            body=csv_content,
            status=200,
        )

        save_path = tmp_path / "btc_data.csv"
        fetch_coinmetrics_btc_csv(save_path=save_path)

        assert save_path.exists()
        saved_df = pd.read_csv(save_path)
        assert len(saved_df) == 1
        assert "PriceUSD" in saved_df.columns

    @responses.activate
    def test_duplicate_dates_handled(self):
        """Test that duplicate dates are removed (keep last)."""
        csv_content = (
            "time,PriceUSD\n"
            "2024-01-01,50000.0\n"
            "2024-01-01,50500.0\n"
        )
        responses.add(
            responses.GET,
            COINMETRICS_BTC_CSV_URL,
            body=csv_content,
            status=200,
        )

        df = fetch_coinmetrics_btc_csv()

        assert len(df) == 1
        assert df.iloc[0]["PriceUSD"] == 50500.0


def test_module_dunder_main_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResponse:
        content = b"time,PriceUSD,CapMVRVCur\n2024-01-01,50000,2.0\n"

        @staticmethod
        def raise_for_status():
            return None

    monkeypatch.setattr("requests.get", lambda *args, **kwargs: _FakeResponse())
    module_path = Path(coinmetrics_btc_csv_module.__file__).resolve()
    runpy.run_path(str(module_path), run_name="__main__")
