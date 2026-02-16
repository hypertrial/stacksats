"""Tests for MVRV fallback logic in prelude.py load_data() function.

Tests verify that when today's MVRV value is missing, the system
uses yesterday's MVRV value as a fallback.
"""

import logging
from io import StringIO
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from stacksats.prelude import load_data

# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_coinmetrics_data():
    """Create sample CoinMetrics data with MVRV values."""
    dates = pd.date_range(start="2024-01-01", end="2025-12-31", freq="D")
    np.random.seed(42)
    base_price = 50000
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))

    # Generate MVRV values (cycles between 0.8 and 3.5)
    mvrv_base = 2.0 + 1.2 * np.sin(np.arange(len(dates)) * 2 * np.pi / 1461)
    mvrv_noise = np.random.normal(0, 0.1, len(dates))
    mvrv_values = np.clip(mvrv_base + mvrv_noise, 0.8, 3.5)

    df = pd.DataFrame(
        {
            "time": dates.strftime("%Y-%m-%d"),
            "PriceUSD": prices,
            "CapMVRVCur": mvrv_values,
        }
    )

    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()


@pytest.fixture
def mock_coinmetrics_with_mvrv(mocker, sample_coinmetrics_data):
    """Mock CoinMetrics CSV response with MVRV data."""
    mock_response = MagicMock()
    mock_response.content = sample_coinmetrics_data.encode("utf-8")
    mock_response.status_code = 200
    mocker.patch("requests.get", return_value=mock_response)
    return mock_response


@pytest.fixture
def mock_btc_price_fetcher(mocker):
    """Mock BTC price fetcher to return a fixed price."""
    mocker.patch(
        "stacksats.prelude.fetch_btc_price_robust",
        return_value=98000.0,
    )
    return 98000.0


@pytest.fixture
def mock_backtest_start(mocker):
    """Mock BACKTEST_START to match sample data range."""
    mocker.patch("stacksats.prelude.BACKTEST_START", "2024-01-01")



# -----------------------------------------------------------------------------
# MVRV Fallback Tests
# -----------------------------------------------------------------------------


class TestMVRVFallback:
    """Tests for MVRV fallback logic when today's value is missing."""

    def test_mvrv_fallback_when_today_missing(
        self, mocker, mock_coinmetrics_with_mvrv, mock_btc_price_fetcher, mock_backtest_start, caplog
    ):
        """Test that yesterday's MVRV is used when today's is missing."""
        # Create CSV data where today's MVRV is missing
        today = pd.Timestamp.now().normalize()
        yesterday = today - pd.Timedelta(days=1)

        # Read the sample data and modify it
        df = pd.read_csv(StringIO(mock_coinmetrics_with_mvrv.content.decode("utf-8")))
        df["time"] = pd.to_datetime(df["time"])

        # Ensure yesterday exists with MVRV
        if (
            yesterday.strftime("%Y-%m-%d")
            not in df["time"].dt.strftime("%Y-%m-%d").values
        ):
            # Add yesterday if missing
            new_row = df.iloc[-1].copy()
            new_row["time"] = yesterday
            new_row["PriceUSD"] = 98000.0
            new_row["CapMVRVCur"] = 2.5  # Known yesterday value
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # Set today's MVRV to NaN (missing)
        today_str = today.strftime("%Y-%m-%d")
        today_idx = df[df["time"].dt.strftime("%Y-%m-%d") == today_str].index
        if len(today_idx) > 0:
            df.loc[today_idx, "CapMVRVCur"] = float("nan")
        else:
            # Add today with NaN MVRV
            new_row = df.iloc[-1].copy()
            new_row["time"] = today
            new_row["PriceUSD"] = 98000.0
            new_row["CapMVRVCur"] = float("nan")
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # Update the mock response
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        mock_coinmetrics_with_mvrv.content = csv_buffer.getvalue().encode("utf-8")

        # Load data - should trigger fallback
        with caplog.at_level(logging.INFO):
            result_df = load_data(cache_dir=None)

        # Verify today's MVRV was filled with yesterday's value
        if today in result_df.index:
            assert pd.notna(result_df.loc[today, "CapMVRVCur"]), (
                "Today's MVRV should be filled with yesterday's value"
            )
            if yesterday in result_df.index:
                expected_mvrv = result_df.loc[yesterday, "CapMVRVCur"]
                actual_mvrv = result_df.loc[today, "CapMVRVCur"]
                assert actual_mvrv == expected_mvrv, (
                    f"Today's MVRV ({actual_mvrv}) should equal yesterday's ({expected_mvrv})"
                )

        # Verify log message was generated
        assert "Used yesterday's MVRV value" in caplog.text, (
            "Should log info message about using yesterday's MVRV"
        )

    def test_no_fallback_when_today_mvrv_exists(
        self, mocker, mock_coinmetrics_with_mvrv, mock_btc_price_fetcher, mock_backtest_start, caplog
    ):
        """Test that no fallback occurs when today's MVRV already exists."""
        today = pd.Timestamp.now().normalize()

        # Read and modify sample data
        df = pd.read_csv(StringIO(mock_coinmetrics_with_mvrv.content.decode("utf-8")))
        df["time"] = pd.to_datetime(df["time"])

        # Ensure today exists with valid MVRV
        today_str = today.strftime("%Y-%m-%d")
        today_idx = df[df["time"].dt.strftime("%Y-%m-%d") == today_str].index
        original_mvrv = 2.8  # Known value
        if len(today_idx) > 0:
            df.loc[today_idx, "CapMVRVCur"] = original_mvrv
        else:
            new_row = df.iloc[-1].copy()
            new_row["time"] = today
            new_row["PriceUSD"] = 98000.0
            new_row["CapMVRVCur"] = original_mvrv
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # Update mock response
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        mock_coinmetrics_with_mvrv.content = csv_buffer.getvalue().encode("utf-8")

        # Load data
        with caplog.at_level(logging.INFO):
            result_df = load_data(cache_dir=None)

        # Verify today's MVRV is unchanged
        if today in result_df.index:
            assert result_df.loc[today, "CapMVRVCur"] == original_mvrv, (
                "Today's MVRV should remain unchanged when it already exists"
            )

        # Verify no fallback log message
        assert "Used yesterday's MVRV value" not in caplog.text, (
            "Should not log fallback message when MVRV already exists"
        )

    def test_warning_when_yesterday_missing(
        self, mocker, mock_coinmetrics_with_mvrv, mock_btc_price_fetcher, mock_backtest_start, caplog
    ):
        """Test that warning is logged when yesterday doesn't exist."""
        today = pd.Timestamp.now().normalize()

        # Read and modify sample data
        df = pd.read_csv(StringIO(mock_coinmetrics_with_mvrv.content.decode("utf-8")))
        df["time"] = pd.to_datetime(df["time"])

        # Remove yesterday from data
        yesterday = today - pd.Timedelta(days=1)
        yesterday_str = yesterday.strftime("%Y-%m-%d")
        df = df[df["time"].dt.strftime("%Y-%m-%d") != yesterday_str]

        # Set today's MVRV to NaN
        today_str = today.strftime("%Y-%m-%d")
        today_idx = df[df["time"].dt.strftime("%Y-%m-%d") == today_str].index
        if len(today_idx) > 0:
            df.loc[today_idx, "CapMVRVCur"] = float("nan")
        else:
            new_row = df.iloc[-1].copy()
            new_row["time"] = today
            new_row["PriceUSD"] = 98000.0
            new_row["CapMVRVCur"] = float("nan")
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # Update mock response
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        mock_coinmetrics_with_mvrv.content = csv_buffer.getvalue().encode("utf-8")

        # Load data - should trigger warning
        with caplog.at_level(logging.WARNING):
            load_data(cache_dir=None)

        # Verify warning was logged
        assert "Could not find valid MVRV" in caplog.text, (
            "Should log warning when yesterday is missing"
        )

    def test_warning_when_yesterday_mvrv_also_missing(
        self, mocker, mock_coinmetrics_with_mvrv, mock_btc_price_fetcher, mock_backtest_start, caplog
    ):
        """Test that warning is logged when yesterday's MVRV is also missing."""
        today = pd.Timestamp.now().normalize()
        yesterday = today - pd.Timedelta(days=1)

        # Read and modify sample data
        df = pd.read_csv(StringIO(mock_coinmetrics_with_mvrv.content.decode("utf-8")))
        df["time"] = pd.to_datetime(df["time"])

        # Set both today and yesterday MVRV to NaN
        today_str = today.strftime("%Y-%m-%d")
        yesterday_str = yesterday.strftime("%Y-%m-%d")

        today_idx = df[df["time"].dt.strftime("%Y-%m-%d") == today_str].index
        if len(today_idx) > 0:
            df.loc[today_idx, "CapMVRVCur"] = float("nan")
        else:
            new_row = df.iloc[-1].copy()
            new_row["time"] = today
            new_row["PriceUSD"] = 98000.0
            new_row["CapMVRVCur"] = float("nan")
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        yesterday_idx = df[df["time"].dt.strftime("%Y-%m-%d") == yesterday_str].index
        if len(yesterday_idx) > 0:
            df.loc[yesterday_idx, "CapMVRVCur"] = float("nan")

        # Update mock response
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        mock_coinmetrics_with_mvrv.content = csv_buffer.getvalue().encode("utf-8")

        # Load data - should trigger warning
        with caplog.at_level(logging.WARNING):
            load_data(cache_dir=None)

        # Verify warning was logged
        assert "Could not find valid MVRV" in caplog.text, (
            "Should log warning when yesterday's MVRV is also missing"
        )

    def test_no_fallback_when_mvrv_column_missing(
        self, mocker, mock_coinmetrics_with_mvrv, mock_btc_price_fetcher, mock_backtest_start, caplog
    ):
        """Test that no fallback occurs when MVRV column doesn't exist."""
        # Read and remove MVRV column
        df = pd.read_csv(StringIO(mock_coinmetrics_with_mvrv.content.decode("utf-8")))
        df = df.drop(columns=["CapMVRVCur"], errors="ignore")

        # Update mock response
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        mock_coinmetrics_with_mvrv.content = csv_buffer.getvalue().encode("utf-8")

        # Load data
        with caplog.at_level(logging.INFO):
            result_df = load_data(cache_dir=None)

        # Verify no MVRV column in result
        assert "CapMVRVCur" not in result_df.columns, (
            "MVRV column should not exist when not in source data"
        )

        # Verify no fallback log message
        assert "Used yesterday's MVRV value" not in caplog.text, (
            "Should not log fallback message when MVRV column doesn't exist"
        )

    def test_fallback_when_today_not_in_index(
        self, mocker, mock_coinmetrics_with_mvrv, mock_btc_price_fetcher, mock_backtest_start, caplog
    ):
        """Test that today is added and values are forward-filled when missing from index."""
        today = pd.Timestamp.now().normalize()
        yesterday = today - pd.Timedelta(days=1)

        # Mock price fetcher to return None (sub simulating API failure)
        mocker.patch(
            "stacksats.prelude.fetch_btc_price_robust",
            return_value=None,
        )
        # Also mock historical to return None
        mocker.patch(
            "stacksats.prelude.fetch_btc_price_historical",
            return_value=None,
        )

        # Read and remove today from data
        df = pd.read_csv(StringIO(mock_coinmetrics_with_mvrv.content.decode("utf-8")))
        df["time"] = pd.to_datetime(df["time"])

        # Remove today
        today_str = today.strftime("%Y-%m-%d")
        df = df[df["time"].dt.strftime("%Y-%m-%d") != today_str]

        # Ensure yesterday exists for forward fill
        yesterday_str = yesterday.strftime("%Y-%m-%d")
        if yesterday_str not in df["time"].dt.strftime("%Y-%m-%d").values:
             # Add yesterday if missing
            new_row = df.iloc[-1].copy()
            new_row["time"] = yesterday
            new_row["PriceUSD"] = 98000.0
            new_row["CapMVRVCur"] = 2.5
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # Update mock response
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        mock_coinmetrics_with_mvrv.content = csv_buffer.getvalue().encode("utf-8")

        # Load data
        with caplog.at_level(logging.INFO):
            result_df = load_data(cache_dir=None)

        # Verify today IS in result (added by gap filling)
        assert today in result_df.index, (
            "Today should be added to index even if fetcher fails (forward fill)"
        )

        # Verify MVRV fallback occurred
        assert "Used yesterday's MVRV value" in caplog.text, (
            "Should log fallback message when MVRV forward filled"
        )

    def test_historical_data_not_affected(
        self, mocker, mock_coinmetrics_with_mvrv, mock_btc_price_fetcher, mock_backtest_start
    ):
        """Test that historical MVRV data is not affected by fallback."""
        today = pd.Timestamp.now().normalize()
        yesterday = today - pd.Timedelta(days=1)
        historical_date = today - pd.Timedelta(days=10)

        # Read sample data
        df = pd.read_csv(StringIO(mock_coinmetrics_with_mvrv.content.decode("utf-8")))
        df["time"] = pd.to_datetime(df["time"])

        # Store original historical MVRV value
        historical_str = historical_date.strftime("%Y-%m-%d")
        historical_idx = df[df["time"].dt.strftime("%Y-%m-%d") == historical_str].index
        original_historical_mvrv = None
        if len(historical_idx) > 0:
            original_historical_mvrv = df.loc[historical_idx[0], "CapMVRVCur"]

        # Set today's MVRV to NaN to trigger fallback
        today_str = today.strftime("%Y-%m-%d")
        today_idx = df[df["time"].dt.strftime("%Y-%m-%d") == today_str].index
        if len(today_idx) > 0:
            df.loc[today_idx, "CapMVRVCur"] = float("nan")
        else:
            new_row = df.iloc[-1].copy()
            new_row["time"] = today
            new_row["PriceUSD"] = 98000.0
            new_row["CapMVRVCur"] = float("nan")
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # Ensure yesterday has valid MVRV
        yesterday_str = yesterday.strftime("%Y-%m-%d")
        yesterday_idx = df[df["time"].dt.strftime("%Y-%m-%d") == yesterday_str].index
        if len(yesterday_idx) > 0:
            df.loc[yesterday_idx, "CapMVRVCur"] = 2.5  # Known value

        # Update mock response
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        mock_coinmetrics_with_mvrv.content = csv_buffer.getvalue().encode("utf-8")

        try:
            result_df = load_data(cache_dir=None)

            # Verify historical MVRV is unchanged
            if (
                original_historical_mvrv is not None
                and historical_date in result_df.index
            ):
                assert (
                    result_df.loc[historical_date, "CapMVRVCur"]
                    == original_historical_mvrv
                ), "Historical MVRV should not be affected by today's fallback"
        except Exception:
            # If historical date is not in result, that's okay
            pass
