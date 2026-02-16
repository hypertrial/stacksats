"""Tests for gap filling logic in prelude.py."""

import pytest
import pandas as pd
import responses
from io import BytesIO
from unittest.mock import patch

from stacksats.prelude import load_data

@pytest.fixture
def mock_coinmetrics_csv():
    """Create a mock CoinMetrics CSV with a gap."""
    dates = pd.date_range(start="2024-01-01", end="2024-01-05", freq="D")
    df = pd.DataFrame(index=dates)
    df["PriceUSD"] = [40000.0, 41000.0, None, 43000.0, 44000.0]  # Hole on Jan 3rd
    df["CapMVRVCur"] = 2.0
    df.index.name = "time"

    csv_buf = BytesIO()
    df.to_csv(csv_buf)
    csv_buf.seek(0)
    return csv_buf.read()

class TestPreludeGapFilling:
    """Tests for load_data() gap filling behavior."""

    @responses.activate
    @patch("stacksats.prelude.BACKTEST_START", "2024-01-01")
    @patch("stacksats.prelude.fetch_btc_price_historical")
    @patch("stacksats.prelude.fetch_btc_price_robust")
    def test_load_data_fills_historical_gap(self, mock_robust, mock_historical, mock_coinmetrics_csv):
        """Test that load_data identifies and fills a gap in historical data."""
        # Mock CoinMetrics response
        responses.add(
            responses.GET,
            "https://raw.githubusercontent.com/coinmetrics/data/refs/heads/master/csv/btc.csv",
            body=mock_coinmetrics_csv,
            status=200,
        )

        # Jan 3rd is missing
        gap_date = pd.Timestamp("2024-01-03")
        mock_historical.return_value = 42000.0

        # Mock robust fetcher for "today" (even if Jan 5 is latest in CSV, pd.Timestamp.now() will be later)
        mock_robust.return_value = 45000.0

        df = load_data(cache_dir=None)

        # Verify Jan 3rd was filled
        assert df.loc[gap_date, "PriceUSD_coinmetrics"] == 42000.0
        assert mock_historical.called

        # Verify no NaNs in backtest range
        assert not df.loc["2024-01-01":"2024-01-03", "PriceUSD_coinmetrics"].isna().any()

    @responses.activate
    @patch("stacksats.prelude.BACKTEST_START", "2024-01-01")
    @patch("stacksats.prelude.fetch_btc_price_historical")
    def test_load_data_forward_fill_fallback(self, mock_historical, mock_coinmetrics_csv):
        """Test that load_data forward-fills if API fetching fails."""
        responses.add(
            responses.GET,
            "https://raw.githubusercontent.com/coinmetrics/data/refs/heads/master/csv/btc.csv",
            body=mock_coinmetrics_csv,
            status=200,
        )

        # Mock API failure for historical fetch
        mock_historical.return_value = None

        # Previous price before hole (Jan 3) is Jan 2nd ($41000)
        df = load_data(cache_dir=None)

        gap_date = pd.Timestamp("2024-01-03")
        # Should have forward-filled from Jan 2nd (41000)
        assert df.loc[gap_date, "PriceUSD_coinmetrics"] == 41000.0

    @responses.activate
    @patch("stacksats.prelude.BACKTEST_START", "2024-01-01")
    def test_load_data_assertion_on_unresolvable_gap(self):
        """Test that load_data raises AssertionError if a gap is truly unresolvable (e.g., first date missing)."""
        # Create CSV where the VERY FIRST date is missing PriceUSD
        df_bad = pd.DataFrame({
            "time": ["2024-01-01", "2024-01-02"],
            "PriceUSD": [None, 40000.0],
            "CapMVRVCur": [2.0, 2.0]
        })
        csv_buf = BytesIO()
        df_bad.to_csv(csv_buf, index=False)

        responses.add(
            responses.GET,
            "https://raw.githubusercontent.com/coinmetrics/data/refs/heads/master/csv/btc.csv",
            body=csv_buf.getvalue(),
            status=200,
        )

        with patch("stacksats.prelude.fetch_btc_price_historical", return_value=None):
            with pytest.raises(AssertionError, match="dates still missing BTC-USD prices"):
                load_data(cache_dir=None)
