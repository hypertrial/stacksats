"""Tests for plotting scripts plot_mvrv.py and plot_weights.py."""

import datetime as dt
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from stacksats.plot_mvrv import main as main_mvrv
from stacksats.plot_weights import main as main_weights


class TestPlottingScripts:
    """Tests for plotting scripts."""

    @patch("stacksats.plot_mvrv.BTCDataProvider.load")
    @patch("stacksats.plot_mvrv.plt.savefig")
    def test_plot_mvrv_main(self, mock_savefig, mock_load):
        """Test plot_mvrv.py main function."""
        dates = pl.datetime_range(
            dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 3),
            interval="1d", eager=True
        ).to_list()
        df = pl.DataFrame({
            "date": dates,
            "mvrv": [1.0, 2.0, 3.0],
            "CapMVRVZ": [0.5, 1.5, 2.5],
        })
        mock_load.return_value = df

        # Call main with no arguments (uses defaults)
        with patch("sys.argv", ["stacksats.plot_mvrv.py"]):
            main_mvrv()

        assert mock_savefig.called
        assert mock_load.called

    @patch("stacksats.plot_weights.get_db_connection")
    @patch("stacksats.plot_weights.plt.savefig")
    @patch("stacksats.plot_weights.validate_date_range")
    def test_plot_weights_main(self, mock_validate, mock_savefig, mock_get_db):
        """Test plot_weights.py main function."""
        # Mock database connection
        mock_conn = MagicMock()
        mock_get_db.return_value = mock_conn
        mock_validate.return_value = True

        # Mock cursor.fetchall() since plot_weights.py uses it instead of read_sql_query
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = [
            ("2024-01-01", 0.5, 50000.0, 1),
            ("2024-01-02", 0.5, 51000.0, 2)
        ]

        # Call main with positional arguments
        with patch("sys.argv", ["stacksats.plot_weights.py", "2024-01-01", "2024-12-31"]):
            main_weights()

        assert mock_get_db.called
        assert cursor.execute.called
        assert mock_savefig.called
        assert mock_conn.close.called

    @patch("stacksats.plot_weights.get_db_connection")
    @patch("stacksats.plot_weights.get_oldest_date_range")
    def test_plot_weights_empty_df(self, mock_get_oldest, mock_get_db):
        """Test plot_weights.py with empty data handling."""
        mock_get_db.return_value = MagicMock()
        # Mock get_oldest_date_range to raise Exception if no ranges found
        mock_get_oldest.side_effect = Exception("No date ranges found")

        with patch("sys.argv", ["stacksats.plot_weights.py"]):
            with pytest.raises(SystemExit) as excinfo:
                main_weights()
            assert excinfo.value.code == 1

        assert mock_get_oldest.called
