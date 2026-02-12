"""Tests for Modal app functions in modal_app.py."""

from unittest.mock import patch

import pandas as pd
from stacksats.modal_app import (
    process_start_date_batch_modal,
    daily_export,
    daily_export_retry,
)


class TestModalAppFunctions:
    """Tests for functions in modal_app.py."""

    def test_process_start_date_batch_modal(self):
        """Test the batch processing function used by Modal workers."""
        # Patching inside the function's scope logic
        with patch("pickle.loads") as mock_loads, \
             patch("pandas.to_datetime") as mock_to_datetime, \
             patch("stacksats.export_weights.process_start_date_batch") as mock_process_batch:
            
            # Setup mock data
            mock_loads.side_effect = [
                pd.DataFrame({"feat": [1]}), # features_df
                pd.DataFrame({"PriceUSD": [50000]}) # btc_df
            ]
            mock_to_datetime.return_value = pd.Timestamp("2024-01-01")
            mock_process_batch.return_value = pd.DataFrame({"id": [1]})
            
            args = (
                "2024-01-01",
                ["2025-01-01"],
                "2024-06-01",
                "PriceUSD",
                b"features_pickle",
                b"btc_pickle"
            )
            
            # Access underlying function
            func = process_start_date_batch_modal.get_raw_f()
            result = func(args)
            
            assert isinstance(result, pd.DataFrame)
            assert mock_loads.call_count == 2
            assert mock_process_batch.called

    @patch("stacksats.modal_app.run_export.remote")
    @patch("stacksats.export_weights.get_db_connection")
    @patch("stacksats.export_weights.create_table_if_not_exists")
    @patch("stacksats.export_weights.table_is_empty")
    @patch("stacksats.export_weights.update_today_weights")
    def test_daily_export_logic(self, mock_update, mock_empty, mock_create, mock_get_db, mock_run_remote):
        """Test daily_export logic flow."""
        # Setup mocks
        mock_run_remote.return_value = (pd.DataFrame(), {"export_date": "2024-01-01", "date_ranges": 10, "range_start": "A", "range_end": "B"})
        mock_empty.return_value = False
        mock_update.return_value = 5
        
        # Access underlying function
        func = daily_export.get_raw_f()
        result = func()
        
        assert result["status"] == "success"
        assert result["rows_affected"] == 5
        assert mock_run_remote.called

    @patch("stacksats.export_weights.today_data_exists")
    @patch("stacksats.export_weights.get_db_connection")
    @patch("stacksats.modal_app.run_export.remote")
    @patch("stacksats.export_weights.table_is_empty")
    @patch("stacksats.export_weights.create_table_if_not_exists")
    def test_daily_export_retry_skips_if_data_exists(self, mock_create, mock_empty, mock_run_remote, mock_get_db, mock_today_exists):
        """Test that daily_export_retry skips if data already exists."""
        mock_today_exists.return_value = True
        mock_empty.return_value = False
        
        # Access underlying function
        func = daily_export_retry.get_raw_f()
        result = func()
        
        assert result["status"] == "skipped"
        assert result["reason"] == "data_already_exists"
        assert not mock_run_remote.called

    @patch("stacksats.export_weights.today_data_exists")
    @patch("stacksats.export_weights.get_db_connection")
    @patch("stacksats.modal_app.run_export.remote")
    @patch("stacksats.export_weights.table_is_empty")
    @patch("stacksats.export_weights.create_table_if_not_exists")
    @patch("stacksats.export_weights.update_today_weights")
    def test_daily_export_retry_runs_if_data_missing(self, mock_update, mock_create, mock_empty, mock_run_remote, mock_get_db, mock_today_exists):
        """Test that daily_export_retry runs if data is missing."""
        mock_today_exists.return_value = False
        mock_empty.return_value = False
        
        df_mock = pd.DataFrame({"DCA_date": ["2024-01-01"]})
        # Note: pd.Timestamp.now().strftime("%Y-%m-%d") is used in the function,
        # so we should mock it to match our df_mock
        with patch("pandas.Timestamp.now") as mock_now:
            mock_now.return_value = pd.Timestamp("2024-01-01")
            
            mock_run_remote.return_value = (df_mock, {"status": "success"})
            mock_update.return_value = 10
            
            # Access underlying function
            func = daily_export_retry.get_raw_f()
            result = func()
            
            assert result["status"] == "success"
            assert result["rows_affected"] == 10
            assert mock_run_remote.called
