"""Tests for Modal app functions in modal_app.py."""

from unittest.mock import MagicMock, patch

import pandas as pd
from stacksats.modal_app import (
    daily_export,
    daily_export_retry,
    process_start_date_batch_modal,
    run_export,
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

    def test_run_export_initializes_table_before_lock_reads(self):
        """Test run_export creates table before querying locked history."""
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        btc_df = pd.DataFrame(
            {"PriceUSD_coinmetrics": [100.0, 101.0, 102.0]},
            index=idx,
        )
        features_df = btc_df.copy()
        date_ranges = [(idx.min(), idx.max())]
        grouped_ranges = {idx.min(): [idx.max()]}
        batch_result = pd.DataFrame(
            {
                "id": [0, 1, 2],
                "start_date": [idx.min().strftime("%Y-%m-%d")] * 3,
                "end_date": [idx.max().strftime("%Y-%m-%d")] * 3,
                "DCA_date": idx.strftime("%Y-%m-%d"),
                "btc_usd": [100.0, 101.0, 102.0],
                "weight": [0.3, 0.3, 0.4],
            }
        )

        call_order = []
        mock_conn = MagicMock()
        with patch("stacksats.export_weights.get_db_connection", return_value=mock_conn), patch(
            "stacksats.prelude.load_data", return_value=btc_df
        ), patch(
            "stacksats.model_development.precompute_features", return_value=features_df
        ), patch(
            "stacksats.prelude.generate_date_ranges", return_value=date_ranges
        ), patch(
            "stacksats.prelude.group_ranges_by_start_date",
            return_value=grouped_ranges,
        ), patch.object(
            process_start_date_batch_modal, "map", return_value=[batch_result]
        ), patch(
            "stacksats.export_weights.load_locked_weights_for_window", return_value=None
        ) as mock_load_locked, patch(
            "stacksats.export_weights.create_table_if_not_exists"
        ) as mock_create_table:
            mock_create_table.side_effect = lambda _conn: call_order.append("create")
            mock_load_locked.side_effect = (
                lambda *_args, **_kwargs: call_order.append("load_locked")
            )

            func = run_export.get_raw_f()
            final_df, metadata = func(
                range_start="2024-01-01",
                range_end="2024-01-03",
                min_range_length_days=1,
                btc_price_col="PriceUSD_coinmetrics",
            )

            assert not final_df.empty
            assert metadata["date_ranges"] == 1
            assert mock_conn.close.called
            assert call_order[0] == "create"
            assert "load_locked" in call_order[1:]
