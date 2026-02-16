"""Tests for database operations in export_weights.py."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from stacksats.export_weights import (
    create_table_if_not_exists,
    insert_all_data,
    table_is_empty,
    today_data_exists,
    update_today_weights,
)


@pytest.fixture
def mock_conn():
    """Mock database connection and cursor."""
    conn = MagicMock()
    cursor = MagicMock()
    # Mock connection encoding for execute_values
    cursor.connection.encoding = "utf-8"
    # Mock rowcount to avoid formatting errors in logs
    cursor.rowcount = 10
    conn.cursor.return_value.__enter__.return_value = cursor
    return conn


class TestDatabaseOperations:
    """Tests for database helper functions in export_weights.py."""

    def test_create_table_if_not_exists(self, mock_conn):
        """Test that CREATE TABLE SQL is executed."""
        create_table_if_not_exists(mock_conn)
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        # Should execute at least once (CREATE TABLE)
        assert cursor.execute.called
        call_args = cursor.execute.call_args_list[0][0][0]
        assert "CREATE TABLE IF NOT EXISTS bitcoin_dca" in call_args

    def test_table_is_empty_true(self, mock_conn):
        """Test table_is_empty returns True when count is 0."""
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = (0,)

        assert table_is_empty(mock_conn) is True
        assert "SELECT COUNT(*)" in cursor.execute.call_args[0][0]

    def test_table_is_empty_false(self, mock_conn):
        """Test table_is_empty returns False when count > 0."""
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = (100,)

        assert table_is_empty(mock_conn) is False

    def test_today_data_exists_true(self, mock_conn):
        """Test today_data_exists returns True when today's data is found."""
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = (1,)

        assert today_data_exists(mock_conn, "2024-01-01") is True
        assert "SELECT COUNT(*)" in cursor.execute.call_args[0][0]
        assert "2024-01-01" in cursor.execute.call_args[0][1]

    def test_today_data_exists_false(self, mock_conn):
        """Test today_data_exists returns False when today's data is missing."""
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = (0,)

        assert today_data_exists(mock_conn, "2024-01-01") is False

    @patch("time.time", side_effect=[100, 101, 102, 103])
    def test_insert_all_data_copy_success(self, mock_time, mock_conn):
        """Test insert_all_data uses COPY FROM successfully."""
        df = pd.DataFrame({
            "day_index": [0],
            "start_date": ["2024-01-01"],
            "end_date": ["2025-01-01"],
            "date": ["2024-01-01"],
            "price_usd": [50000.0],
            "weight": [1.0]
        })

        cursor = mock_conn.cursor.return_value.__enter__.return_value
        # Mock fetchone for count before/after
        cursor.fetchone.side_effect = [(0,), (1,)]
        cursor.rowcount = 1

        actual_inserted = insert_all_data(mock_conn, df)

        assert actual_inserted == 1
        assert cursor.copy_from.called
        assert "temp_bitcoin_dca" in cursor.copy_from.call_args[0][1]

    @patch("stacksats.export_weights.execute_values")
    @patch("time.time", side_effect=[100, 101, 102, 103])
    def test_insert_all_data_fallback_to_execute_values(self, mock_time, mock_execute_values, mock_conn):
        """Test insert_all_data falls back to execute_values if COPY fails."""
        df = pd.DataFrame({
            "day_index": [0],
            "start_date": ["2024-01-01"],
            "end_date": ["2025-01-01"],
            "date": ["2024-01-01"],
            "price_usd": [50000.0],
            "weight": [1.0]
        })

        cursor = mock_conn.cursor.return_value.__enter__.return_value
        # Make copy_from raise an error to trigger fallback
        cursor.copy_from.side_effect = Exception("COPY failed")
        cursor.fetchone.side_effect = [(0,), (1,)]
        cursor.rowcount = 1

        actual_inserted = insert_all_data(mock_conn, df)

        assert actual_inserted == 1
        assert mock_execute_values.called

    @patch("stacksats.export_weights.get_current_btc_price")
    @patch("time.time", side_effect=[100, 101, 102, 103])
    def test_update_today_weights(self, mock_time, mock_get_price, mock_conn):
        """Test update_today_weights executes bulk update SQL."""
        mock_get_price.return_value = 60000.0

        df = pd.DataFrame({
            "day_index": [0, 1],
            "start_date": ["2024-01-01", "2024-01-01"],
            "end_date": ["2025-01-01", "2025-01-01"],
            "date": ["2024-01-01", "2024-01-02"],
            "weight": [1.1, 1.2],
            "price_usd": [50000.0, 50000.0]
        })

        today_str = "2024-01-01"
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        cursor.rowcount = 1

        total_updated = update_today_weights(mock_conn, df, today_str)

        assert total_updated == 1
        assert cursor.execute.called
        # Check if UPDATE SQL was used
        update_calls = [c[0][0] for c in cursor.execute.call_args_list if "UPDATE" in c[0][0]]
        assert len(update_calls) > 0
        assert "60000" in str(cursor.execute.call_args_list)
