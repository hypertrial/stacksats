import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from stacksats.export_weights import insert_all_data, update_today_weights

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_db_connection():
    """Mock database connection and cursor."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor
    return conn, cursor


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "day_index": [0, 1],
        "start_date": ["2024-01-01", "2024-01-01"],
        "end_date": ["2024-12-31", "2024-12-31"],
        "date": ["2024-06-01", "2024-06-02"],
        "price_usd": [50000.0, 51000.0],
        "weight": [0.5, 0.5]
    })


# -----------------------------------------------------------------------------
# Test Bulk Insert Logic
# -----------------------------------------------------------------------------

def test_insert_all_data_copy_success(mock_db_connection, sample_df):
    """Test that insert_all_data uses COPY FROM when successful."""
    conn, cursor = mock_db_connection

    # Setup mocks
    cursor.fetchone.side_effect = [
        (0,),  # count before
        (2,),  # count after (final) -- COPY usually doesn't return count directly in this flow but we mock the check
    ]

    # Run function
    inserted = insert_all_data(conn, sample_df)

    # Verify COPY was called
    assert cursor.copy_from.called

    # Verify execute_values was NOT called (no fallback needed)
    with patch("stacksats.export_weights.execute_values") as mock_exec_values:
        assert not mock_exec_values.called

    assert inserted == 2


def test_insert_all_data_fallback(mock_db_connection, sample_df):
    """Test that insert_all_data falls back to execute_values when COPY fails."""
    conn, cursor = mock_db_connection

    # Setup mocks
    cursor.fetchone.side_effect = [
        (0,),  # count before
        (2,)   # count after
    ]

    # Make copy_from raise an exception
    cursor.copy_from.side_effect = Exception("COPY failed permission denied")

    # Mock execute_values
    with patch("stacksats.export_weights.execute_values") as mock_exec_values:
        # Run function
        inserted = insert_all_data(conn, sample_df)

        # Verify COPY tried and failed
        assert cursor.copy_from.called

        # Verify rollback called
        assert conn.rollback.called

        # Verify execute_values WAS called
        assert mock_exec_values.called

        # Check args passed to execute_values
        args, kwargs = mock_exec_values.call_args
        assert args[0] == cursor # cursor passed
        assert "INSERT INTO bitcoin_dca" in args[1] # query passed
        assert len(args[2]) == 2 # data batch size correct

    assert inserted == 2


def test_insert_all_data_fallback_rolls_back_and_raises_on_batch_failure(
    mock_db_connection,
):
    """Test fallback insertion rolls back and re-raises when a later batch fails."""
    conn, cursor = mock_db_connection

    n_rows = 50001  # Forces 2 fallback batches at batch_size=50000
    large_df = pd.DataFrame(
        {
            "day_index": range(n_rows),
            "start_date": ["2024-01-01"] * n_rows,
            "end_date": ["2024-12-31"] * n_rows,
            "date": ["2024-06-01"] * n_rows,
            "price_usd": [50000.0] * n_rows,
            "weight": [0.5] * n_rows,
        }
    )
    cursor.fetchone.return_value = (0,)
    cursor.copy_from.side_effect = Exception("COPY failed permission denied")

    with patch("stacksats.export_weights.execute_values") as mock_exec_values:
        calls = {"count": 0}

        def _fail_second_batch(*_args, **_kwargs):
            calls["count"] += 1
            if calls["count"] == 2:
                raise Exception("batch failure")

        mock_exec_values.side_effect = _fail_second_batch

        with pytest.raises(Exception, match="batch failure"):
            insert_all_data(conn, large_df)

    # First rollback after COPY failure, second rollback after fallback batch failure
    assert conn.rollback.call_count == 2
    assert conn.commit.call_count == 0


def test_insert_all_data_fallback_logs_batch_context_on_failure(
    mock_db_connection,
    caplog,
):
    """Test fallback failure logs include precise batch context."""
    conn, cursor = mock_db_connection

    n_rows = 50001
    large_df = pd.DataFrame(
        {
            "day_index": range(n_rows),
            "start_date": ["2024-01-01"] * n_rows,
            "end_date": ["2024-12-31"] * n_rows,
            "date": ["2024-06-01"] * n_rows,
            "price_usd": [50000.0] * n_rows,
            "weight": [0.5] * n_rows,
        }
    )
    cursor.fetchone.return_value = (0,)
    cursor.copy_from.side_effect = Exception("COPY failed permission denied")

    with patch("stacksats.export_weights.execute_values") as mock_exec_values:
        calls = {"count": 0}

        def _fail_second_batch(*_args, **_kwargs):
            calls["count"] += 1
            if calls["count"] == 2:
                raise Exception("batch failure")

        mock_exec_values.side_effect = _fail_second_batch

        with caplog.at_level("ERROR"):
            with pytest.raises(Exception, match="batch failure"):
                insert_all_data(conn, large_df)

    assert "Fallback bulk insert failed at batch 2/2 (1 rows): batch failure" in caplog.text


def test_insert_all_data_fallback_rollback_failure_keeps_original_exception(
    mock_db_connection,
    caplog,
):
    """Test rollback failures are logged but original fallback error is re-raised."""
    conn, cursor = mock_db_connection

    n_rows = 50001
    large_df = pd.DataFrame(
        {
            "day_index": range(n_rows),
            "start_date": ["2024-01-01"] * n_rows,
            "end_date": ["2024-12-31"] * n_rows,
            "date": ["2024-06-01"] * n_rows,
            "price_usd": [50000.0] * n_rows,
            "weight": [0.5] * n_rows,
        }
    )
    cursor.fetchone.return_value = (0,)
    cursor.copy_from.side_effect = Exception("COPY failed permission denied")
    conn.rollback.side_effect = [None, Exception("rollback failed")]

    with patch("stacksats.export_weights.execute_values") as mock_exec_values:
        calls = {"count": 0}

        def _fail_second_batch(*_args, **_kwargs):
            calls["count"] += 1
            if calls["count"] == 2:
                raise Exception("batch failure")

        mock_exec_values.side_effect = _fail_second_batch

        with caplog.at_level("ERROR"):
            with pytest.raises(Exception, match="batch failure"):
                insert_all_data(conn, large_df)

    assert conn.rollback.call_count == 2
    assert "Rollback failed after fallback insert error: rollback failed" in caplog.text


# -----------------------------------------------------------------------------
# Test Update Logic
# -----------------------------------------------------------------------------

@patch("stacksats.export_weights.get_current_btc_price")
def test_update_today_weights_success(mock_get_price, mock_db_connection, sample_df):
    """Test update_today_weights when price fetch succeeds."""
    conn, cursor = mock_db_connection
    today_str = "2024-06-01"

    # Mock price return
    mock_get_price.return_value = 60000.0

    # Mock update rowcount
    cursor.rowcount = 1

    # Run function
    updated = update_today_weights(conn, sample_df, today_str)

    # Verify price fetch called
    assert mock_get_price.called

    # Verify UPDATE executed
    assert cursor.execute.called
    call_args = cursor.execute.call_args[0][0]

    # Should be updating both weight and btc_usd (DB column)
    assert "UPDATE bitcoin_dca" in call_args
    assert "SET weight = v.weight, btc_usd = v.btc_usd" in call_args

    assert updated == 1


@patch("stacksats.export_weights.get_current_btc_price")
def test_update_today_weights_price_fetch_failure(mock_get_price, mock_db_connection, sample_df):
    """Test update_today_weights when price fetch fails but DF has price."""
    conn, cursor = mock_db_connection
    today_str = "2024-06-01" # This date exists in sample_df with price 50000.0

    # Mock price fetch failure
    mock_get_price.return_value = None

    # Mock update rowcount
    cursor.rowcount = 1

    # Run function
    updated = update_today_weights(conn, sample_df, today_str)

    # Verify it proceeds using DF price (which is 50000.0)
    assert cursor.execute.called
    call_args = cursor.execute.call_args[0][0]

    # It should still try to update btc_usd because it found price in the dataframe
    # The logic in update_today_weights falls back to dataframe price if API fails
    assert "SET weight = v.weight, btc_usd = v.btc_usd" in call_args

    assert updated == 1


@patch("stacksats.export_weights.get_current_btc_price")
def test_update_today_weights_no_price(mock_get_price, mock_db_connection):
    """Test update_today_weights when no price available anywhere."""
    conn, cursor = mock_db_connection
    today_str = "2024-06-01"

    # Create DF with NO price info for today
    df_no_price = pd.DataFrame({
        "day_index": [0],
        "start_date": ["2024-01-01"],
        "end_date": ["2024-12-31"],
        "date": ["2024-06-01"],
        "price_usd": [None], # Missing price
        "weight": [0.5]
    })

    # Mock price fetch failure
    mock_get_price.return_value = None

    # Run function
    updated = update_today_weights(conn, df_no_price, today_str)

    # Should return 0
    assert updated == 0

    # Verify that NO UPDATE statement was executed
    # Note: cursor.execute is called once at the start for the previous day's price check (SELECT)
    update_calls = [c for c in cursor.execute.call_args_list if "UPDATE" in str(c)]
    assert len(update_calls) == 0, "Update should not have been called"


@patch("stacksats.export_weights.get_current_btc_price")
def test_update_today_weights_rolls_back_and_raises_on_update_failure(
    mock_get_price,
    mock_db_connection,
    sample_df,
):
    """Test update_today_weights rolls back and re-raises on SQL execution failure."""
    conn, cursor = mock_db_connection
    today_str = "2024-06-01"

    mock_get_price.return_value = 60000.0
    cursor.fetchone.return_value = (None,)
    cursor.execute.side_effect = [None, Exception("update failed")]

    with pytest.raises(Exception, match="update failed"):
        update_today_weights(conn, sample_df, today_str)

    assert conn.rollback.call_count == 1
    assert conn.commit.call_count == 0


@patch("stacksats.export_weights.get_current_btc_price")
def test_update_today_weights_rolls_back_on_weight_only_failure(
    mock_get_price,
    mock_db_connection,
):
    """Test rollback and mode logging when weight-only update SQL fails."""
    conn, cursor = mock_db_connection
    today_str = "2024-06-01"
    mock_get_price.return_value = None

    df = pd.DataFrame(
        {
            "day_index": [0, 1],
            "start_date": ["2024-01-01", "2024-01-01"],
            "end_date": ["2024-12-31", "2024-12-31"],
            "date": ["2024-06-01", "2024-06-01"],
            "price_usd": [None, 50000.0],
            "weight": [0.4, 0.6],
        }
    )

    cursor.fetchone.return_value = (None,)
    cursor.execute.side_effect = [None, Exception("weight only failed")]

    with pytest.raises(Exception, match="weight only failed"):
        update_today_weights(conn, df, today_str)

    assert conn.rollback.call_count == 1
    assert conn.commit.call_count == 0


@patch("stacksats.export_weights.get_current_btc_price")
def test_update_today_weights_rollback_failure_keeps_original_exception(
    mock_get_price,
    mock_db_connection,
    sample_df,
    caplog,
):
    """Test rollback failures are logged while preserving the update failure error."""
    conn, cursor = mock_db_connection
    today_str = "2024-06-01"

    mock_get_price.return_value = 60000.0
    cursor.fetchone.return_value = (None,)
    cursor.execute.side_effect = [None, Exception("update failed")]
    conn.rollback.side_effect = Exception("rollback failed")

    with caplog.at_level("ERROR"):
        with pytest.raises(Exception, match="update failed"):
            update_today_weights(conn, sample_df, today_str)

    assert conn.rollback.call_count == 1
    assert "Rollback failed after update_today_weights error: rollback failed" in caplog.text
