from __future__ import annotations

import pickle
from unittest.mock import MagicMock, patch

import pandas as pd

from stacksats.modal_app import daily_export, daily_export_retry, process_start_date_batch_modal


def _pickle_df(df: pd.DataFrame) -> bytes:
    return pickle.dumps(df)


def test_process_start_date_batch_modal_supports_7_args_with_locked_dict() -> None:
    features_df = pd.DataFrame({"x": [1.0]})
    btc_df = pd.DataFrame({"PriceUSD_coinmetrics": [100.0]})
    expected = pd.DataFrame({"id": [1]})

    args = (
        "2024-01-01",
        ["2024-01-02"],
        "2024-01-02",
        "PriceUSD_coinmetrics",
        _pickle_df(features_df),
        _pickle_df(btc_df),
        {"2024-01-02": [0.5]},
    )

    with patch("stacksats.export_weights.process_start_date_batch", return_value=expected) as mock_batch, patch(
        "stacksats.loader.load_strategy"
    ) as mock_load_strategy:
        result = process_start_date_batch_modal.get_raw_f()(args)

    assert result.equals(expected)
    assert mock_batch.called
    assert mock_load_strategy.call_count == 0


def test_process_start_date_batch_modal_supports_7_args_with_strategy_spec() -> None:
    features_df = pd.DataFrame({"x": [1.0]})
    btc_df = pd.DataFrame({"PriceUSD_coinmetrics": [100.0]})
    expected = pd.DataFrame({"id": [1]})
    strategy_obj = object()

    args = (
        "2024-01-01",
        ["2024-01-02"],
        "2024-01-02",
        "PriceUSD_coinmetrics",
        _pickle_df(features_df),
        _pickle_df(btc_df),
        "my_strategy.py:MyStrategy",
    )

    with patch("stacksats.export_weights.process_start_date_batch", return_value=expected) as mock_batch, patch(
        "stacksats.loader.load_strategy", return_value=strategy_obj
    ) as mock_load_strategy:
        result = process_start_date_batch_modal.get_raw_f()(args)

    assert result.equals(expected)
    assert mock_load_strategy.call_count == 1
    assert mock_batch.call_args.kwargs["strategy"] is strategy_obj


def test_process_start_date_batch_modal_rejects_invalid_tuple_length() -> None:
    try:
        process_start_date_batch_modal.get_raw_f()(("too", "short"))
    except ValueError as exc:
        assert "expected 6, 7, or 8 args" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid args length")


@patch("stacksats.modal_app.run_export.remote")
@patch("stacksats.export_weights.get_db_connection")
@patch("stacksats.export_weights.create_table_if_not_exists")
@patch("stacksats.export_weights.table_is_empty")
@patch("stacksats.export_weights.insert_all_data")
def test_daily_export_inserts_all_rows_when_table_empty(
    mock_insert, mock_table_empty, mock_create, mock_get_db, mock_run_remote
):
    mock_conn = MagicMock()
    mock_get_db.return_value = mock_conn
    mock_table_empty.return_value = True
    mock_insert.return_value = 7
    mock_run_remote.return_value = (
        pd.DataFrame({"id": [1]}),
        {
            "export_date": "2024-01-01",
            "date_ranges": 1,
            "range_start": "2024-01-01",
            "range_end": "2024-12-31",
        },
    )

    result = daily_export.get_raw_f()()

    assert result["status"] == "success"
    assert result["rows_affected"] == 7
    assert mock_insert.call_count == 1
    assert mock_conn.close.called


@patch("stacksats.export_weights.get_db_connection")
@patch("stacksats.export_weights.create_table_if_not_exists")
@patch("stacksats.export_weights.table_is_empty")
@patch("stacksats.modal_app.run_export.remote")
def test_daily_export_retry_skips_when_table_empty(
    mock_run_remote, mock_table_empty, mock_create, mock_get_db
):
    mock_conn = MagicMock()
    mock_get_db.return_value = mock_conn
    mock_table_empty.return_value = True

    result = daily_export_retry.get_raw_f()()

    assert result["status"] == "skipped"
    assert result["reason"] == "table_empty"
    assert mock_run_remote.call_count == 0
    assert mock_conn.close.called


@patch("stacksats.export_weights.today_data_exists")
@patch("stacksats.export_weights.get_db_connection")
@patch("stacksats.modal_app.run_export.remote")
@patch("stacksats.export_weights.table_is_empty")
@patch("stacksats.export_weights.create_table_if_not_exists")
def test_daily_export_retry_returns_partial_failure_when_export_has_no_today_rows(
    mock_create, mock_table_empty, mock_run_remote, mock_get_db, mock_today_exists
):
    mock_conn = MagicMock()
    mock_get_db.return_value = mock_conn
    mock_table_empty.return_value = False
    mock_today_exists.return_value = False

    with patch("pandas.Timestamp.now", return_value=pd.Timestamp("2024-01-01")):
        mock_run_remote.return_value = (
            pd.DataFrame({"DCA_date": ["2023-12-31"]}),
            {"export_date": "2024-01-01"},
        )
        result = daily_export_retry.get_raw_f()()

    assert result["status"] == "partial_failure"
    assert result["reason"] == "no_data_in_export"


@patch("stacksats.export_weights.update_today_weights")
@patch("stacksats.export_weights.today_data_exists")
@patch("stacksats.export_weights.get_db_connection")
@patch("stacksats.modal_app.run_export.remote")
@patch("stacksats.export_weights.table_is_empty")
@patch("stacksats.export_weights.create_table_if_not_exists")
def test_daily_export_retry_returns_partial_failure_when_no_rows_updated(
    mock_create,
    mock_table_empty,
    mock_run_remote,
    mock_get_db,
    mock_today_exists,
    mock_update,
):
    mock_conn = MagicMock()
    mock_get_db.return_value = mock_conn
    mock_table_empty.return_value = False
    mock_today_exists.return_value = False
    mock_update.return_value = 0

    with patch("pandas.Timestamp.now", return_value=pd.Timestamp("2024-01-01")):
        mock_run_remote.return_value = (
            pd.DataFrame({"DCA_date": ["2024-01-01"]}),
            {"export_date": "2024-01-01"},
        )
        result = daily_export_retry.get_raw_f()()

    assert result["status"] == "partial_failure"
    assert result["reason"] == "no_rows_updated"


@patch("stacksats.export_weights.today_data_exists")
@patch("stacksats.export_weights.get_db_connection")
@patch("stacksats.export_weights.table_is_empty")
@patch("stacksats.export_weights.create_table_if_not_exists")
def test_daily_export_retry_returns_error_on_exception(
    mock_create, mock_table_empty, mock_get_db, mock_today_exists
):
    mock_conn = MagicMock()
    mock_get_db.return_value = mock_conn
    mock_table_empty.return_value = False
    mock_today_exists.side_effect = RuntimeError("boom")

    with patch("pandas.Timestamp.now", return_value=pd.Timestamp("2024-01-01")):
        result = daily_export_retry.get_raw_f()()

    assert result["status"] == "error"
    assert "boom" in result["error"]
    assert mock_conn.close.called
