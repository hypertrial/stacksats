from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from stacksats.export_weights import (
    get_db_connection,
    load_locked_weights_for_window,
    process_start_date_batch,
    update_today_weights,
)
from stacksats.strategy_types import BaseStrategy, DayState


def _mock_conn_with_rows(rows):
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = rows
    return conn, cursor


def _sample_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    features_df = pd.DataFrame(
        {
            "PriceUSD_coinmetrics": [100.0, 101.0],
            "mvrv_zscore": [0.0, 0.1],
        },
        index=idx,
    )
    btc_df = pd.DataFrame({"PriceUSD_coinmetrics": [100.0, 101.0]}, index=idx)
    return features_df, btc_df, idx.min(), idx.max()


def test_load_locked_weights_for_window_returns_none_when_lock_end_before_start() -> None:
    conn, _ = _mock_conn_with_rows([])

    result = load_locked_weights_for_window(
        conn,
        start_date="2024-01-10",
        end_date="2024-12-31",
        lock_end_date="2024-01-01",
    )

    assert result is None


def test_load_locked_weights_for_window_returns_none_when_no_rows() -> None:
    conn, _ = _mock_conn_with_rows([])

    result = load_locked_weights_for_window(
        conn,
        start_date="2024-01-01",
        end_date="2024-01-03",
        lock_end_date="2024-01-03",
    )

    assert result is None


def test_load_locked_weights_for_window_returns_contiguous_prefix() -> None:
    conn, _ = _mock_conn_with_rows(
        [
            ("2024-01-01", 0.1),
            ("2024-01-02", 0.2),
            ("2024-01-03", 0.3),
        ]
    )

    result = load_locked_weights_for_window(
        conn,
        start_date="2024-01-01",
        end_date="2024-01-10",
        lock_end_date="2024-01-03",
    )

    np.testing.assert_allclose(result, np.array([0.1, 0.2, 0.3], dtype=float))


def test_load_locked_weights_for_window_rejects_non_contiguous_history() -> None:
    conn, _ = _mock_conn_with_rows(
        [
            ("2024-01-01", 0.1),
            ("2024-01-03", 0.3),
        ]
    )

    with pytest.raises(ValueError, match="not a contiguous prefix"):
        load_locked_weights_for_window(
            conn,
            start_date="2024-01-01",
            end_date="2024-01-03",
            lock_end_date="2024-01-03",
        )


def test_get_db_connection_requires_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    import stacksats.export_weights as export_weights

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(export_weights, "psycopg2", MagicMock())

    with pytest.raises(ValueError, match="DATABASE_URL environment variable is not set"):
        get_db_connection()


def test_get_db_connection_requires_deploy_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    import stacksats.export_weights as export_weights

    monkeypatch.setattr(export_weights, "psycopg2", None)

    with pytest.raises(ImportError, match="Install deploy extras"):
        get_db_connection()


class _StrategyWithHook(BaseStrategy):
    strategy_id = "test-hook"

    def propose_weight(self, state: DayState) -> float:
        return state.uniform_weight


def test_process_start_date_batch_falls_back_when_strategy_returns_empty(mocker) -> None:
    features_df, btc_df, start_date, end_date = _sample_frames()
    strategy = _StrategyWithHook()
    strategy.compute_weights = MagicMock(return_value=pd.Series(dtype=float))

    fallback = pd.Series([0.4, 0.6], index=pd.date_range(start_date, end_date, freq="D"))
    mocked_fallback = mocker.patch(
        "stacksats.export_weights.compute_window_weights",
        return_value=fallback,
    )

    result = process_start_date_batch(
        start_date,
        [end_date],
        features_df,
        btc_df,
        current_date=end_date,
        btc_price_col="PriceUSD_coinmetrics",
        strategy=strategy,
        enforce_span_contract=False,
    )

    assert mocked_fallback.called
    assert result["weight"].tolist() == [0.4, 0.6]


def test_process_start_date_batch_reindexes_partial_strategy_output() -> None:
    features_df, btc_df, start_date, end_date = _sample_frames()
    strategy = _StrategyWithHook()
    strategy.compute_weights = MagicMock(
        return_value=pd.Series([0.7], index=[start_date], dtype=float)
    )

    result = process_start_date_batch(
        start_date,
        [end_date],
        features_df,
        btc_df,
        current_date=end_date,
        btc_price_col="PriceUSD_coinmetrics",
        strategy=strategy,
        enforce_span_contract=False,
    )

    assert result["weight"].tolist() == [0.7, 0.0]


def test_update_today_weights_returns_zero_when_today_absent() -> None:
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = (None,)

    df = pd.DataFrame(
        {
            "id": [1],
            "start_date": ["2024-01-01"],
            "end_date": ["2024-12-31"],
            "DCA_date": ["2024-01-02"],
            "btc_usd": [50000.0],
            "weight": [1.0],
        }
    )

    updated = update_today_weights(conn, df, today_str="2024-01-01")

    assert updated == 0
    update_calls = [c for c in cursor.execute.call_args_list if "UPDATE" in str(c)]
    assert not update_calls


def test_update_today_weights_uses_weight_only_sql_when_price_stays_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = (None,)
    cursor.rowcount = 2

    monkeypatch.setattr("stacksats.export_weights.get_current_btc_price", lambda previous_price=None: None)

    df = pd.DataFrame(
        {
            "id": [1, 2],
            "start_date": ["2024-01-01", "2024-01-01"],
            "end_date": ["2024-12-31", "2024-12-31"],
            "DCA_date": ["2024-01-01", "2024-01-01"],
            "btc_usd": [None, 50000.0],
            "weight": [0.4, 0.6],
        }
    )

    updated = update_today_weights(conn, df, today_str="2024-01-01")

    assert updated == 2
    sql_text = "\n".join(call.args[0] for call in cursor.execute.call_args_list if call.args)
    assert "SET weight = v.weight" in sql_text
    assert "btc_usd = v.btc_usd" not in sql_text
