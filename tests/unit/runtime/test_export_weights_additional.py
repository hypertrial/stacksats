from __future__ import annotations

import datetime as dt
import builtins
import importlib
from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest

from stacksats.export_weights import (
    get_db_connection,
    get_current_btc_price,
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


def _sample_frames() -> tuple[pl.DataFrame, pl.DataFrame, dt.datetime, dt.datetime]:
    dates = [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)]
    features_df = pl.DataFrame(
        {
            "date": dates,
            "price_usd": [100.0, 101.0],
            "mvrv_zscore": [0.0, 0.1],
        }
    )
    btc_df = pl.DataFrame({"date": dates, "price_usd": [100.0, 101.0]})
    return features_df, btc_df, dates[0], dates[-1]


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


def test_export_weights_import_falls_back_when_dotenv_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stacksats.export_weights as export_weights_module

    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "dotenv":
            raise ImportError("dotenv unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    reloaded = importlib.reload(export_weights_module)

    assert reloaded is export_weights_module


def test_get_current_btc_price_logs_success_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("stacksats.export_weights.fetch_btc_price_robust", lambda previous_price=None: 50000.0)

    price = get_current_btc_price(previous_price=49000.0)

    assert price == 50000.0


class _StrategyWithHook(BaseStrategy):
    strategy_id = "test-hook"

    def propose_weight(self, state: DayState) -> float:
        return state.uniform_weight


def test_process_start_date_batch_falls_back_when_strategy_returns_empty(mocker) -> None:
    features_df, btc_df, start_date, end_date = _sample_frames()
    strategy = _StrategyWithHook()
    strategy.compute_weights = MagicMock(
        return_value=pl.DataFrame(schema={"date": pl.Datetime("us"), "weight": pl.Float64})
    )

    fallback = pl.DataFrame(
        {
            "date": [start_date, end_date],
            "weight": [0.4, 0.6],
        }
    )
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
        btc_price_col="price_usd",
        strategy=strategy,
        enforce_span_contract=False,
    )

    assert mocked_fallback.called
    assert result["weight"].to_list() == [0.4, 0.6]


def test_process_start_date_batch_reindexes_partial_strategy_output() -> None:
    features_df, btc_df, start_date, end_date = _sample_frames()
    strategy = _StrategyWithHook()
    strategy.compute_weights = MagicMock(
        return_value=pl.DataFrame({"date": [start_date], "weight": [0.7]})
    )

    result = process_start_date_batch(
        start_date,
        [end_date],
        features_df,
        btc_df,
        current_date=end_date,
        btc_price_col="price_usd",
        strategy=strategy,
        enforce_span_contract=False,
    )

    assert result["weight"].to_list() == [0.7, 0.0]


def test_process_start_date_batch_does_not_expose_rows_after_end_date() -> None:
    dates = [dt.datetime(2024, 1, 1) + dt.timedelta(days=offset) for offset in range(4)]
    features_df = pl.DataFrame(
        {
            "date": dates,
            "price_usd": [100.0, 101.0, 102.0, 103.0],
            "mvrv_zscore": [0.0, 0.1, 0.2, 0.3],
        }
    )
    btc_df = pl.DataFrame({"date": dates[:2], "price_usd": [100.0, 101.0]})
    strategy = _StrategyWithHook()
    captured_max: list[dt.datetime] = []

    def _compute_weights(ctx):
        captured_max.append(ctx.features_df["date"].max())
        return pl.DataFrame({"date": dates[:2], "weight": [0.5, 0.5]})

    strategy.compute_weights = MagicMock(side_effect=_compute_weights)

    process_start_date_batch(
        dates[0],
        [dates[1]],
        features_df,
        btc_df,
        current_date=dates[-1],
        btc_price_col="price_usd",
        strategy=strategy,
        enforce_span_contract=False,
    )

    assert captured_max == [dates[1]]


def test_update_today_weights_returns_zero_when_today_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = (None,)
    calls = {"count": 0}

    def _fake_get_current_btc_price(previous_price=None):
        del previous_price
        calls["count"] += 1
        return 50000.0

    monkeypatch.setattr("stacksats.export_weights.get_current_btc_price", _fake_get_current_btc_price)

    df = pl.DataFrame(
        {
            "day_index": [0],
            "start_date": ["2024-01-01"],
            "end_date": ["2024-12-31"],
            "date": ["2024-01-02"],
            "price_usd": [50000.0],
            "weight": [1.0],
        }
    )

    updated = update_today_weights(conn, df, today_str="2024-01-01")

    assert updated == 0
    assert calls["count"] == 0
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

    df = pl.DataFrame(
        {
            "day_index": [0, 1],
            "start_date": ["2024-01-01", "2024-01-01"],
            "end_date": ["2024-12-31", "2024-12-31"],
            "date": ["2024-01-01", "2024-01-01"],
            "price_usd": [None, 50000.0],
            "weight": [0.4, 0.6],
        }
    )

    updated = update_today_weights(conn, df, today_str="2024-01-01")

    assert updated == 2
    sql_text = "\n".join(call.args[0] for call in cursor.execute.call_args_list if call.args)
    assert "SET weight = v.weight" in sql_text
    assert "btc_usd = v.btc_usd" not in sql_text


def test_get_current_btc_price_returns_none_when_all_sources_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "stacksats.export_weights.fetch_btc_price_robust",
        lambda previous_price=None: None,
    )

    price = get_current_btc_price(previous_price=49000.0)

    assert price is None


def test_update_today_weights_raises_when_required_columns_missing() -> None:
    conn = MagicMock()
    df_missing_price = pl.DataFrame(
        {
            "day_index": [0],
            "start_date": ["2024-01-01"],
            "end_date": ["2024-12-31"],
            "date": ["2024-01-01"],
            "weight": [1.0],
        }
    )

    with pytest.raises(ValueError, match="requires canonical columns"):
        update_today_weights(conn, df_missing_price, today_str="2024-01-01")


def test_update_today_weights_returns_zero_when_no_api_price_and_today_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = (None,)

    monkeypatch.setattr(
        "stacksats.export_weights.get_current_btc_price",
        lambda previous_price=None: None,
    )

    df = pl.DataFrame(
        {
            "day_index": [0],
            "start_date": ["2024-01-01"],
            "end_date": ["2024-12-31"],
            "date": ["2024-01-02"],
            "price_usd": [50000.0],
            "weight": [1.0],
        }
    )

    updated = update_today_weights(conn, df, today_str="2024-01-01")

    assert updated == 0
