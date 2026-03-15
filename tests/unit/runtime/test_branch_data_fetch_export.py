from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock

import polars as pl
import pytest

import stacksats.export_weights as export_weights
from stacksats.btc_price_fetcher import fetch_btc_price_robust
from stacksats.export_weights import get_db_connection, process_start_date_batch, update_today_weights
from stacksats.strategy_types import BaseStrategy


def test_fetch_btc_price_robust_handles_keyerror_source_then_fallback() -> None:
    def bad_source() -> float:
        raise KeyError("usd")

    def good_source() -> float:
        return 42000.0

    price = fetch_btc_price_robust(
        sources=[(bad_source, "BadSource"), (good_source, "GoodSource")]
    )
    assert price == 42000.0


def test_fetch_btc_price_robust_handles_unexpected_source_exception() -> None:
    def broken_source() -> float:
        raise RuntimeError("boom")

    price = fetch_btc_price_robust(sources=[(broken_source, "BrokenSource")])
    assert price is None


def _sample_frames() -> tuple[dt.datetime, dt.datetime, pl.DataFrame, pl.DataFrame]:
    dates = [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)]
    start_date, end_date = dates[0], dates[-1]
    features_df = pl.DataFrame(
        {
            "date": dates,
            "price_usd": [100.0, 101.0],
            "mvrv_zscore": [0.0, 0.1],
        }
    )
    btc_df = pl.DataFrame({"date": dates, "price_usd": [100.0, 101.0]})
    return start_date, end_date, features_df, btc_df


class _NoHookStrategy(BaseStrategy):
    strategy_id = "no-hook"


def test_process_start_date_batch_rejects_non_strategy_type() -> None:
    start_date, end_date, features_df, btc_df = _sample_frames()
    with pytest.raises(TypeError, match="strategy must subclass BaseStrategy"):
        process_start_date_batch(
            start_date=start_date,
            end_dates=[end_date],
            features_df=features_df,
            btc_df=btc_df,
            current_date=end_date,
            btc_price_col="price_usd",
            strategy=object(),
            enforce_span_contract=False,
        )


def test_process_start_date_batch_rejects_no_hook_strategy() -> None:
    start_date, end_date, features_df, btc_df = _sample_frames()
    with pytest.raises(TypeError, match="must implement propose_weight"):
        process_start_date_batch(
            start_date=start_date,
            end_dates=[end_date],
            features_df=features_df,
            btc_df=btc_df,
            current_date=end_date,
            btc_price_col="price_usd",
            strategy=_NoHookStrategy(),
            enforce_span_contract=False,
        )


def test_get_db_connection_uses_database_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_psycopg2 = MagicMock()
    fake_conn = object()
    mock_psycopg2.connect.return_value = fake_conn
    monkeypatch.setenv("DATABASE_URL", "postgres://unit-test")
    monkeypatch.setattr(export_weights, "psycopg2", mock_psycopg2)

    conn = get_db_connection()

    assert conn is fake_conn
    mock_psycopg2.connect.assert_called_once_with("postgres://unit-test")


def test_update_today_weights_continues_when_previous_price_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.rowcount = 1
    cursor.fetchone.return_value = None

    def _execute(sql, params=None):
        del params
        if "SELECT btc_usd FROM bitcoin_dca" in sql:
            raise RuntimeError("lookup failed")
        return None

    cursor.execute.side_effect = _execute

    observed: dict[str, float | None] = {}

    def _fake_get_current_btc_price(previous_price=None):
        observed["previous_price"] = previous_price
        return 61000.0

    monkeypatch.setattr(
        "stacksats.export_weights.get_current_btc_price",
        _fake_get_current_btc_price,
    )

    df = pl.DataFrame(
        {
            "day_index": [0],
            "start_date": ["2024-01-01"],
            "end_date": ["2024-12-31"],
            "date": ["2024-01-01"],
            "price_usd": [50000.0],
            "weight": [1.0],
        }
    )

    updated = update_today_weights(conn, df, today_str="2024-01-01")

    assert updated == 1
    assert observed["previous_price"] is None
