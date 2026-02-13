from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

import stacksats.export_weights as export_weights
from stacksats.btc_api.coinmetrics_btc_csv import fetch_coinmetrics_btc_csv
from stacksats.btc_price_fetcher import _load_coinmetrics_data, fetch_btc_price_robust
from stacksats.data_btc import BTCDataProvider, _is_cache_usable
from stacksats.export_weights import get_db_connection, process_start_date_batch, update_today_weights
from stacksats.strategy_types import BaseStrategy


def _csv_bytes(rows: list[dict]) -> bytes:
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def _mock_response(content: bytes) -> MagicMock:
    response = MagicMock()
    response.content = content
    response.raise_for_status.return_value = None
    return response


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


def test_load_coinmetrics_data_requires_price_column(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _mock_response(b"time,CapMVRVCur\n2024-01-01,2.0\n")
    monkeypatch.setattr(
        "stacksats.btc_price_fetcher.requests.get",
        lambda *_args, **_kwargs: response,
    )

    _load_coinmetrics_data.cache_clear()
    try:
        with pytest.raises(ValueError, match="PriceUSD column not found"):
            _load_coinmetrics_data()
    finally:
        _load_coinmetrics_data.cache_clear()


def test_fetch_coinmetrics_csv_wraps_parser_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _mock_response(b"time,PriceUSD\n2024-01-01,40000.0\n")
    monkeypatch.setattr(
        "stacksats.btc_api.coinmetrics_btc_csv.requests.get",
        lambda *_args, **_kwargs: response,
    )

    def _explode(*_args, **_kwargs):
        raise RuntimeError("parser failed")

    monkeypatch.setattr("stacksats.btc_api.coinmetrics_btc_csv.pd.read_csv", _explode)

    with pytest.raises(ValueError, match="Invalid CSV data"):
        fetch_coinmetrics_btc_csv()


def test_is_cache_usable_false_for_empty_csv() -> None:
    empty_csv = b"time,PriceUSD\n"
    assert (
        _is_cache_usable(
            empty_csv,
            backtest_start=pd.Timestamp("2024-01-01"),
            today=pd.Timestamp("2024-01-10"),
        )
        is False
    )


def test_is_cache_usable_false_when_all_timestamps_invalid() -> None:
    invalid_dates_csv = b"time,PriceUSD\nnot-a-date,41000.0\nstill-bad,42000.0\n"
    assert (
        _is_cache_usable(
            invalid_dates_csv,
            backtest_start=pd.Timestamp("2024-01-01"),
            today=pd.Timestamp("2024-01-10"),
        )
        is False
    )


def test_data_provider_fetches_and_writes_cache_when_file_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = pd.Timestamp("2024-01-02")
    remote_bytes = _csv_bytes(
        [
            {"time": "2024-01-01", "PriceUSD": 40000.0, "CapMVRVCur": 2.0},
            {"time": "2024-01-02", "PriceUSD": 40100.0, "CapMVRVCur": 2.1},
        ]
    )
    mocked_get = MagicMock(return_value=_mock_response(remote_bytes))
    monkeypatch.setattr("stacksats.data_btc.requests.get", mocked_get)

    provider = BTCDataProvider(cache_dir=str(tmp_path), max_age_hours=24, clock=lambda: now)
    df = provider.load(backtest_start="2024-01-01")

    cache_file = tmp_path / "coinmetrics_btc.csv"
    assert cache_file.exists()
    assert cache_file.read_bytes() == remote_bytes
    assert mocked_get.call_count == 1
    assert float(df.loc[pd.Timestamp("2024-01-02"), "PriceUSD_coinmetrics"]) == 40100.0


def test_data_provider_raises_when_price_column_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    now = pd.Timestamp("2024-01-02")
    remote_bytes = b"time,NotPrice\n2024-01-01,1\n2024-01-02,2\n"
    monkeypatch.setattr(
        "stacksats.data_btc.requests.get",
        lambda *_args, **_kwargs: _mock_response(remote_bytes),
    )

    provider = BTCDataProvider(cache_dir=None, clock=lambda: now)
    with pytest.raises(ValueError, match="PriceUSD column not found"):
        provider.load(backtest_start="2024-01-01")


def _sample_frames() -> tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    start_date, end_date = idx.min(), idx.max()
    features_df = pd.DataFrame(
        {
            "PriceUSD_coinmetrics": [100.0, 101.0],
            "mvrv_zscore": [0.0, 0.1],
        },
        index=idx,
    )
    btc_df = pd.DataFrame({"PriceUSD_coinmetrics": [100.0, 101.0]}, index=idx)
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
            btc_price_col="PriceUSD_coinmetrics",
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
            btc_price_col="PriceUSD_coinmetrics",
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

    df = pd.DataFrame(
        {
            "id": [1],
            "start_date": ["2024-01-01"],
            "end_date": ["2024-12-31"],
            "DCA_date": ["2024-01-01"],
            "btc_usd": [50000.0],
            "weight": [1.0],
        }
    )

    updated = update_today_weights(conn, df, today_str="2024-01-01")

    assert updated == 1
    assert observed["previous_price"] is None
