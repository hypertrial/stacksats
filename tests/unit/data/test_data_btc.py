from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from stacksats.data_btc import BTCDataProvider, DataLoadError


def _create_parquet_fixture(
    path: Path,
    *,
    start: str = "2024-01-01",
    days: int = 10,
    include_mvrv: bool = True,
    skip_day: str | None = None,
    null_price_day: str | None = None,
) -> None:
    idx = pd.date_range(start, periods=days, freq="D")
    rows = []
    for i, ts in enumerate(idx):
        day = ts.strftime("%Y-%m-%d")
        if skip_day is not None and day == skip_day:
            continue
        price_val: float | None = float(40_000 + (i * 100))
        if null_price_day is not None and day == null_price_day:
            price_val = None
        row: dict[str, object] = {"date": day, "price_usd": price_val}
        if include_mvrv:
            row["mvrv"] = 1.2 + (0.01 * i)
        rows.append(row)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index("date")
    df.to_parquet(path)


def test_load_success_from_parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pq_path = tmp_path / "bitcoin_analytics.parquet"
    _create_parquet_fixture(pq_path, start="2024-01-01", days=5)
    monkeypatch.setenv("STACKSATS_ANALYTICS_PARQUET", str(pq_path))
    provider = BTCDataProvider(clock=lambda: dt.datetime(2024, 1, 5))

    df = provider.load(backtest_start="2024-01-01", end_date="2024-01-05")

    assert "price_usd" in df.columns
    assert "mvrv" in df.columns
    assert "date" in df.columns
    assert df["date"].min() is not None
    assert str(df["date"].min())[:10] == "2024-01-01"
    assert str(df["date"].max())[:10] == "2024-01-05"
    last_row = df.sort("date").tail(1)
    assert last_row.height == 1 and float(last_row["price_usd"][0]) > 0.0


def test_load_missing_parquet_file_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STACKSATS_ANALYTICS_PARQUET", "/tmp/does-not-exist.parquet")
    provider = BTCDataProvider(clock=lambda: dt.datetime(2024, 1, 5))
    with pytest.raises(DataLoadError, match="Parquet file not found"):
        provider.load(backtest_start="2024-01-01")


def test_load_empty_parquet_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pq_path = tmp_path / "empty.parquet"
    pd.DataFrame(columns=["date", "price_usd"]).to_parquet(pq_path)
    monkeypatch.setenv("STACKSATS_ANALYTICS_PARQUET", str(pq_path))
    provider = BTCDataProvider(clock=lambda: dt.datetime(2024, 1, 5))
    with pytest.raises(DataLoadError, match="Parquet file is empty"):
        provider.load(backtest_start="2024-01-01")


def test_load_parquet_missing_price_column_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pq_path = tmp_path / "no_price.parquet"
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3, freq="D"), "mvrv": [1.0, 1.1, 1.2]})
    df = df.set_index("date")
    df.to_parquet(pq_path)
    monkeypatch.setenv("STACKSATS_ANALYTICS_PARQUET", str(pq_path))
    provider = BTCDataProvider(clock=lambda: dt.datetime(2024, 1, 3))
    with pytest.raises(DataLoadError, match="must contain a 'price_usd' column"):
        provider.load(backtest_start="2024-01-01", end_date="2024-01-03")


def test_load_stale_data_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pq_path = tmp_path / "stale.parquet"
    _create_parquet_fixture(pq_path, start="2024-01-01", days=3)
    monkeypatch.setenv("STACKSATS_ANALYTICS_PARQUET", str(pq_path))
    provider = BTCDataProvider(clock=lambda: dt.datetime(2024, 2, 20), max_staleness_days=3)
    with pytest.raises(DataLoadError, match="does not cover requested end_date"):
        provider.load(backtest_start="2024-01-01", end_date="2024-02-20")


def test_load_missing_dates_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pq_path = tmp_path / "gappy.parquet"
    # only write 4 days, skip 2024-01-03
    rows = [
        {"date": "2024-01-01", "price_usd": 40000.0, "mvrv": 1.2},
        {"date": "2024-01-02", "price_usd": 40100.0, "mvrv": 1.21},
        {"date": "2024-01-04", "price_usd": 40300.0, "mvrv": 1.23},
        {"date": "2024-01-05", "price_usd": 40400.0, "mvrv": 1.24},
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index("date")
    df.to_parquet(pq_path)
    monkeypatch.setenv("STACKSATS_ANALYTICS_PARQUET", str(pq_path))
    provider = BTCDataProvider(clock=lambda: dt.datetime(2024, 1, 5))
    with pytest.raises(DataLoadError, match="missing dates"):
        provider.load(backtest_start="2024-01-01", end_date="2024-01-05")


def test_load_missing_price_values_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pq_path = tmp_path / "null-price.parquet"
    rows = [
        {"date": "2024-01-01", "price_usd": 40000.0, "mvrv": 1.2},
        {"date": "2024-01-02", "price_usd": 40100.0, "mvrv": 1.21},
        {"date": "2024-01-03", "price_usd": 40200.0, "mvrv": 1.22},
        {"date": "2024-01-04", "price_usd": None, "mvrv": 1.23},
        {"date": "2024-01-05", "price_usd": 40400.0, "mvrv": 1.24},
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index("date")
    df.to_parquet(pq_path)
    monkeypatch.setenv("STACKSATS_ANALYTICS_PARQUET", str(pq_path))
    provider = BTCDataProvider(clock=lambda: dt.datetime(2024, 1, 5))
    with pytest.raises(DataLoadError, match="missing price_usd values"):
        provider.load(backtest_start="2024-01-01", end_date="2024-01-05")
