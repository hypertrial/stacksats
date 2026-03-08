from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from stacksats.data_btc import BTCDataProvider, DataLoadError


def _create_duckdb_fixture(
    path: Path,
    *,
    start: str = "2024-01-01",
    days: int = 10,
    include_mvrv: bool = True,
    skip_day: str | None = None,
    null_price_day: str | None = None,
) -> None:
    con = duckdb.connect(str(path))
    try:
        con.execute("CREATE TABLE metrics_price (date_day DATE, metric VARCHAR, value DOUBLE)")
        con.execute("CREATE TABLE metrics_distribution (date_day DATE, metric VARCHAR, value DOUBLE)")
        idx = pd.date_range(start, periods=days, freq="D")
        price_rows: list[tuple[str, str, float | None]] = []
        mvrv_rows: list[tuple[str, str, float]] = []
        for i, ts in enumerate(idx):
            day = ts.strftime("%Y-%m-%d")
            if skip_day is not None and day == skip_day:
                continue
            price_val: float | None = float(40_000 + (i * 100))
            if null_price_day is not None and day == null_price_day:
                price_val = None
            price_rows.append((day, "price_close", price_val))
            if include_mvrv:
                mvrv_rows.append((day, "mvrv", float(1.2 + (0.01 * i))))
        con.executemany("INSERT INTO metrics_price VALUES (?, ?, ?)", price_rows)
        if include_mvrv and mvrv_rows:
            con.executemany("INSERT INTO metrics_distribution VALUES (?, ?, ?)", mvrv_rows)
    finally:
        con.close()


def test_load_success_from_duckdb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "bitcoin_analytics.duckdb"
    _create_duckdb_fixture(db_path, start="2024-01-01", days=5)
    monkeypatch.setenv("STACKSATS_ANALYTICS_DUCKDB", str(db_path))
    provider = BTCDataProvider(clock=lambda: pd.Timestamp("2024-01-05"))

    df = provider.load(backtest_start="2024-01-01", end_date="2024-01-05")

    assert list(df.columns) == ["price_usd", "mvrv"]
    assert df.index.min() == pd.Timestamp("2024-01-01")
    assert df.index.max() == pd.Timestamp("2024-01-05")
    assert float(df.loc[pd.Timestamp("2024-01-05"), "price_usd"]) > 0.0


def test_load_missing_duckdb_file_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STACKSATS_ANALYTICS_DUCKDB", "/tmp/does-not-exist.duckdb")
    provider = BTCDataProvider(clock=lambda: pd.Timestamp("2024-01-05"))
    with pytest.raises(DataLoadError, match="DuckDB file not found"):
        provider.load(backtest_start="2024-01-01")


def test_load_missing_required_table_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "broken.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE TABLE metrics_price (date_day DATE, metric VARCHAR, value DOUBLE)")
        con.execute("INSERT INTO metrics_price VALUES ('2024-01-01', 'price_close', 42000.0)")
    finally:
        con.close()
    monkeypatch.setenv("STACKSATS_ANALYTICS_DUCKDB", str(db_path))
    provider = BTCDataProvider(clock=lambda: pd.Timestamp("2024-01-05"))
    with pytest.raises(DataLoadError, match="Could not query required BRK tables/metrics"):
        provider.load(backtest_start="2024-01-01")


def test_load_stale_data_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "stale.duckdb"
    _create_duckdb_fixture(db_path, start="2024-01-01", days=3)
    monkeypatch.setenv("STACKSATS_ANALYTICS_DUCKDB", str(db_path))
    provider = BTCDataProvider(clock=lambda: pd.Timestamp("2024-02-20"), max_staleness_days=3)
    with pytest.raises(DataLoadError, match="does not cover requested end_date"):
        provider.load(backtest_start="2024-01-01", end_date="2024-02-20")


def test_load_missing_dates_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "gappy.duckdb"
    _create_duckdb_fixture(db_path, start="2024-01-01", days=5, skip_day="2024-01-03")
    monkeypatch.setenv("STACKSATS_ANALYTICS_DUCKDB", str(db_path))
    provider = BTCDataProvider(clock=lambda: pd.Timestamp("2024-01-05"))
    with pytest.raises(DataLoadError, match="missing dates"):
        provider.load(backtest_start="2024-01-01", end_date="2024-01-05")


def test_load_missing_price_values_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "null-price.duckdb"
    _create_duckdb_fixture(
        db_path,
        start="2024-01-01",
        days=5,
        null_price_day="2024-01-04",
    )
    monkeypatch.setenv("STACKSATS_ANALYTICS_DUCKDB", str(db_path))
    provider = BTCDataProvider(clock=lambda: pd.Timestamp("2024-01-05"))
    with pytest.raises(DataLoadError, match="missing price_usd values"):
        provider.load(backtest_start="2024-01-01", end_date="2024-01-05")
