from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from stacksats.data_btc import BTCDataProvider, DataLoadError, _resolve_duckdb_path


def _write_duckdb(
    path: Path,
    *,
    prices: list[tuple[str, float | None]],
    mvrv: list[tuple[str, float | None]] | None = None,
) -> None:
    con = duckdb.connect(str(path))
    try:
        con.execute("CREATE TABLE metrics_price (date_day DATE, metric VARCHAR, value DOUBLE)")
        con.execute("CREATE TABLE metrics_distribution (date_day DATE, metric VARCHAR, value DOUBLE)")
        if prices:
            con.executemany(
                "INSERT INTO metrics_price VALUES (?, 'price_close', ?)",
                prices,
            )
        if mvrv is not None:
            con.executemany(
                "INSERT INTO metrics_distribution VALUES (?, 'mvrv', ?)",
                mvrv,
            )
    finally:
        con.close()


def test_resolve_duckdb_path_uses_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "fixture.duckdb"
    _write_duckdb(db_path, prices=[("2024-01-01", 40000.0)], mvrv=[("2024-01-01", 2.0)])
    monkeypatch.delenv("STACKSATS_ANALYTICS_DUCKDB", raising=False)

    resolved = _resolve_duckdb_path(str(db_path))

    assert resolved == db_path


def test_load_rejects_invalid_end_date_format(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "fixture.duckdb"
    _write_duckdb(db_path, prices=[("2024-01-01", 40000.0)], mvrv=[("2024-01-01", 2.0)])
    monkeypatch.setenv("STACKSATS_ANALYTICS_DUCKDB", str(db_path))
    provider = BTCDataProvider(clock=lambda: pd.Timestamp("2024-01-01"))

    with pytest.raises(ValueError, match="Invalid end_date value"):
        provider.load(backtest_start="2024-01-01", end_date="not-a-date")


def test_load_rejects_end_date_before_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "fixture.duckdb"
    _write_duckdb(db_path, prices=[("2024-01-01", 40000.0)], mvrv=[("2024-01-01", 2.0)])
    monkeypatch.setenv("STACKSATS_ANALYTICS_DUCKDB", str(db_path))
    provider = BTCDataProvider(clock=lambda: pd.Timestamp("2024-01-10"))

    with pytest.raises(ValueError, match="end_date must be on or after backtest_start"):
        provider.load(backtest_start="2024-01-10", end_date="2024-01-01")


def test_load_fails_when_price_metric_rows_are_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "no-price.duckdb"
    _write_duckdb(db_path, prices=[], mvrv=[("2024-01-01", 2.0)])
    monkeypatch.setenv("STACKSATS_ANALYTICS_DUCKDB", str(db_path))
    provider = BTCDataProvider(clock=lambda: pd.Timestamp("2024-01-01"))

    with pytest.raises(DataLoadError, match="No rows found for metrics_price.price_close"):
        provider.load(backtest_start="2024-01-01")


def test_load_fails_when_all_price_values_are_nan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "all-nan.duckdb"
    _write_duckdb(
        db_path,
        prices=[("2024-01-01", None), ("2024-01-02", None)],
        mvrv=[("2024-01-01", 2.0), ("2024-01-02", 2.1)],
    )
    monkeypatch.setenv("STACKSATS_ANALYTICS_DUCKDB", str(db_path))
    provider = BTCDataProvider(clock=lambda: pd.Timestamp("2024-01-02"))

    with pytest.raises(DataLoadError, match="contains no valid price_usd values"):
        provider.load(backtest_start="2024-01-01", end_date="2024-01-02")


def test_load_fails_when_data_stale_but_end_date_covered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "stale-but-covered.duckdb"
    _write_duckdb(
        db_path,
        prices=[("2024-01-01", 40000.0), ("2024-01-02", 40100.0)],
        mvrv=[("2024-01-01", 2.0), ("2024-01-02", 2.1)],
    )
    monkeypatch.setenv("STACKSATS_ANALYTICS_DUCKDB", str(db_path))
    provider = BTCDataProvider(
        clock=lambda: pd.Timestamp("2024-01-10"),
        max_staleness_days=3,
    )

    with pytest.raises(DataLoadError, match="BRK data is stale for runtime usage"):
        provider.load(backtest_start="2024-01-01", end_date="2024-01-02")


def test_load_rejects_backtest_start_after_end_date_when_clamped_by_now(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "window-empty.duckdb"
    _write_duckdb(
        db_path,
        prices=[("2024-01-01", 40000.0), ("2024-01-02", 40100.0)],
        mvrv=[("2024-01-01", 2.0), ("2024-01-02", 2.1)],
    )
    monkeypatch.setenv("STACKSATS_ANALYTICS_DUCKDB", str(db_path))
    provider = BTCDataProvider(clock=lambda: pd.Timestamp("2024-01-02"))

    with pytest.raises(ValueError, match="end_date must be on or after backtest_start"):
        provider.load(backtest_start="2024-01-03", end_date="2024-01-03")
