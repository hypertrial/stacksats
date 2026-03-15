from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from stacksats.data_btc import BTCDataProvider, DataLoadError, _resolve_parquet_path


def _write_parquet(
    path: Path,
    *,
    prices: list[tuple[str, float | None]],
    mvrv: list[tuple[str, float | None]] | None = None,
) -> None:
    rows = [{"date": d, "price_usd": p} for d, p in prices]
    if mvrv is not None:
        mvrv_map = dict(mvrv)
        for r in rows:
            r["mvrv"] = mvrv_map.get(r["date"])
    pl.DataFrame(rows).write_parquet(path)


def test_resolve_parquet_path_uses_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pq_path = tmp_path / "fixture.parquet"
    _write_parquet(pq_path, prices=[("2024-01-01", 40000.0)], mvrv=[("2024-01-01", 2.0)])
    monkeypatch.delenv("STACKSATS_ANALYTICS_PARQUET", raising=False)

    resolved = _resolve_parquet_path(str(pq_path))

    assert resolved == pq_path


def test_load_rejects_invalid_end_date_format(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pq_path = tmp_path / "fixture.parquet"
    _write_parquet(pq_path, prices=[("2024-01-01", 40000.0)], mvrv=[("2024-01-01", 2.0)])
    monkeypatch.setenv("STACKSATS_ANALYTICS_PARQUET", str(pq_path))
    provider = BTCDataProvider(clock=lambda: dt.datetime(2024, 1, 1))

    with pytest.raises(ValueError, match="Invalid end_date value"):
        provider.load(backtest_start="2024-01-01", end_date="not-a-date")


def test_load_rejects_end_date_before_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pq_path = tmp_path / "fixture.parquet"
    _write_parquet(pq_path, prices=[("2024-01-01", 40000.0)], mvrv=[("2024-01-01", 2.0)])
    monkeypatch.setenv("STACKSATS_ANALYTICS_PARQUET", str(pq_path))
    provider = BTCDataProvider(clock=lambda: dt.datetime(2024, 1, 10))

    with pytest.raises(ValueError, match="end_date must be on or after backtest_start"):
        provider.load(backtest_start="2024-01-10", end_date="2024-01-01")


def test_load_fails_when_price_rows_are_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pq_path = tmp_path / "no-price.parquet"
    pl.DataFrame(schema={"date": pl.Datetime("us"), "price_usd": pl.Float64}).write_parquet(pq_path)
    monkeypatch.setenv("STACKSATS_ANALYTICS_PARQUET", str(pq_path))
    provider = BTCDataProvider(clock=lambda: dt.datetime(2024, 1, 1))

    with pytest.raises(DataLoadError, match="Parquet file is empty"):
        provider.load(backtest_start="2024-01-01")


def test_load_fails_when_all_price_values_are_nan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pq_path = tmp_path / "all-nan.parquet"
    _write_parquet(
        pq_path,
        prices=[("2024-01-01", None), ("2024-01-02", None)],
        mvrv=[("2024-01-01", 2.0), ("2024-01-02", 2.1)],
    )
    monkeypatch.setenv("STACKSATS_ANALYTICS_PARQUET", str(pq_path))
    provider = BTCDataProvider(clock=lambda: dt.datetime(2024, 1, 2))

    with pytest.raises(DataLoadError, match="contains no valid price_usd values"):
        provider.load(backtest_start="2024-01-01", end_date="2024-01-02")


def test_load_fails_when_data_stale_but_end_date_covered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pq_path = tmp_path / "stale-but-covered.parquet"
    _write_parquet(
        pq_path,
        prices=[("2024-01-01", 40000.0), ("2024-01-02", 40100.0)],
        mvrv=[("2024-01-01", 2.0), ("2024-01-02", 2.1)],
    )
    monkeypatch.setenv("STACKSATS_ANALYTICS_PARQUET", str(pq_path))
    provider = BTCDataProvider(
        clock=lambda: dt.datetime(2024, 1, 10),
        max_staleness_days=3,
    )

    with pytest.raises(DataLoadError, match="BRK data is stale for runtime usage"):
        provider.load(backtest_start="2024-01-01", end_date="2024-01-02")


def test_load_rejects_backtest_start_after_end_date_when_clamped_by_now(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pq_path = tmp_path / "window-empty.parquet"
    _write_parquet(
        pq_path,
        prices=[("2024-01-01", 40000.0), ("2024-01-02", 40100.0)],
        mvrv=[("2024-01-01", 2.0), ("2024-01-02", 2.1)],
    )
    monkeypatch.setenv("STACKSATS_ANALYTICS_PARQUET", str(pq_path))
    provider = BTCDataProvider(clock=lambda: dt.datetime(2024, 1, 2))

    with pytest.raises(ValueError, match="end_date must be on or after backtest_start"):
        provider.load(backtest_start="2024-01-03", end_date="2024-01-03")
