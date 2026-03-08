from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from stacksats.feature_providers import (
    DUCKDB_ANALYTICS_METRIC_FAMILIES,
    DuckDBAnalyticsFeatureProvider,
)


def _btc_df(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price_usd": np.linspace(10000.0, 20000.0, len(index)),
        },
        index=index,
    )


def _create_duckdb_fixture(path: Path, *, mutate_last: bool = False) -> None:
    con = duckdb.connect(str(path))
    try:
        for table_name, metrics in DUCKDB_ANALYTICS_METRIC_FAMILIES.items():
            con.execute(
                f"CREATE TABLE {table_name} (date_day DATE, metric VARCHAR, value DOUBLE)"
            )
            rows: list[tuple[object, str, float]] = []
            dates = pd.date_range("2024-01-01", periods=25, freq="D")
            for day_idx, date in enumerate(dates):
                for metric_idx, metric_name in enumerate(metrics):
                    value = float((day_idx + 1) * (metric_idx + 2))
                    if (
                        mutate_last
                        and date == dates[-1]
                        and table_name == "metrics_cointime"
                        and metric_name == metrics[0]
                    ):
                        value = value * 10_000.0
                    rows.append((date.date(), metric_name, value))
            con.executemany(
                f"INSERT INTO {table_name} VALUES (?, ?, ?)",
                rows,
            )
    finally:
        con.close()


def test_duckdb_provider_missing_path_fails(tmp_path: Path) -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    provider = DuckDBAnalyticsFeatureProvider(
        default_duckdb_path=str(tmp_path / "missing.duckdb")
    )
    with pytest.raises(ValueError, match="DuckDB file not found"):
        provider.materialize(
            _btc_df(idx),
            start_date=idx[0],
            end_date=idx[-1],
            as_of_date=idx[-1],
        )


def test_duckdb_provider_materialize_is_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "analytics.duckdb"
    _create_duckdb_fixture(db_path)
    monkeypatch.setenv("STACKSATS_ANALYTICS_DUCKDB", str(db_path))
    provider = DuckDBAnalyticsFeatureProvider()

    idx = pd.date_range("2024-01-01", periods=20, freq="D")
    btc_df = _btc_df(idx)
    first = provider.materialize(
        btc_df,
        start_date=idx[3],
        end_date=idx[-1],
        as_of_date=idx[-2],
    )
    second = provider.materialize(
        btc_df,
        start_date=idx[3],
        end_date=idx[-1],
        as_of_date=idx[-2],
    )

    assert first.equals(second)
    assert first.index.min() == idx[3]
    assert first.index.max() == idx[-2]
    assert "ddb_cointime_reserve_risk_level" in first.columns
    assert "ddb_cross_reserve_drawdown" in first.columns
    assert np.isfinite(first.to_numpy(dtype=float)).all()
    assert float(np.abs(first.to_numpy(dtype=float)).max()) <= provider.max_abs_feature


def test_duckdb_provider_is_as_of_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stable_db = tmp_path / "stable.duckdb"
    revised_db = tmp_path / "revised.duckdb"
    _create_duckdb_fixture(stable_db, mutate_last=False)
    _create_duckdb_fixture(revised_db, mutate_last=True)

    idx = pd.date_range("2024-01-01", periods=20, freq="D")
    btc_df = _btc_df(idx)
    as_of = idx[-2]

    monkeypatch.setenv("STACKSATS_ANALYTICS_DUCKDB", str(stable_db))
    stable = DuckDBAnalyticsFeatureProvider().materialize(
        btc_df,
        start_date=idx[0],
        end_date=idx[-1],
        as_of_date=as_of,
    )
    monkeypatch.setenv("STACKSATS_ANALYTICS_DUCKDB", str(revised_db))
    revised = DuckDBAnalyticsFeatureProvider().materialize(
        btc_df,
        start_date=idx[0],
        end_date=idx[-1],
        as_of_date=as_of,
    )
    assert stable.equals(revised)


def test_duckdb_provider_missing_required_table_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "partial.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE TABLE metrics_market (date_day DATE, metric VARCHAR, value DOUBLE)")
        con.execute(
            "INSERT INTO metrics_market VALUES ('2024-01-01', 'price_drawdown', 0.1)"
        )
    finally:
        con.close()

    monkeypatch.setenv("STACKSATS_ANALYTICS_DUCKDB", str(db_path))
    provider = DuckDBAnalyticsFeatureProvider()
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    with pytest.raises(ValueError, match="missing required DuckDB tables"):
        provider.materialize(
            _btc_df(idx),
            start_date=idx[0],
            end_date=idx[-1],
            as_of_date=idx[-1],
        )
