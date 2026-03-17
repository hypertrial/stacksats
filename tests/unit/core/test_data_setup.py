from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from stacksats import data_setup


def test_load_packaged_manifest_has_real_drive_asset() -> None:
    manifest = data_setup.load_manifest()
    assert manifest.parquet.source == "gdrive"
    assert manifest.parquet.file_id == "1jKRRU7l9kOMdGI_hIJGg02X3jWTMPJsw"
    assert manifest.schema.source == "packaged"
    assert manifest.schema.resource == "merged-metrics-parquet-schema.md"


def test_packaged_demo_parquet_path_exists() -> None:
    with data_setup.packaged_demo_parquet_path() as path:
        assert path.exists()
        frame = pl.read_parquet(path)
    assert {"date", "price_usd", "mvrv"}.issubset(set(frame.columns))
    assert frame.height > 3000


def test_resolve_runtime_parquet_prefers_env_over_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / "env.parquet"
    explicit_path = tmp_path / "explicit.parquet"
    pl.DataFrame({"date": ["2024-01-01"], "price_usd": [1.0]}).write_parquet(env_path)
    pl.DataFrame({"date": ["2024-01-01"], "price_usd": [2.0]}).write_parquet(explicit_path)
    managed_path = tmp_path / "managed.parquet"
    monkeypatch.setattr(data_setup, "MANAGED_RUNTIME_PARQUET", managed_path)
    monkeypatch.setenv("STACKSATS_ANALYTICS_PARQUET", str(env_path))

    resolved = data_setup.resolve_runtime_parquet(str(explicit_path))
    assert resolved.path == env_path
    assert resolved.source == "env STACKSATS_ANALYTICS_PARQUET"


def test_resolve_runtime_parquet_uses_managed_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    managed_path = tmp_path / "bitcoin_analytics.parquet"
    pl.DataFrame({"date": ["2024-01-01"], "price_usd": [1.0]}).write_parquet(managed_path)
    monkeypatch.delenv("STACKSATS_ANALYTICS_PARQUET", raising=False)
    monkeypatch.setattr(data_setup, "MANAGED_RUNTIME_PARQUET", managed_path)

    resolved = data_setup.resolve_runtime_parquet(None)
    assert resolved.path == managed_path
    assert resolved.source == "managed default"


def test_prepare_runtime_parquet_projects_long_format(tmp_path: Path) -> None:
    source = tmp_path / "merged_metrics.parquet"
    output = tmp_path / "bitcoin_analytics.parquet"
    rows = [
        {"day_utc": "2024-01-01", "metric": "market_cap", "value": 100.0},
        {"day_utc": "2024-01-01", "metric": "supply_btc", "value": 2.0},
        {"day_utc": "2024-01-01", "metric": "mvrv", "value": 1.1},
        {"day_utc": "2024-01-01", "metric": "adjusted_sopr", "value": 1.0},
        {"day_utc": "2024-01-01", "metric": "adjusted_sopr_7d_ema", "value": 1.0},
        {"day_utc": "2024-01-01", "metric": "realized_cap_growth_rate", "value": 0.1},
        {"day_utc": "2024-01-01", "metric": "market_cap_growth_rate", "value": 0.2},
    ]
    pl.DataFrame(rows).write_parquet(source)

    prepared = data_setup.prepare_runtime_parquet(source, output=output)
    frame = pl.read_parquet(prepared)
    assert prepared == output.resolve()
    assert {"date", "price_usd", "mvrv"}.issubset(set(frame.columns))
    assert float(frame["price_usd"][0]) == 50.0


def test_data_doctor_reports_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("STACKSATS_ANALYTICS_PARQUET", raising=False)
    monkeypatch.setattr(data_setup, "MANAGED_RUNTIME_PARQUET", tmp_path / "missing.parquet")
    report = data_setup.data_doctor()
    assert report["status"] == "missing"
    assert "next_steps" in report


def test_data_doctor_reports_ok(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pq_path = tmp_path / "bitcoin_analytics.parquet"
    pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "price_usd": [1.0, 2.0],
            "mvrv": [1.1, 1.2],
        }
    ).write_parquet(pq_path)
    monkeypatch.setenv("STACKSATS_ANALYTICS_PARQUET", str(pq_path))

    report = data_setup.data_doctor()
    assert report["status"] == "ok"
    assert report["resolved_path"] == str(pq_path)
    assert report["coverage_start"] == "2024-01-01"
    assert report["coverage_end"] == "2024-01-02"


def test_data_doctor_reports_daily_gaps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pq_path = tmp_path / "bitcoin_analytics.parquet"
    pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-03"],
            "price_usd": [1.0, 2.0],
        }
    ).write_parquet(pq_path)
    monkeypatch.setenv("STACKSATS_ANALYTICS_PARQUET", str(pq_path))

    report = data_setup.data_doctor()
    assert report["status"] == "warning"
    assert report["has_daily_gaps"] is True
    assert report["gap_count"] == 1
    assert report["first_gap_after"] == "2024-01-01"
    assert report["first_gap_before"] == "2024-01-03"
