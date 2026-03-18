from __future__ import annotations

import json
import time
from pathlib import Path

import polars as pl
import pytest

from stacksats import data_setup


def _valid_manifest_base() -> dict:
    return {
        "gdrive_folder_url": "https://drive.google.com/drive/folders/abc",
        "parquet": {
            "name": "test.parquet",
            "source": "gdrive",
            "file_id": "valid_file_id_123",
            "sha256": "a" * 64,
            "size_bytes": 100,
            "version": "1.0",
        },
        "schema": {
            "name": "schema.md",
            "source": "packaged",
            "resource": "merged-metrics-parquet-schema.md",
            "sha256": "b" * 64,
            "size_bytes": 50,
            "version": "1.0",
        },
        "updated_at_utc": "2026-01-01T00:00:00Z",
    }


def test_is_placeholder_covers_all_branches() -> None:
    """_is_placeholder covers REPLACE_WITH_, angle brackets, and non-placeholder branches."""
    assert data_setup._is_placeholder("<REPLACE_WITH_ID>") is True
    assert data_setup._is_placeholder("REPLACE_WITH_PARQUET_ID") is True
    assert data_setup._is_placeholder("TODO") is True
    assert data_setup._is_placeholder("valid_file_id_123") is False


def test_load_manifest_rejects_non_dict_root(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(data_setup.ManifestError, match="Manifest root must be an object"):
        data_setup.load_manifest(tmp_path / "manifest.json")


def test_load_manifest_rejects_missing_top_level_keys(tmp_path: Path) -> None:
    base = _valid_manifest_base()
    base.pop("schema")
    (tmp_path / "manifest.json").write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(data_setup.ManifestError, match="missing top-level keys"):
        data_setup.load_manifest(tmp_path / "manifest.json")


def test_load_manifest_rejects_invalid_gdrive_url(tmp_path: Path) -> None:
    base = _valid_manifest_base()
    base["gdrive_folder_url"] = "https://example.com/not-google-drive"
    (tmp_path / "manifest.json").write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(data_setup.ManifestError, match="gdrive_folder_url"):
        data_setup.load_manifest(tmp_path / "manifest.json")


def test_load_manifest_rejects_empty_updated_at_utc(tmp_path: Path) -> None:
    base = _valid_manifest_base()
    base["updated_at_utc"] = ""
    (tmp_path / "manifest.json").write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(data_setup.ManifestError, match="updated_at_utc"):
        data_setup.load_manifest(tmp_path / "manifest.json")


def test_load_manifest_rejects_source_cannot_be_inferred(tmp_path: Path) -> None:
    base = _valid_manifest_base()
    base["parquet"]["source"] = None
    del base["parquet"]["file_id"]
    base["parquet"]["sha256"] = "a" * 64
    base["parquet"]["size_bytes"] = 100
    base["parquet"]["version"] = "1.0"
    base["parquet"]["name"] = "x.parquet"
    (tmp_path / "manifest.json").write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(data_setup.ManifestError, match="could not be inferred"):
        data_setup.load_manifest(tmp_path / "manifest.json")


def test_load_manifest_rejects_parquet_non_object(tmp_path: Path) -> None:
    base = _valid_manifest_base()
    base["parquet"] = "not an object"
    (tmp_path / "manifest.json").write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(data_setup.ManifestError, match="parquet.*must be an object"):
        data_setup.load_manifest(tmp_path / "manifest.json")


def test_load_manifest_rejects_parquet_missing_keys(tmp_path: Path) -> None:
    base = _valid_manifest_base()
    base["parquet"] = {"name": "x", "sha256": "a" * 64}
    (tmp_path / "manifest.json").write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(data_setup.ManifestError, match="missing keys"):
        data_setup.load_manifest(tmp_path / "manifest.json")


def test_load_manifest_rejects_gdrive_placeholder_file_id(tmp_path: Path) -> None:
    base = _valid_manifest_base()
    base["parquet"]["file_id"] = "REPLACE_WITH_PARQUET_FILE_ID"
    (tmp_path / "manifest.json").write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(data_setup.ManifestError, match="placeholder"):
        data_setup.load_manifest(tmp_path / "manifest.json")


def test_load_manifest_rejects_gdrive_missing_file_id(tmp_path: Path) -> None:
    base = _valid_manifest_base()
    base["parquet"]["source"] = "gdrive"
    del base["parquet"]["file_id"]
    (tmp_path / "manifest.json").write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(data_setup.ManifestError, match="file_id.*required"):
        data_setup.load_manifest(tmp_path / "manifest.json")


def test_load_manifest_rejects_packaged_missing_resource(tmp_path: Path) -> None:
    base = _valid_manifest_base()
    base["schema"]["source"] = "packaged"
    del base["schema"]["resource"]
    (tmp_path / "manifest.json").write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(data_setup.ManifestError, match="resource.*required"):
        data_setup.load_manifest(tmp_path / "manifest.json")


def test_load_manifest_rejects_invalid_sha256(tmp_path: Path) -> None:
    base = _valid_manifest_base()
    base["parquet"]["sha256"] = "not-64-hex-chars"
    (tmp_path / "manifest.json").write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(data_setup.ManifestError, match="sha256"):
        data_setup.load_manifest(tmp_path / "manifest.json")


def test_load_manifest_rejects_invalid_size_bytes(tmp_path: Path) -> None:
    base = _valid_manifest_base()
    base["parquet"]["size_bytes"] = 0
    (tmp_path / "manifest.json").write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(data_setup.ManifestError, match="size_bytes"):
        data_setup.load_manifest(tmp_path / "manifest.json")


def test_load_manifest_rejects_invalid_source(tmp_path: Path) -> None:
    base = _valid_manifest_base()
    base["parquet"]["source"] = "s3"
    (tmp_path / "manifest.json").write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(data_setup.ManifestError, match="gdrive.*packaged"):
        data_setup.load_manifest(tmp_path / "manifest.json")


def test_load_manifest_rejects_empty_asset_name(tmp_path: Path) -> None:
    base = _valid_manifest_base()
    base["parquet"]["name"] = ""
    (tmp_path / "manifest.json").write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(data_setup.ManifestError, match="name.*non-empty"):
        data_setup.load_manifest(tmp_path / "manifest.json")


def test_load_manifest_rejects_empty_version(tmp_path: Path) -> None:
    base = _valid_manifest_base()
    base["parquet"]["version"] = ""
    (tmp_path / "manifest.json").write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(data_setup.ManifestError, match="version.*non-empty"):
        data_setup.load_manifest(tmp_path / "manifest.json")


def test_load_manifest_rejects_packaged_empty_resource(tmp_path: Path) -> None:
    base = _valid_manifest_base()
    base["schema"]["resource"] = ""
    (tmp_path / "manifest.json").write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(data_setup.ManifestError, match="resource.*non-empty"):
        data_setup.load_manifest(tmp_path / "manifest.json")


def test_load_manifest_valid_with_explicit_path(tmp_path: Path) -> None:
    base = _valid_manifest_base()
    (tmp_path / "manifest.json").write_text(json.dumps(base), encoding="utf-8")
    manifest = data_setup.load_manifest(tmp_path / "manifest.json")
    assert manifest.parquet.source == "gdrive"
    assert manifest.schema.source == "packaged"
    assert manifest.parquet.file_id == "valid_file_id_123"


def test_load_packaged_manifest_has_real_drive_asset() -> None:
    manifest = data_setup.load_manifest()
    assert manifest.parquet.source == "gdrive"
    assert manifest.parquet.file_id == "1jKRRU7l9kOMdGI_hIJGg02X3jWTMPJsw"
    assert manifest.schema.source == "packaged"
    assert manifest.schema.resource == "merged-metrics-parquet-schema.md"


def test_packaged_bytes_returns_binary_asset() -> None:
    """packaged_bytes reads a packaged binary asset."""
    raw = data_setup.packaged_bytes("merged-metrics-parquet-schema.md")
    assert isinstance(raw, bytes)
    assert len(raw) > 0
    assert b"#" in raw or b"schema" in raw.lower()


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


def _sha256(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


def test_fetch_assets_with_explicit_schema_dir(tmp_path: Path) -> None:
    """fetch_assets uses schema_dir when it differs from target_dir."""
    manifest_path = tmp_path / "manifest.json"
    parquet_payload = b"P" * 64
    schema_payload = b"S" * 32
    manifest = {
        "gdrive_folder_url": "https://drive.google.com/drive/folders/abc",
        "parquet": {
            "name": "data.parquet",
            "file_id": "f1",
            "sha256": _sha256(parquet_payload),
            "size_bytes": len(parquet_payload),
            "version": "1",
        },
        "schema": {
            "name": "schema.md",
            "file_id": "f2",
            "sha256": _sha256(schema_payload),
            "size_bytes": len(schema_payload),
            "version": "1",
        },
        "updated_at_utc": "2026-01-01T00:00:00Z",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def fake_dl(fid: str, out: Path) -> int:
        data = parquet_payload if fid == "f1" else schema_payload
        out.write_bytes(data)
        return len(data)

    target_dir = tmp_path / "target"
    schema_dir = tmp_path / "schema"
    p, s = data_setup.fetch_assets(
        manifest_path=manifest_path,
        target_dir=target_dir,
        schema_dir=schema_dir,
        downloader=fake_dl,
    )
    assert p == target_dir / "data.parquet"
    assert s == schema_dir / "schema.md"
    assert s.read_bytes() == schema_payload


def test_fetch_assets_downloader_returns_zero_raises(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    payload = b"x" * 16
    manifest = {
        "gdrive_folder_url": "https://drive.google.com/drive/folders/abc",
        "parquet": {
            "name": "x.parquet",
            "file_id": "f1",
            "sha256": _sha256(payload),
            "size_bytes": len(payload),
            "version": "1",
        },
        "schema": {
            "name": "x.md",
            "file_id": "f2",
            "sha256": _sha256(payload),
            "size_bytes": len(payload),
            "version": "1",
        },
        "updated_at_utc": "2026-01-01T00:00:00Z",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def zero_dl(_fid: str, out: Path) -> int:
        out.write_bytes(b"")
        return 0

    with pytest.raises(data_setup.DownloadError, match="no bytes"):
        data_setup.fetch_assets(
            manifest_path=manifest_path,
            target_dir=tmp_path / "target",
            downloader=zero_dl,
        )


def test_fetch_assets_size_mismatch_raises(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    payload = b"x" * 16
    manifest = {
        "gdrive_folder_url": "https://drive.google.com/drive/folders/abc",
        "parquet": {
            "name": "x.parquet",
            "file_id": "f1",
            "sha256": _sha256(payload),
            "size_bytes": 100,
            "version": "1",
        },
        "schema": {
            "name": "x.md",
            "file_id": "f2",
            "sha256": _sha256(payload),
            "size_bytes": len(payload),
            "version": "1",
        },
        "updated_at_utc": "2026-01-01T00:00:00Z",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def wrong_size_dl(fid: str, out: Path) -> int:
        data = payload if fid == "f1" else payload
        out.write_bytes(data)
        return len(data)

    with pytest.raises(data_setup.DownloadError, match="Size mismatch"):
        data_setup.fetch_assets(
            manifest_path=manifest_path,
            target_dir=tmp_path / "target",
            downloader=wrong_size_dl,
        )


def test_fetch_assets_sha_mismatch_raises(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    expected = b"expected" * 8
    wrong = b"wrong____" * 8
    wrong = wrong[: len(expected)]
    manifest = {
        "gdrive_folder_url": "https://drive.google.com/drive/folders/abc",
        "parquet": {
            "name": "x.parquet",
            "file_id": "f1",
            "sha256": _sha256(expected),
            "size_bytes": len(expected),
            "version": "1",
        },
        "schema": {
            "name": "x.md",
            "file_id": "f2",
            "sha256": _sha256(expected),
            "size_bytes": len(expected),
            "version": "1",
        },
        "updated_at_utc": "2026-01-01T00:00:00Z",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def wrong_sha_dl(fid: str, out: Path) -> int:
        out.write_bytes(wrong)
        return len(wrong)

    with pytest.raises(data_setup.DownloadError, match="SHA-256 mismatch"):
        data_setup.fetch_assets(
            manifest_path=manifest_path,
            target_dir=tmp_path / "target",
            downloader=wrong_sha_dl,
        )


def test_fetch_assets_existing_file_skip_when_match(tmp_path: Path) -> None:
    """Existing file that matches manifest is skipped (no downloader call)."""
    schema_content = data_setup.packaged_bytes("merged-metrics-parquet-schema.md")
    manifest_path = tmp_path / "manifest.json"
    parquet_payload = b"P" * 64
    manifest = {
        "gdrive_folder_url": "https://drive.google.com/drive/folders/abc",
        "parquet": {
            "name": "x.parquet",
            "file_id": "f1",
            "sha256": _sha256(parquet_payload),
            "size_bytes": len(parquet_payload),
            "version": "1",
        },
        "schema": {
            "name": "merged-metrics-parquet-schema.md",
            "source": "packaged",
            "resource": "merged-metrics-parquet-schema.md",
            "sha256": _sha256(schema_content),
            "size_bytes": len(schema_content),
            "version": "1",
        },
        "updated_at_utc": "2026-01-01T00:00:00Z",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    target_dir = tmp_path / "target"
    target_dir.mkdir()
    parquet_path = target_dir / "x.parquet"
    schema_path = target_dir / "merged-metrics-parquet-schema.md"
    parquet_path.write_bytes(parquet_payload)
    schema_path.write_bytes(schema_content)

    calls: list[str] = []

    def track_dl(fid: str, out: Path) -> int:
        calls.append(fid)
        out.write_bytes(parquet_payload)
        return len(parquet_payload)

    p, s = data_setup.fetch_assets(
        manifest_path=manifest_path,
        target_dir=target_dir,
        downloader=track_dl,
    )
    assert p == parquet_path.resolve()
    assert s == schema_path.resolve()
    assert calls == []


def test_fetch_assets_existing_file_mismatch_raises(tmp_path: Path) -> None:
    """Existing file that does not match manifest raises unless overwrite."""
    parquet_payload = b"P" * 64
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "gdrive_folder_url": "https://drive.google.com/drive/folders/abc",
        "parquet": {
            "name": "x.parquet",
            "file_id": "f1",
            "sha256": _sha256(parquet_payload),
            "size_bytes": len(parquet_payload),
            "version": "1",
        },
        "schema": {
            "name": "x.md",
            "file_id": "f2",
            "sha256": _sha256(parquet_payload),
            "size_bytes": len(parquet_payload),
            "version": "1",
        },
        "updated_at_utc": "2026-01-01T00:00:00Z",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    target_dir = tmp_path / "target"
    target_dir.mkdir()
    (target_dir / "x.parquet").write_bytes(b"wrong")
    (target_dir / "x.md").write_bytes(b"wrong")

    def fake_dl(_fid: str, out: Path) -> int:
        out.write_bytes(parquet_payload)
        return len(parquet_payload)

    with pytest.raises(data_setup.DownloadError, match="--overwrite"):
        data_setup.fetch_assets(
            manifest_path=manifest_path,
            target_dir=target_dir,
            downloader=fake_dl,
        )


def test_fetch_assets_packaged_schema_uses_copy(tmp_path: Path) -> None:
    """Schema with source=packaged uses _copy_packaged_asset (no downloader)."""
    schema_content = data_setup.packaged_bytes("merged-metrics-parquet-schema.md")
    parquet_payload = b"P" * 64
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "gdrive_folder_url": "https://drive.google.com/drive/folders/abc",
        "parquet": {
            "name": "x.parquet",
            "file_id": "f1",
            "sha256": _sha256(parquet_payload),
            "size_bytes": len(parquet_payload),
            "version": "1",
        },
        "schema": {
            "name": "merged-metrics-parquet-schema.md",
            "source": "packaged",
            "resource": "merged-metrics-parquet-schema.md",
            "sha256": _sha256(schema_content),
            "size_bytes": len(schema_content),
            "version": "1",
        },
        "updated_at_utc": "2026-01-01T00:00:00Z",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def parquet_only_dl(fid: str, out: Path) -> int:
        if fid != "f1":
            raise RuntimeError("Schema should use packaged copy, not downloader")
        out.write_bytes(parquet_payload)
        return len(parquet_payload)

    target_dir = tmp_path / "target"
    p, s = data_setup.fetch_assets(
        manifest_path=manifest_path,
        target_dir=target_dir,
        downloader=parquet_only_dl,
    )
    assert s.read_bytes() == schema_content
    assert p.read_bytes() == parquet_payload


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


def test_resolve_runtime_parquet_raises_when_none_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("STACKSATS_ANALYTICS_PARQUET", raising=False)
    monkeypatch.setattr(data_setup, "MANAGED_RUNTIME_PARQUET", tmp_path / "missing.parquet")
    with pytest.raises(FileNotFoundError, match="No runtime parquet could be resolved"):
        data_setup.resolve_runtime_parquet(None)


def test_project_runtime_parquet_canonical_long_format(tmp_path: Path) -> None:
    """Directly exercise project_runtime_parquet with day_utc/metric/value schema."""
    source = tmp_path / "canonical.parquet"
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
    frame = data_setup.project_runtime_parquet(source)
    assert {"date", "price_usd", "mvrv"}.issubset(set(frame.columns))
    assert float(frame["price_usd"][0]) == 50.0


def test_project_runtime_parquet_unsupported_schema_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.parquet"
    pl.DataFrame({"col_a": [1], "col_b": [2]}).write_parquet(bad)
    with pytest.raises(ValueError, match="Unsupported source parquet"):
        data_setup.project_runtime_parquet(bad)


def test_prepare_runtime_parquet_from_wide_format(tmp_path: Path) -> None:
    """Exercise project_runtime_parquet with date/price_usd schema (wide format)."""
    source = tmp_path / "wide.parquet"
    output = tmp_path / "out.parquet"
    pl.DataFrame({"date": ["2024-01-01", "2024-01-02"], "price_usd": [1.0, 2.0]}).write_parquet(
        source
    )
    result = data_setup.prepare_runtime_parquet(source, output=output)
    assert result == output.resolve()
    frame = pl.read_parquet(result)
    assert frame.height == 2
    assert "date" in frame.columns and "price_usd" in frame.columns


def test_prepare_runtime_parquet_raises_when_source_missing(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent.parquet"
    output = tmp_path / "out.parquet"
    with pytest.raises(FileNotFoundError, match="Source parquet not found"):
        data_setup.prepare_runtime_parquet(missing, output=output)


def test_prepare_runtime_parquet_raises_when_output_exists(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    output = tmp_path / "output.parquet"
    pl.DataFrame({"date": ["2024-01-01"], "price_usd": [1.0]}).write_parquet(source)
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="already exists"):
        data_setup.prepare_runtime_parquet(source, output=output, overwrite=False)


def test_latest_fetched_parquet_returns_newest(tmp_path: Path) -> None:
    """Exercise latest_fetched_parquet success path."""
    tmp_path.mkdir(exist_ok=True)
    older = tmp_path / "a.parquet"
    newer = tmp_path / "b.parquet"
    pl.DataFrame({"day_utc": [], "metric": [], "value": []}).write_parquet(older)
    time.sleep(0.02)  # Ensure newer has later mtime on fast systems
    pl.DataFrame({"day_utc": [], "metric": [], "value": []}).write_parquet(newer)
    result = data_setup.latest_fetched_parquet(tmp_path)
    assert result == newer.resolve()


def test_latest_fetched_parquet_raises_when_empty(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(FileNotFoundError, match="No canonical parquet found"):
        data_setup.latest_fetched_parquet(tmp_path)


def test_data_doctor_reports_invalid_when_parquet_bad(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bad = tmp_path / "bad.parquet"
    pl.DataFrame({"x": [1]}).write_parquet(bad)
    monkeypatch.setenv("STACKSATS_ANALYTICS_PARQUET", str(bad))
    report = data_setup.data_doctor()
    assert report["status"] == "invalid"
    assert "error" in report


def test_data_doctor_empty_frame_reports_none_coverage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    empty = tmp_path / "empty.parquet"
    pl.DataFrame({"date": [], "price_usd": []}).write_parquet(empty)
    monkeypatch.setenv("STACKSATS_ANALYTICS_PARQUET", str(empty))
    report = data_setup.data_doctor()
    assert report["status"] == "ok"
    assert report["coverage_start"] is None
    assert report["coverage_end"] is None
