from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


def _load_fetch_module():
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "fetch_brk_data.py"
    spec = importlib.util.spec_from_file_location("fetch_brk_data_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_manifest(
    path: Path,
    *,
    duckdb_payload: bytes,
    schema_payload: bytes,
    duckdb_file_id: str = "duckdb_file_id_123",
    schema_file_id: str = "schema_file_id_123",
) -> None:
    manifest = {
        "gdrive_folder_url": (
            "https://drive.google.com/drive/folders/"
            "1SvAwcdegMzgPANM4pnuTH_9DbNEyXt8N?usp=drive_link"
        ),
        "duckdb": {
            "name": "bitcoin_analytics.duckdb",
            "file_id": duckdb_file_id,
            "sha256": _sha256(duckdb_payload),
            "size_bytes": len(duckdb_payload),
            "version": "test",
        },
        "schema": {
            "name": "bitcoin-analytics-duckdb-schema.md",
            "file_id": schema_file_id,
            "sha256": _sha256(schema_payload),
            "size_bytes": len(schema_payload),
            "version": "test",
        },
        "updated_at_utc": "2026-03-08T21:00:00Z",
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_load_manifest_validates_required_keys(tmp_path: Path) -> None:
    mod = _load_fetch_module()
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        duckdb_payload=b"duckdb-bytes",
        schema_payload=b"schema-bytes",
    )

    manifest = mod.load_manifest(manifest_path)
    assert manifest.duckdb.name == "bitcoin_analytics.duckdb"
    assert manifest.schema.name == "bitcoin-analytics-duckdb-schema.md"

    broken = json.loads(manifest_path.read_text(encoding="utf-8"))
    broken.pop("schema")
    manifest_path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(mod.ManifestError, match="missing top-level keys"):
        mod.load_manifest(manifest_path)


def test_load_manifest_rejects_missing_drive_metadata(tmp_path: Path) -> None:
    mod = _load_fetch_module()
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        duckdb_payload=b"a",
        schema_payload=b"b",
        duckdb_file_id="REPLACE_WITH_DUCKDB_FILE_ID",
    )

    with pytest.raises(mod.ManifestError, match="missing or placeholder"):
        mod.load_manifest(manifest_path)


def test_load_manifest_does_not_reject_valid_file_id_with_todo_substring(tmp_path: Path) -> None:
    mod = _load_fetch_module()
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        duckdb_payload=b"x",
        schema_payload=b"y",
        duckdb_file_id="1a2b3cTodoLikeButValidId",
    )

    manifest = mod.load_manifest(manifest_path)
    assert manifest.duckdb.file_id == "1a2b3cTodoLikeButValidId"


def test_fetch_assets_downloads_and_verifies_hashes(tmp_path: Path) -> None:
    mod = _load_fetch_module()
    manifest_path = tmp_path / "manifest.json"
    duckdb_payload = b"D" * 128
    schema_payload = b"S" * 64
    _write_manifest(
        manifest_path,
        duckdb_payload=duckdb_payload,
        schema_payload=schema_payload,
    )

    payload_by_id = {
        "duckdb_file_id_123": duckdb_payload,
        "schema_file_id_123": schema_payload,
    }

    def fake_downloader(file_id: str, output_path: Path) -> int:
        data = payload_by_id[file_id]
        output_path.write_bytes(data)
        return len(data)

    target_dir = tmp_path / "target"
    schema_dir = tmp_path / "schema"
    duckdb_path, schema_path = mod.fetch_assets(
        manifest_path=manifest_path,
        target_dir=target_dir,
        schema_dir=schema_dir,
        downloader=fake_downloader,
    )

    assert duckdb_path.read_bytes() == duckdb_payload
    assert schema_path.read_bytes() == schema_payload


def test_fetch_assets_fails_on_checksum_mismatch(tmp_path: Path) -> None:
    mod = _load_fetch_module()
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        duckdb_payload=b"duckdb-expected",
        schema_payload=b"schema-expected",
    )

    payload_by_id = {
        "duckdb_file_id_123": b"duckdb-expected",
        "schema_file_id_123": b"schema-expectee",
    }

    def fake_downloader(file_id: str, output_path: Path) -> int:
        data = payload_by_id[file_id]
        output_path.write_bytes(data)
        return len(data)

    with pytest.raises(mod.DownloadError, match="SHA-256 mismatch"):
        mod.fetch_assets(
            manifest_path=manifest_path,
            target_dir=tmp_path / "target",
            schema_dir=tmp_path / "schema",
            downloader=fake_downloader,
        )


def test_existing_file_skip_and_overwrite_behavior(tmp_path: Path) -> None:
    mod = _load_fetch_module()
    manifest_path = tmp_path / "manifest.json"
    duckdb_payload = b"duckdb-good"
    schema_payload = b"schema-good"
    _write_manifest(
        manifest_path,
        duckdb_payload=duckdb_payload,
        schema_payload=schema_payload,
    )

    target_dir = tmp_path / "target"
    schema_dir = tmp_path / "schema"
    target_dir.mkdir(parents=True, exist_ok=True)
    schema_dir.mkdir(parents=True, exist_ok=True)
    duckdb_path = target_dir / "bitcoin_analytics.duckdb"
    schema_path = schema_dir / "bitcoin-analytics-duckdb-schema.md"
    duckdb_path.write_bytes(duckdb_payload)
    schema_path.write_bytes(schema_payload)

    calls: list[str] = []

    def fake_downloader(file_id: str, output_path: Path) -> int:
        calls.append(file_id)
        output_path.write_bytes(b"x")
        return 1

    mod.fetch_assets(
        manifest_path=manifest_path,
        target_dir=target_dir,
        schema_dir=schema_dir,
        downloader=fake_downloader,
    )
    assert calls == []

    duckdb_path.write_bytes(b"bad")
    with pytest.raises(mod.DownloadError, match="Pass --overwrite"):
        mod.fetch_assets(
            manifest_path=manifest_path,
            target_dir=target_dir,
            schema_dir=schema_dir,
            downloader=fake_downloader,
        )

    def overwrite_downloader(file_id: str, output_path: Path) -> int:
        if file_id == "duckdb_file_id_123":
            output_path.write_bytes(duckdb_payload)
            return len(duckdb_payload)
        output_path.write_bytes(schema_payload)
        return len(schema_payload)

    mod.fetch_assets(
        manifest_path=manifest_path,
        target_dir=target_dir,
        schema_dir=schema_dir,
        overwrite=True,
        downloader=overwrite_downloader,
    )
    assert duckdb_path.read_bytes() == duckdb_payload


def test_main_prints_export_command_and_writes_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = _load_fetch_module()
    manifest_path = tmp_path / "manifest.json"
    duckdb_payload = b"D" * 32
    schema_payload = b"S" * 32
    _write_manifest(
        manifest_path,
        duckdb_payload=duckdb_payload,
        schema_payload=schema_payload,
    )

    payload_by_id = {
        "duckdb_file_id_123": duckdb_payload,
        "schema_file_id_123": schema_payload,
    }

    def fake_download(file_id: str, output_path: Path) -> int:
        data = payload_by_id[file_id]
        output_path.write_bytes(data)
        return len(data)

    monkeypatch.setattr(mod, "_download_from_gdrive", fake_download)

    target_dir = tmp_path / "target"
    schema_dir = tmp_path / "schema"
    exit_code = mod.main(
        [
            "--manifest",
            str(manifest_path),
            "--target-dir",
            str(target_dir),
            "--schema-dir",
            str(schema_dir),
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "export STACKSATS_ANALYTICS_DUCKDB=" in output
    assert str((target_dir / "bitcoin_analytics.duckdb").resolve()) in output
    assert (target_dir / "bitcoin_analytics.duckdb").read_bytes() == duckdb_payload
    assert (schema_dir / "bitcoin-analytics-duckdb-schema.md").read_bytes() == schema_payload


def test_main_returns_1_on_manifest_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load_fetch_module()
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        duckdb_payload=b"x",
        schema_payload=b"y",
        duckdb_file_id="REPLACE_WITH_DUCKDB_FILE_ID",
    )

    exit_code = mod.main(["--manifest", str(manifest_path)])
    err = capsys.readouterr().err
    assert exit_code == 1
    assert "[fetch_brk_data] ERROR:" in err
