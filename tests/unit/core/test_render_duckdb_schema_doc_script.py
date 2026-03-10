from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import duckdb


def _build_fixture_db(path: Path) -> None:
    con = duckdb.connect(str(path))
    try:
        con.execute(
            """
            CREATE TABLE _long_load_runs (
              run_id VARCHAR PRIMARY KEY,
              source_path VARCHAR NOT NULL,
              start_date DATE NOT NULL,
              status VARCHAR NOT NULL,
              error_message VARCHAR,
              started_at TIMESTAMP NOT NULL,
              finished_at TIMESTAMP
            )
            """
        )
        con.execute(
            """
            CREATE TABLE metrics_price (
              date_day DATE NOT NULL,
              metric VARCHAR NOT NULL,
              value DOUBLE
            )
            """
        )
        con.execute(
            """
            CREATE TABLE metrics_distribution (
              date_day DATE NOT NULL,
              metric VARCHAR NOT NULL,
              value DOUBLE
            )
            """
        )
        con.execute(
            """
            INSERT INTO _long_load_runs VALUES
            ('r1', '/tmp/source.csv', DATE '2025-01-01', 'ok', NULL, TIMESTAMP '2025-01-01 00:00:00', NULL)
            """
        )
        con.execute(
            """
            INSERT INTO metrics_price VALUES
            (DATE '2025-01-01', 'price_close', 100.0),
            (DATE '2025-01-02', 'price_close', 101.0)
            """
        )
        con.execute(
            """
            INSERT INTO metrics_distribution VALUES
            (DATE '2025-01-01', 'mvrv', 1.1),
            (DATE '2025-01-02', 'mvrv', 1.2)
            """
        )
    finally:
        con.close()


def _write_manifest(path: Path) -> None:
    payload = {
        "gdrive_folder_url": (
            "https://drive.google.com/drive/folders/"
            "1SvAwcdegMzgPANM4pnuTH_9DbNEyXt8N?usp=drive_link"
        )
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_render_duckdb_schema_doc_script_generates_and_checks(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "render_duckdb_schema_doc.py"
    db_path = tmp_path / "fixture.duckdb"
    manifest_path = tmp_path / "manifest.json"
    output_path = tmp_path / "schema.md"

    _build_fixture_db(db_path)
    _write_manifest(manifest_path)

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--duckdb-path",
            str(db_path),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    content = output_path.read_text(encoding="utf-8")
    assert "# Bitcoin Analytics DuckDB Schema" in content
    assert "metrics_price" in content
    assert "google.com/drive/folders" in content
    assert "Regenerate this page" in content

    check_result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--duckdb-path",
            str(db_path),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--check",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert check_result.returncode == 0, check_result.stderr or check_result.stdout
    assert "up to date" in check_result.stdout

    output_path.write_text(content + "\n<!-- drift -->\n", encoding="utf-8")
    stale_result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--duckdb-path",
            str(db_path),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--check",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert stale_result.returncode == 1
    assert "out of date" in stale_result.stdout
