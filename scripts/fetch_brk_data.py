#!/usr/bin/env python3
"""Download BRK DuckDB + schema assets from Google Drive with checksum validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import hashlib
import http.cookiejar
import json
import re
import shutil
import sys
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = ROOT / "data" / "brk_data_manifest.json"
DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id={file_id}"
REQUEST_TIMEOUT_SECONDS = 60

PLACEHOLDER_EXACT = {"TODO", "TBD", "REPLACE"}


class ManifestError(ValueError):
    """Raised when the data manifest is malformed."""


class DownloadError(RuntimeError):
    """Raised when an asset download or verification fails."""


@dataclass(frozen=True)
class AssetSpec:
    name: str
    file_id: str
    sha256: str
    size_bytes: int
    version: str


@dataclass(frozen=True)
class DataManifest:
    gdrive_folder_url: str
    duckdb: AssetSpec
    schema: AssetSpec
    updated_at_utc: str


def _is_placeholder(text: str) -> bool:
    value = text.strip()
    upper = value.upper()
    if "<" in value or ">" in value:
        return True
    if upper in PLACEHOLDER_EXACT:
        return True
    return upper.startswith("REPLACE_WITH_")


def _parse_asset(name: str, payload: object) -> AssetSpec:
    if not isinstance(payload, dict):
        raise ManifestError(f"Manifest field '{name}' must be an object.")

    required = ("name", "file_id", "sha256", "size_bytes", "version")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ManifestError(f"Manifest field '{name}' missing keys: {', '.join(missing)}.")

    asset_name = str(payload["name"]).strip()
    file_id = str(payload["file_id"]).strip()
    sha256 = str(payload["sha256"]).strip().lower()
    version = str(payload["version"]).strip()

    if not asset_name:
        raise ManifestError(f"Manifest field '{name}.name' must be non-empty.")
    if not file_id or _is_placeholder(file_id):
        raise ManifestError(
            f"Manifest field '{name}.file_id' is missing or placeholder. "
            "Set it to a real Google Drive file id."
        )
    if not re.fullmatch(r"[a-f0-9]{64}", sha256):
        raise ManifestError(f"Manifest field '{name}.sha256' must be a 64-char hex digest.")

    size_bytes = payload["size_bytes"]
    if not isinstance(size_bytes, int) or size_bytes <= 0:
        raise ManifestError(f"Manifest field '{name}.size_bytes' must be a positive integer.")
    if not version:
        raise ManifestError(f"Manifest field '{name}.version' must be non-empty.")

    return AssetSpec(
        name=asset_name,
        file_id=file_id,
        sha256=sha256,
        size_bytes=size_bytes,
        version=version,
    )


def load_manifest(path: Path) -> DataManifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ManifestError("Manifest root must be an object.")

    required = ("gdrive_folder_url", "duckdb", "schema", "updated_at_utc")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ManifestError(f"Manifest missing top-level keys: {', '.join(missing)}.")

    folder_url = str(raw["gdrive_folder_url"]).strip()
    updated = str(raw["updated_at_utc"]).strip()
    if not folder_url.startswith("https://drive.google.com/drive/folders/"):
        raise ManifestError("Manifest field 'gdrive_folder_url' must be a Google Drive folder URL.")
    if not updated:
        raise ManifestError("Manifest field 'updated_at_utc' must be non-empty.")

    return DataManifest(
        gdrive_folder_url=folder_url,
        duckdb=_parse_asset("duckdb", raw["duckdb"]),
        schema=_parse_asset("schema", raw["schema"]),
        updated_at_utc=updated,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stream_response_to_file(response, output: Path) -> int:
    total = 0
    with output.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            handle.write(chunk)
    return total


def _extract_confirm_token(html: str) -> str | None:
    match = re.search(r"[?&]confirm=([0-9A-Za-z_-]+)", html)
    return match.group(1) if match else None


def _download_from_gdrive(file_id: str, output_path: Path) -> int:
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))

    first_url = DOWNLOAD_URL.format(file_id=urllib.parse.quote(file_id, safe=""))
    with opener.open(first_url, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        content_type = (response.headers.get("Content-Type") or "").lower()
        if "text/html" in content_type:
            html = response.read().decode("utf-8", errors="ignore")
            token = _extract_confirm_token(html)
            if not token:
                raise DownloadError(
                    "Google Drive returned HTML without a download token. "
                    "Check file sharing permissions and file_id."
                )
            confirm_url = (
                "https://drive.google.com/uc?export=download"
                f"&confirm={urllib.parse.quote(token, safe='')}"
                f"&id={urllib.parse.quote(file_id, safe='')}"
            )
            with opener.open(confirm_url, timeout=REQUEST_TIMEOUT_SECONDS) as second:
                return _stream_response_to_file(second, output_path)

        return _stream_response_to_file(response, output_path)


def _download_and_verify(
    asset: AssetSpec,
    destination: Path,
    *,
    overwrite: bool,
    downloader,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not overwrite:
        existing_sha = _sha256(destination)
        if existing_sha != asset.sha256:
            raise DownloadError(
                f"{destination} exists but hash does not match manifest. "
                "Pass --overwrite to replace it."
            )
        existing_size = destination.stat().st_size
        if existing_size != asset.size_bytes:
            raise DownloadError(
                f"{destination} exists but size does not match manifest. "
                "Pass --overwrite to replace it."
            )
        print(f"[fetch_brk_data] Existing file verified, skipping: {destination}")
        return

    tmp = destination.with_suffix(destination.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    try:
        bytes_written = downloader(asset.file_id, tmp)
        if bytes_written <= 0:
            raise DownloadError(f"Download produced no bytes for {asset.name}.")

        actual_size = tmp.stat().st_size
        if actual_size != asset.size_bytes:
            raise DownloadError(
                f"Size mismatch for {asset.name}: expected {asset.size_bytes}, got {actual_size}."
            )

        actual_sha = _sha256(tmp)
        if actual_sha != asset.sha256:
            raise DownloadError(
                f"SHA-256 mismatch for {asset.name}: expected {asset.sha256}, got {actual_sha}."
            )

        shutil.move(str(tmp), str(destination))
        print(f"[fetch_brk_data] Downloaded and verified: {destination}")
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def fetch_assets(
    *,
    manifest_path: Path,
    target_dir: Path,
    schema_dir: Path,
    overwrite: bool = False,
    downloader=None,
) -> tuple[Path, Path]:
    if downloader is None:
        downloader = _download_from_gdrive

    manifest = load_manifest(manifest_path)

    duckdb_path = (target_dir / manifest.duckdb.name).resolve()
    schema_path = (schema_dir / manifest.schema.name).resolve()

    _download_and_verify(
        manifest.duckdb,
        duckdb_path,
        overwrite=overwrite,
        downloader=downloader,
    )
    _download_and_verify(
        manifest.schema,
        schema_path,
        overwrite=overwrite,
        downloader=downloader,
    )

    print(f"[fetch_brk_data] Source folder: {manifest.gdrive_folder_url}")
    print(f"[fetch_brk_data] Manifest updated_at_utc: {manifest.updated_at_utc}")
    print(f"export STACKSATS_ANALYTICS_DUCKDB={duckdb_path}")

    return duckdb_path, schema_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch BRK DuckDB and schema assets from Google Drive using manifest checksums."
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Path to data manifest JSON.",
    )
    parser.add_argument(
        "--target-dir",
        default=".",
        help="Directory where the DuckDB file will be written.",
    )
    parser.add_argument(
        "--schema-dir",
        default="docs/reference",
        help="Directory where the schema markdown file will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files (required when local files differ from manifest).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = Path(args.manifest).expanduser().resolve()
    target_dir = Path(args.target_dir).expanduser().resolve()
    schema_dir = Path(args.schema_dir).expanduser().resolve()

    try:
        fetch_assets(
            manifest_path=manifest_path,
            target_dir=target_dir,
            schema_dir=schema_dir,
            overwrite=bool(args.overwrite),
        )
        return 0
    except (ManifestError, DownloadError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"[fetch_brk_data] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
