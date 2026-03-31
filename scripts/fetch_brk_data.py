#!/usr/bin/env python3
"""Thin wrapper around stacksats.data.data_setup fetch helpers."""

from __future__ import annotations

from pathlib import Path
import argparse
from json import JSONDecodeError
import sys

from stacksats.data.data_setup import (
    DownloadError,
    MANAGED_BRK_DIR,
    ManifestError,
    _download_from_gdrive,
    fetch_assets,
    load_manifest,
)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download canonical BRK assets with checksum validation.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional explicit BRK manifest JSON path. Defaults to the packaged manifest.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=MANAGED_BRK_DIR,
        help="Destination directory for downloaded parquet.",
    )
    parser.add_argument(
        "--schema-dir",
        type=Path,
        default=None,
        help="Optional destination directory for schema markdown.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files even when they already exist.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        parquet_path, schema_path = fetch_assets(
            manifest_path=args.manifest,
            target_dir=args.target_dir,
            schema_dir=args.schema_dir,
            overwrite=args.overwrite,
            downloader=_download_from_gdrive,
        )
    except (ManifestError, DownloadError, FileNotFoundError, JSONDecodeError) as exc:
        print(f"[fetch_brk_data] ERROR: {exc}", file=sys.stderr)
        return 1

    manifest = load_manifest(args.manifest)
    print(f"[fetch_brk_data] Source folder: {manifest.gdrive_folder_url}")
    print(f"[fetch_brk_data] Manifest updated_at_utc: {manifest.updated_at_utc}")
    print(f"[fetch_brk_data] Schema asset: {schema_path}")
    print(f"export STACKSATS_ANALYTICS_PARQUET={parquet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
